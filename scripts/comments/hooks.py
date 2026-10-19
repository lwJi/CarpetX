from __future__ import annotations

import json
import os
import sys
from collections import Counter
from dataclasses import replace
from typing import Optional

from files import PROJECT, Lang, changed, covered, is_regular, merge_base, parallel, read_text, repo_name, show, untracked
from lex_clang import ToolMissing
from report import render
from rules import check, lex, may_hold_comments, regions_of
from tokens import DIRECTIVE_PLACEMENTS, Kind, Region, Token


def emit(out, message: dict) -> None:
    out.write(json.dumps(message) + "\n")


def skipped(out, e: Exception) -> int:
    why = str(e) if isinstance(e, ToolMissing) else f"{type(e).__name__}: {e}"
    emit(out, {"systemMessage": f"comment check skipped: {why}"})
    return 0


def simulate_edit(tool_name: str, tool_input: dict, before: Optional[str]) -> Optional[str]:
    if tool_name == "Write":
        return tool_input["content"]
    if tool_name != "Edit":
        raise ValueError(f"unexpected tool {tool_name!r}")
    old, new = tool_input["old_string"], tool_input["new_string"]
    if old == "":
        return new if not before else None
    if before is None:
        return None
    replace_all = bool(tool_input.get("replace_all", False))
    count = before.count(old)
    if count == 0 or (count > 1 and not replace_all):
        return None
    return before.replace(old, new, -1 if replace_all else 1)


def comment_key(token: Token) -> str:
    return " ".join(token.text.split())


def directive_key(region: Region) -> str:
    return " ".join(t.text for t in region.tokens if t.kind in (Kind.CODE, Kind.DIRECTIVE))


def added(before: str, after: str, lang: Lang) -> list:
    if not may_hold_comments(after, lang):
        return []
    tokens = lex(after, lang)
    found = check(tokens, lang)
    if not found:
        return []
    old_tokens = lex(before, lang) if may_hold_comments(before, lang) else []
    old_comments = Counter(comment_key(t) for t in old_tokens if t.kind is Kind.COMMENT)
    old_directives = Counter(
        directive_key(r) for r in regions_of(old_tokens, lang) if r.placement in DIRECTIVE_PLACEMENTS
    )
    flagged = {
        i
        for v in found
        if v.region.placement not in DIRECTIVE_PLACEMENTS
        for i in range(v.region.first, v.region.last + 1)
    }
    comments = [i for i, t in enumerate(tokens) if t.kind is Kind.COMMENT]
    new = set()
    for i in sorted(comments, key=lambda i: (i in flagged, i)):
        key = comment_key(tokens[i])
        if old_comments[key]:
            old_comments[key] -= 1
        else:
            new.add(i)
    result = []
    for v in found:
        if v.region.placement in DIRECTIVE_PLACEMENTS:
            key = directive_key(v.region)
            if old_directives[key]:
                old_directives[key] -= 1
            else:
                result.append(v)
            continue
        first_new = next((i for i in range(v.region.first, v.region.last + 1) if i in new), None)
        if first_new is not None:
            result.append(replace(v, shown=tokens[first_new]))
    return result


def hook_edit(raw, out=None, root: str = PROJECT) -> int:
    out = out or sys.stdout
    try:
        payload = json.loads(raw)
        tool_name = payload["tool_name"]
        tool_input = payload["tool_input"]
        path = os.path.join(payload.get("cwd") or os.getcwd(), tool_input["file_path"])
        name = repo_name(path, root)
        lang = covered(name, root) if name is not None else None
        if lang is None:
            return 0
        before = read_text(path) if os.path.isfile(path) else None
        after = simulate_edit(tool_name, tool_input, before)
        if after is None:
            return 0
        found = added(before or "", after, lang)
        if not found:
            return 0
        reason = render([(name, v) for v in found])
    except Exception as e:
        return skipped(out, e)
    emit(
        out,
        {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": reason,
            }
        },
    )
    return 0


def branch_changes(root: str) -> list[tuple[str, Optional[str], str, Lang]]:
    base = merge_base(root)
    pairs = {new: old for old, new in changed(root, base)}
    for name in untracked(root):
        pairs.setdefault(name, None)
    found = []
    for name in sorted(pairs):
        path = os.path.join(root, name)
        lang = covered(name, root)
        if lang is not None and is_regular(path):
            found.append((base, pairs[name], name, lang))
    return found


def hook_stop(raw, out=None, root: str = PROJECT) -> int:
    out = out or sys.stdout
    try:
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise ValueError("the hook payload is not a JSON object")
        if payload.get("stop_hook_active"):
            return 0

        def judge(item):
            base, old, name, lang = item
            after = read_text(os.path.join(root, name))
            return [(name, v) for v in added(show(root, base, old), after, lang)]

        found = [entry for entries in parallel(judge, branch_changes(root)) for entry in entries]
        if not found:
            return 0
        reason = render(found)
    except Exception as e:
        return skipped(out, e)
    emit(out, {"decision": "block", "reason": reason})
    return 0
