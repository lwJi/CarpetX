from __future__ import annotations

import json
import os
import sys
from collections import Counter
from dataclasses import replace
from typing import Optional

from files import PROJECT, Lang, covered, read_text, repo_name
from lex_clang import ToolMissing
from report import render
from rules import check, lex, may_hold_comments
from tokens import Kind, Token


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


def added(before: str, after: str, lang: Lang) -> list:
    if not may_hold_comments(after, lang):
        return []
    tokens = lex(after, lang)
    found = check(tokens, lang)
    if not found:
        return []
    old = Counter()
    if may_hold_comments(before, lang):
        old.update(comment_key(t) for t in lex(before, lang) if t.kind is Kind.COMMENT)
    flagged = {i for v in found for i in range(v.region.first, v.region.last + 1)}
    comments = [i for i, t in enumerate(tokens) if t.kind is Kind.COMMENT]
    new = set()
    for i in sorted(comments, key=lambda i: (i in flagged, i)):
        key = comment_key(tokens[i])
        if old[key]:
            old[key] -= 1
        else:
            new.add(i)
    result = []
    for v in found:
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
