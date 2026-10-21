from __future__ import annotations

import json
import os
import sys
from collections import Counter
from typing import Iterable, Optional

from files import (
    PROJECT,
    Lang,
    changed,
    covered,
    is_regular,
    merge_base,
    merged_tips,
    parallel,
    read_text,
    repo_name,
    show,
    untracked,
)
from lex_clang import ToolMissing
from report import render
from rules import Unit, units


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


def added(befores: Iterable[str], after: str, lang: Lang) -> list[Unit]:
    mine = units(after, lang)
    if all(u.allowed for u in mine):
        return []
    old: Counter = Counter()
    for text in befores:
        old |= Counter(u.key for u in units(text, lang))
    old -= Counter(u.key for u in mine if u.allowed)
    found = []
    for u in mine:
        if u.allowed:
            continue
        if old[u.key] > 0:
            old[u.key] -= 1
        else:
            found.append(u)
    return found


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
        found = added([before or ""], after, lang)
        if not found:
            return 0
        reason = render([(name, u) for u in found])
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


def branch_changes(root: str, base: Optional[str] = None) -> list[tuple[str, Optional[str], str, Lang]]:
    base = merge_base(root) if base is None else base
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


def added_since(root: str, base: Optional[str] = None) -> list:
    base = merge_base(root) if base is None else base
    tips = merged_tips(root, base)

    def judge(item):
        base, old, name, lang = item
        befores = [show(root, base, old)] + [show(root, tip, name) or show(root, tip, old) for tip in tips]
        after = read_text(os.path.join(root, name))
        return [(name, u) for u in added(befores, after, lang)]

    return [entry for entries in parallel(judge, branch_changes(root, base)) for entry in entries]


def hook_stop(raw, out=None, root: str = PROJECT) -> int:
    out = out or sys.stdout
    try:
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise ValueError("the hook payload is not a JSON object")
        if payload.get("stop_hook_active"):
            return 0
        found = added_since(root)
        if not found:
            return 0
        reason = render(found)
    except Exception as e:
        return skipped(out, e)
    emit(out, {"decision": "block", "reason": reason})
    return 0
