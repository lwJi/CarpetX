from __future__ import annotations

import json
import os
import sys
from typing import Optional

from files import (
    PROJECT,
    Lang,
    changes_since,
    covered,
    is_regular,
    merge_base,
    merged_tips,
    parallel,
    read_text,
    repo_name,
    show,
)
from judge import Verdict, judge, judge_all
from report import render_change


def emit(out, message: dict) -> None:
    out.write(json.dumps(message) + "\n")


def skipped(out, e: Exception) -> int:
    emit(out, {"systemMessage": f"comment check skipped: {type(e).__name__}: {e}"})
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


def starts(root: str, base: Optional[str] = None) -> list[str]:
    base = merge_base(root) if base is None else base
    return [base] + merged_tips(root, base)


def judge_since(root: str, base: Optional[str] = None) -> list[tuple[str, Verdict]]:
    revs = starts(root, base)
    renames = [changes_since(root, rev) for rev in revs]
    names = sorted(set(renames[0]).intersection(*renames[1:]))
    items = [(name, lang) for name in names for lang in [covered(name, root)] if lang is not None]

    def judged(item: tuple[str, Lang]) -> tuple[str, Verdict]:
        name, lang = item
        path = os.path.join(root, name)
        after = read_text(path) if is_regular(path) else ""
        return name, judge_all((show(root, rev, old[name]) for rev, old in zip(revs, renames)), after, lang)

    return [(name, verdict) for name, verdict in parallel(judged, items) if verdict]


def judge_edit(root: str, name: str, before: str, after: str, lang: Lang) -> Verdict:
    verdict = judge(before, after, lang)
    if not verdict:
        return verdict
    try:
        others = [judge(show(root, rev, changes_since(root, rev).get(name, name)), after, lang) for rev in starts(root)]
    except Exception:
        return verdict
    for other in others:
        verdict = verdict.within(other)
    return verdict


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
        verdict = judge_edit(root, name, before or "", after, lang)
        if not verdict:
            return 0
        reason = render_change([(name, verdict)], edit=True)
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


def hook_stop(raw, out=None, root: str = PROJECT) -> int:
    out = out or sys.stdout
    try:
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise ValueError("the hook payload is not a JSON object")
        if payload.get("stop_hook_active"):
            return 0
        found = judge_since(root)
        if not found:
            return 0
        reason = render_change(found)
    except Exception as e:
        return skipped(out, e)
    emit(out, {"decision": "block", "reason": reason})
    return 0
