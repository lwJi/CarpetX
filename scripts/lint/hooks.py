from __future__ import annotations

import json
import os
import sys

from boundary.doc import MAP
from boundary.guide import GUIDE
from checks import CHECKS
from files import PROJECT, covered, merge_base, read_text, repo_name, simulate_edit
from run import check_edit, check_since, has_findings, render_change

GENERATED = {
    MAP: f"{MAP} is written by python3 scripts/lint doc --write from scripts/lint/boundary/rules.py, doc.py and symbols.py: change those and run it.",
    GUIDE: f"{GUIDE} is written by python3 scripts/lint doc --write from the rows of scripts/lint/boundary/guide.py, TestCarpetXAPI/param.ccl and TestCarpetXAPI/test/errors: change those and run it.",
}


def emit(out, message: dict) -> None:
    out.write(json.dumps(message) + "\n")


def skipped(out, e: Exception) -> int:
    emit(out, {"systemMessage": f"lint check skipped: {type(e).__name__}: {e}"})
    return 0


def hook_edit(raw, out=None, root: str = PROJECT) -> int:
    out = out or sys.stdout
    try:
        checks = CHECKS()
        payload = json.loads(raw)
        tool_name = payload["tool_name"]
        tool_input = payload["tool_input"]
        path = os.path.join(payload.get("cwd") or os.getcwd(), tool_input["file_path"])
        name = repo_name(path, root)
        if name in GENERATED:
            return deny(out, GENERATED[name])
        lang = covered(name, root) if name is not None else None
        if lang is None:
            return 0
        before = read_text(path) if os.path.isfile(path) else None
        after = simulate_edit(tool_name, tool_input, before)
        if after is None:
            return 0
        found = check_edit(root, name, before or "", after, lang, checks)
        if not has_findings(found):
            return 0
        reason = render_change(found, edit=True)
    except Exception as e:
        return skipped(out, e)
    return deny(out, reason)


def deny(out, reason: str) -> int:
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
        checks = CHECKS()
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise ValueError("the hook payload is not a JSON object")
        if payload.get("stop_hook_active"):
            return 0
        found = check_since(root, merge_base(root), checks)
        if not has_findings(found):
            return 0
        reason = render_change(found)
    except Exception as e:
        return skipped(out, e)
    emit(out, {"decision": "block", "reason": reason})
    return 0
