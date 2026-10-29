from __future__ import annotations

import json
import os
import sys

from rules import CHECKED, findings, new_findings, render, since_findings
from files import PROJECT, covered, merge_base, read_text, repo_name, simulate_edit


def emit(out, message: dict) -> None:
    out.write(json.dumps(message) + "\n")


def skipped(out, e: Exception) -> int:
    emit(out, {"systemMessage": f"boundary check skipped: {type(e).__name__}: {e}"})
    return 0


def hook_edit(raw, out=None, root: str = PROJECT) -> int:
    out = out or sys.stdout
    try:
        payload = json.loads(raw)
        tool_name = payload["tool_name"]
        tool_input = payload["tool_input"]
        path = os.path.join(payload.get("cwd") or os.getcwd(), tool_input["file_path"])
        name = repo_name(path, root)
        lang = covered(name, root) if name is not None else None
        if lang not in CHECKED:
            return 0
        before = read_text(path) if os.path.isfile(path) else None
        after = simulate_edit(tool_name, tool_input, before)
        if after is None:
            return 0
        new = new_findings(findings(name, before or "", lang), findings(name, after, lang))
        if not new:
            return 0
        reason = render([(name, f) for f in new])
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
        found = since_findings(root, merge_base(root))
        if not found:
            return 0
        reason = render(found)
    except Exception as e:
        return skipped(out, e)
    emit(out, {"decision": "block", "reason": reason})
    return 0
