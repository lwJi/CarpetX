import json
import os
import subprocess
import sys
from collections import Counter

from files import Lang, covered, list_covered, read_text

TOOL = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(os.path.dirname(TOOL))


def emit(out, message: dict) -> None:
    print(json.dumps(message), file=out or sys.stdout)


def skipped(out, e: Exception) -> None:
    emit(out, {"systemMessage": f"comment check skipped: {type(e).__name__}: {e}"})


def repo_name(path: str, root: str) -> str | None:
    rel = os.path.relpath(os.path.realpath(path), os.path.realpath(root))
    if rel == os.pardir or rel.startswith(os.pardir + os.sep):
        return None
    return rel.replace(os.sep, "/")


def ignored(root: str, name: str) -> bool:
    probe = subprocess.run(["git", "-C", root, "check-ignore", "-q", "--", name], capture_output=True)
    return probe.returncode == 0


def post_edit_text(tool_name: str, tool_input: dict, before: str | None) -> str | None:
    if tool_name == "Write":
        return tool_input["content"]
    if tool_name != "Edit":
        raise ValueError(f"unexpected tool {tool_name!r}")
    old, new = tool_input["old_string"], tool_input["new_string"]
    replace_all = bool(tool_input.get("replace_all", False))
    if before is None:
        return new if old == "" else None
    count = before.count(old)
    if count == 0 or (count > 1 and not replace_all):
        return None
    return before.replace(old, new) if replace_all else before.replace(old, new, 1)


def _key(v) -> tuple[str, ...]:
    return tuple(line.strip() for line in v.group.text.split("\n"))


def added_violations(before: str | None, after: str, lang: Lang) -> list:
    from rules import check

    found = check(after, lang)
    if not found:
        return []
    remaining = Counter(_key(v) for v in check(before, lang)) if before else Counter()
    added = []
    for v in found:
        if remaining[_key(v)]:
            remaining[_key(v)] -= 1
        else:
            added.append(v)
    return added


def hook_edit(raw: str | bytes, out=None, root: str = PROJECT) -> int:
    try:
        payload = json.loads(raw)
        tool_input = payload["tool_input"]
        path = os.path.join(payload.get("cwd") or os.getcwd(), tool_input["file_path"])
        name = repo_name(path, root)
        lang = covered(name) if name is not None else None
        if lang is None or ignored(root, name):
            return 0
        before = read_text(path) if os.path.isfile(path) else None
        after = post_edit_text(payload.get("tool_name"), tool_input, before)
        if after is None:
            return 0
        found = added_violations(before, after, lang)
        if not found:
            return 0
        from report import report

        reason = report([(name, v) for v in found])
    except Exception as e:
        skipped(out, e)
        return 0
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


def hook_stop(raw: str | bytes, out=None, root: str = PROJECT) -> int:
    try:
        payload = json.loads(raw)
        if payload.get("stop_hook_active"):
            return 0
        from report import report
        from rules import check

        found = []
        for f in list_covered([root], untracked=True):
            found.extend((f.name, v) for v in check(read_text(f.path), f.lang))
        if not found:
            return 0
        reason = report(found)
    except Exception as e:
        skipped(out, e)
        return 0
    emit(out, {"decision": "block", "reason": reason})
    return 0
