from __future__ import annotations

import json
import os
import sys
from collections import Counter
from typing import Callable, Iterable, Optional

from files import (
    PROJECT,
    Lang,
    covered,
    covered_in,
    differing,
    is_regular,
    merge_base,
    merged_tips,
    parallel,
    read_text,
    repo_name,
    show,
    untracked,
)
from report import render
from rules import Unit, units

Tree = Callable[[str], str]


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


def counts(found: Iterable[Unit]) -> Counter:
    return Counter(u.key for u in found)


def surplus(befores: Iterable[Counter], after: list[Unit]) -> list[int]:
    old: Counter = Counter()
    for count in befores:
        old |= count
    old -= counts(u for u in after if u.allowed)
    found = []
    for i, u in enumerate(after):
        if u.allowed:
            continue
        if old[u.key] > 0:
            old[u.key] -= 1
        else:
            found.append(i)
    return found


def added(befores: Iterable[str], after: str, lang: Lang) -> list[Unit]:
    mine = units(after, lang)
    if all(u.allowed for u in mine):
        return []
    return [mine[i] for i in surplus((counts(units(text, lang)) for text in befores), mine)]


def added_across(names: list[tuple[str, Lang]], befores: list[Tree], after: Tree) -> list[tuple[str, Unit]]:
    def judge(item):
        name, lang = item
        texts = [read(name) for read in befores]
        text = after(name)
        lexed = {t: units(t, lang) for t in {*texts, text}}
        olds = [counts(lexed[t]) for t in texts]
        own = set(surplus(olds, lexed[text]))
        return olds, [(i in own, name, u) for i, u in enumerate(lexed[text])]

    judged = parallel(judge, names)
    pools = [sum((olds[k] for olds, _ in judged), Counter()) for k in range(len(befores))]
    entries = sorted((e for _, es in judged for e in es), key=lambda e: (e[0], e[1], e[2].line))
    mine = [u for _, _, u in entries]
    found = [(entries[i][1], entries[i][2]) for i in surplus(pools, mine)]
    return sorted(found, key=lambda e: (e[0], e[1].line))


def at(root: str, rev: str) -> Tree:
    return lambda name: show(root, rev, name)


def working(root: str) -> Tree:
    def read(name: str) -> str:
        path = os.path.join(root, name)
        return read_text(path) if is_regular(path) else ""

    return read


def starts(root: str, base: Optional[str] = None) -> list[str]:
    base = merge_base(root) if base is None else base
    return [base] + merged_tips(root, base)


def change(root: str, revs: list[str], extra: Iterable[str] = ()) -> list[tuple[str, Lang]]:
    names = set(extra).union(untracked(root), *(differing(root, rev) for rev in revs))
    return covered_in(sorted(names), root, revs)


def added_since(root: str, base: Optional[str] = None) -> list[tuple[str, Unit]]:
    revs = starts(root, base)
    return added_across(change(root, revs), [at(root, rev) for rev in revs], working(root))


def added_by_edit(root: str, name: str, after: str, found: list[Unit]) -> list[Unit]:
    now = working(root)

    def edited(other: str) -> str:
        return after if other == name else now(other)

    try:
        revs = starts(root)
        trees = [at(root, rev) for rev in revs] + [now]
        budget = counts(u for _, u in added_across(change(root, revs, [name]), trees, edited))
    except Exception:
        return found
    kept = []
    for u in reversed(found):
        if budget[u.key] > 0:
            budget[u.key] -= 1
            kept.append(u)
    return kept[::-1]


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
        if found:
            found = added_by_edit(root, name, after, found)
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
