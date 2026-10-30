from __future__ import annotations

from collections import Counter
from typing import Iterable

from checks import Check
from files import Lang

from .rules import CHECKED, RULE, Finding, findings, new_findings, render


def covers(name: str, lang: Lang) -> bool:
    return lang in CHECKED


def common(a: list[Finding], b: list[Finding]) -> list[Finding]:
    keep = Counter(a) & Counter(b)
    found = []
    for f in a:
        if keep[f] > 0:
            keep[f] -= 1
            found.append(f)
    return found


def judge(name: str, befores: Iterable[tuple[str, str]], after: str, lang: Lang) -> list[Finding]:
    now = findings(name, after, lang)
    verdict = None
    for old_name, text in befores:
        new = new_findings(findings(old_name, text, lang), now)
        verdict = new if verdict is None else common(verdict, new)
        if not verdict:
            break
    return [] if verdict is None else verdict


def render_change(found: list[tuple[str, list[Finding]]], edit: bool = False) -> str:
    return render([(name, f) for name, verdict in found for f in verdict])


CHECK = Check("boundary", RULE, covers, findings, judge, render, render_change)
