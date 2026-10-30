from __future__ import annotations

from checks import Check
from files import Lang

from .judge import Verdict, judge as judge_pair
from .report import RULE, render, render_change
from .rules import Unit, units


def covers(name: str, lang: Lang) -> bool:
    return True


def found(name: str, text: str, lang: Lang) -> list[Unit]:
    return [u for u in units(text, lang) if not u.allowed]


def judge(name: str, old_name: str, before: str, after: str, lang: Lang) -> Verdict:
    return judge_pair(before, after, lang)


CHECK = Check("no-comments", RULE, covers, found, judge, Verdict.within, render, render_change)
