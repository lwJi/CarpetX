from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Iterable

from files import Lang
from .rules import Unit, may_hold_comments
from .spans import Comment, Layout, layout


@dataclass(frozen=True)
class Touched:
    first: int
    last: int
    unit: Unit


@dataclass(frozen=True)
class Verdict:
    added: tuple = ()
    touched: tuple = ()

    def __bool__(self) -> bool:
        return bool(self.added or self.touched)

    def lines(self) -> set:
        return {u.line for u in self.added} | {line for t in self.touched for line in range(t.first, t.last + 1)}

    def within(self, other: Verdict) -> Verdict:
        flagged = other.lines()
        return Verdict(
            tuple(u for u in self.added if u.line in flagged),
            tuple(t for t in self.touched if any(line in flagged for line in range(t.first, t.last + 1))),
        )


def opcodes(a, b) -> list:
    return SequenceMatcher(None, a, b, autojunk=False).get_opcodes()


def judge(before: str, after: str, lang: Lang) -> Verdict:
    if not may_hold_comments(after, lang):
        return Verdict()
    new = layout(after, lang)
    if all(u.allowed for u in new.units):
        return Verdict()
    old = layout(before, lang)

    on_line: dict = {}
    for i, u in enumerate(new.units):
        on_line.setdefault(u.line, []).append(i)
    same: dict = {}
    fresh, moved = set(), set()
    for tag, i1, i2, j1, j2 in opcodes(old.rows, new.rows):
        if tag == "equal":
            same.update((j1 + k + 1, i1 + k + 1) for k in range(j2 - j1))
            continue
        pool = Counter(key for _, keys in old.rows[i1:i2] for key in keys)
        for line in range(j1 + 1, j2 + 1):
            for i in on_line.get(line, ()):
                if pool[new.units[i].key] > 0:
                    pool[new.units[i].key] -= 1
                    moved.add(i)
                else:
                    fresh.add(i)

    old_code = [code for code, _ in old.rows]
    new_code = [code for code, _ in new.rows]
    edits = [op for op in opcodes(old_code, new_code) if op[0] != "equal"]
    kept = set(same.values())

    def code_edited_under(b: Comment, c: Comment) -> bool:
        for _, i1, i2, j1, j2 in edits:
            removed = [line for line in range(i1 + 1, i2 + 1) if old_code[line - 1]]
            if any(b.first <= line <= b.end for line in removed):
                return True
            inserted = [line for line in range(j1 + 1, j2 + 1) if new_code[line - 1]]
            if removed or not inserted:
                continue
            if b.first <= i1 < b.end:
                return True
            if i1 == b.end and not b.inline and any(c.first <= line <= c.end for line in inserted):
                return True
        return False

    def fingerprint(lay: Layout, c: Comment) -> tuple:
        words = tuple(lay.units[i].key for i in c.units)
        return words, tuple(code for code, _ in lay.rows[c.first - 1 : c.end] if code)

    touched = []
    for c in new.comments:
        known = [new.units[i] for i in c.units if i not in fresh]
        if all(u.allowed for u in known):
            continue
        sources = {old.at[same[u.line]] for u in known if same.get(u.line) in old.at}
        if len(sources) == 1 and fingerprint(new, c) == fingerprint(old, next(iter(sources))):
            continue
        hit = any(i in moved for i in c.units) or any(
            code_edited_under(b, c) or any(old.units[i].line not in kept for i in b.units) for b in sources
        )
        if hit:
            touched.append(Touched(c.first, c.last, next(u for u in known if not u.allowed)))
    inside = {line for t in touched for line in range(t.first, t.last + 1)}
    added = tuple(new.units[i] for i in sorted(fresh) if not new.units[i].allowed and new.units[i].line not in inside)
    return Verdict(added, tuple(touched))


def judge_all(befores: Iterable[str], after: str, lang: Lang) -> Verdict:
    verdict = None
    for before in befores:
        other = judge(before, after, lang)
        verdict = other if verdict is None else verdict.within(other)
        if not verdict:
            break
    return Verdict() if verdict is None else verdict
