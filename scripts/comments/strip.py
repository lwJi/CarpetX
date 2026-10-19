import bisect
import re

import lex_cxx
import lex_hash
from files import Lang
from lex_cxx import Span
from rules import Violation, classify, group, lex

_HORIZONTAL = " \t"
_KEEP_SPACE_AFTER = "([{,"
_NO_SPACE_BEFORE = ")]},;"
_WORD = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_")
_SPACED_BEFORE_PAREN = frozenset(
    {"if", "for", "while", "switch", "catch", "return", "case", "throw", "constexpr", "and", "or", "not"}
)


class Refused(Exception):
    def __init__(self, line: int, reason: str):
        super().__init__(f"{line}: {reason}")
        self.line = line
        self.reason = reason


def _line_before(text: str, junction: int) -> tuple[int, int] | None:
    if junction == 0:
        return None
    return text.rfind("\n", 0, junction - 1) + 1, junction


def _line_after(text: str, junction: int) -> tuple[int, int] | None:
    if junction >= len(text):
        return None
    newline = text.find("\n", junction)
    return junction, len(text) if newline < 0 else newline + 1


def _blank(text: str, bounds: tuple[int, int] | None) -> bool:
    return bounds is not None and not text[bounds[0] : bounds[1]].strip()


def _cut(text: str, bounds: tuple[int, int]) -> str:
    return text[: bounds[0]] + text[bounds[1] :]


def _final_line(text: str, bounds: tuple[int, int], pending: list[Span]) -> str:
    start, end = bounds
    parts, last = [], start
    for s in pending:
        if s.start < end and start < s.end:
            parts.append(text[last : max(s.start, last)])
            last = max(last, min(s.end, end))
    parts.append(text[last:end])
    return "".join(parts)


def _blank_run(text: str, junction: int, after: bool) -> int:
    count = 0
    bounds = _line_after(text, junction) if after else _line_before(text, junction)
    while _blank(text, bounds):
        count += 1
        bounds = _line_after(text, bounds[1]) if after else _line_before(text, bounds[0])
    return count


def _tidy(
    text: str,
    junction: int,
    keep_prev: bool = False,
    keep_next: bool = False,
    pending: list[Span] = (),
) -> str:
    prev, nxt = _line_before(text, junction), _line_after(text, junction)
    if _blank(text, prev) and _blank(text, nxt):
        for _ in range(min(_blank_run(text, junction, False), _blank_run(text, junction, True))):
            if not keep_next:
                text = _cut(text, _line_after(text, junction))
            elif not keep_prev:
                prev = _line_before(text, junction)
                text = _cut(text, prev)
                junction = prev[0]
        prev, nxt = _line_before(text, junction), _line_after(text, junction)
    while (
        _blank(text, nxt)
        and not keep_next
        and (prev is None or _final_line(text, prev, pending).rstrip().endswith("{"))
    ):
        text = _cut(text, nxt)
        nxt = _line_after(text, junction)
    while (
        _blank(text, prev)
        and not keep_prev
        and (nxt is None or text[nxt[0] : nxt[1]].lstrip().startswith("}"))
    ):
        text = _cut(text, prev)
        junction = prev[0]
        prev = _line_before(text, junction)
    return text


def _name_before_paren(before: str) -> bool:
    name = re.search(r"\w+$", before.rstrip())
    return name is not None and name.group() not in _SPACED_BEFORE_PAREN


def _inside(ranges: list[tuple[int, int]], offset: int) -> bool:
    k = bisect.bisect_right(ranges, (offset, float("inf"))) - 1
    return k >= 0 and offset < ranges[k][1]


def _delete_disabled(text: str, span: Span, pending: list[Span]) -> str:
    closes, opens = lex_cxx.guard_shape(span)
    text = _cut(text, (span.start, span.end))
    return _tidy(text, span.start, keep_prev=closes, keep_next=opens, pending=pending)


def _delete_comment(
    text: str, span: Span, pending: list[Span], tidy: bool = True, in_directive: bool = False
) -> str:
    line_start = text.rfind("\n", 0, span.start) + 1
    newline = text.find("\n", span.end)
    line_end = len(text) if newline < 0 else newline
    before = text[line_start : span.start]
    after = text[span.end : line_end]
    if not before.strip() and not after.strip():
        cut_end = line_end + 1 if newline >= 0 else line_end
        text = _cut(text, (line_start, cut_end))
        return _tidy(text, line_start, pending=pending) if tidy else text
    if not after.strip():
        start = line_start + len(before.rstrip(_HORIZONTAL))
        end = line_end - 1 if after.endswith("\r") else line_end
        return _cut(text, (start, end))
    start, end = span.start, span.end
    previous = text[start - 1] if start > line_start else ""
    following = after.lstrip(_HORIZONTAL)[:1]
    closes_up = (
        following in _NO_SPACE_BEFORE
        or (following == "[" and before.rstrip()[-1:] in _WORD)
        or (following == "(" and _name_before_paren(before))
    )
    if not in_directive and following and closes_up and before.strip():
        while start > line_start and text[start - 1] in _HORIZONTAL:
            start -= 1
        while end < line_end and text[end] in _HORIZONTAL:
            end += 1
        return text[:start] + text[end:]
    if (previous != "" and previous in _HORIZONTAL) or (
        not in_directive and (previous == "" or previous in _KEEP_SPACE_AFTER)
    ):
        while end < line_end and text[end] in _HORIZONTAL:
            end += 1
        return text[:start] + text[end:]
    return text[:start] + ("" if text[end] in _HORIZONTAL else " ") + text[end:]


def violating_spans(text: str, lang: Lang) -> list[Span]:
    spans = []
    for g in group(lex(text, lang), text):
        if isinstance(classify(g, text), Violation):
            spans.extend(g.spans)
    return spans


def strip(text: str, lang: Lang) -> str:
    directives: list[tuple[int, int]] = []
    if lang is Lang.CXX:
        scanned = lex_cxx.scan(text)
        if scanned.refusals:
            raise Refused(*scanned.refusals[0])
        directives = scanned.directives
    spans = sorted(violating_spans(text, lang), key=lambda s: s.start)
    strings = lex_hash.strings(text, lang) if lang is Lang.PAR else []
    for i in reversed(range(len(spans))):
        pending = spans[:i]
        if spans[i].kind == "disabled":
            text = _delete_disabled(text, spans[i], pending)
        else:
            in_string = any(start <= spans[i].start < end for start, end in strings)
            in_directive = _inside(directives, spans[i].start)
            text = _delete_comment(text, spans[i], pending, tidy=not in_string, in_directive=in_directive)
    return text
