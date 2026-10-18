import lex_cxx
from files import Lang
from lex_cxx import Span
from rules import Violation, classify, group, lex

_HORIZONTAL = " \t"
_KEEP_SPACE_AFTER = "([{,"
_NO_SPACE_BEFORE = ")]},;"
_WORD = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_")
_OPERATOR = set("+-*/%<>=&|^!.:#")


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


def _tidy(
    text: str,
    junction: int,
    keep_prev: bool = False,
    keep_next: bool = False,
    pending: list[Span] = (),
) -> str:
    prev, nxt = _line_before(text, junction), _line_after(text, junction)
    if _blank(text, prev) and _blank(text, nxt):
        if not keep_next:
            text = _cut(text, nxt)
        elif not keep_prev:
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


def _would_merge(left: str, right: str) -> bool:
    return (left in _WORD and right in _WORD) or (left in _OPERATOR and right in _OPERATOR)


def _delete_disabled(text: str, span: Span, pending: list[Span]) -> str:
    closes, opens = lex_cxx.guard_shape(span)
    text = _cut(text, (span.start, span.end))
    return _tidy(text, span.start, keep_prev=closes, keep_next=opens, pending=pending)


def _delete_comment(text: str, span: Span, pending: list[Span]) -> str:
    line_start = text.rfind("\n", 0, span.start) + 1
    newline = text.find("\n", span.end)
    line_end = len(text) if newline < 0 else newline
    before = text[line_start : span.start]
    after = text[span.end : line_end]
    if not before.strip() and not after.strip():
        cut_end = line_end + 1 if newline >= 0 else line_end
        return _tidy(_cut(text, (line_start, cut_end)), line_start, pending=pending)
    if not after.strip():
        start = line_start + len(before.rstrip(_HORIZONTAL))
        end = line_end - 1 if after.endswith("\r") else line_end
        return _cut(text, (start, end))
    start, end = span.start, span.end
    previous = text[start - 1] if start > line_start else ""
    following = after.lstrip(_HORIZONTAL)[:1]
    closes_up = following in _NO_SPACE_BEFORE or (following == "[" and before.rstrip()[-1:] in _WORD)
    if following and closes_up and before.strip():
        while start > line_start and text[start - 1] in _HORIZONTAL:
            start -= 1
        while end < line_end and text[end] in _HORIZONTAL:
            end += 1
    elif previous == "" or previous in _HORIZONTAL or previous in _KEEP_SPACE_AFTER:
        while end < line_end and text[end] in _HORIZONTAL:
            end += 1
    else:
        while start > line_start and text[start - 1] in _HORIZONTAL:
            start -= 1
    left = text[start - 1] if start > line_start else ""
    right = text[end] if end < line_end else ""
    replacement = " " if _would_merge(left, right) else ""
    return text[:start] + replacement + text[end:]


def violating_spans(text: str, lang: Lang) -> list[Span]:
    spans = []
    for g in group(lex(text, lang), text):
        if isinstance(classify(g, text), Violation):
            spans.extend(g.spans)
    return spans


def strip(text: str, lang: Lang) -> str:
    if lang is Lang.CXX:
        refusals = lex_cxx.scan(text).refusals
        if refusals:
            raise Refused(*refusals[0])
    spans = sorted(violating_spans(text, lang), key=lambda s: s.start)
    for i in reversed(range(len(spans))):
        pending = spans[:i]
        if spans[i].kind == "disabled":
            text = _delete_disabled(text, spans[i], pending)
        else:
            text = _delete_comment(text, spans[i], pending)
    return text
