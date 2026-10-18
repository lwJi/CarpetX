import re

from files import Lang
from lex_cxx import LineIndex, Span

_NEXT = {Lang.CCL: re.compile(r"""[#"'/]"""), Lang.PAR: re.compile(r'[#"]')}
_QUOTED = {
    '"': re.compile(r'"(?:[^"\\]|\\.)*("?)', re.S),
    "'": re.compile(r"'(?:[^'\\]|\\.)*('?)", re.S),
}


_FORTRAN_NEXT = re.compile(r"""[!"']""")
_FORTRAN_QUOTED = {
    '"': re.compile(r'"(?:[^"\n]|"")*"?'),
    "'": re.compile(r"'(?:[^'\n]|'')*'?"),
}
_DIRECTIVE = re.compile(r"[ \t]*#")


def _hash_span(text: str, start: int, stop: int, index: LineIndex, kind: str = "hash") -> Span:
    newline = text.find("\n", start, stop)
    end = stop if newline < 0 else newline
    if end > start and text[end - 1] == "\r":
        end -= 1
    return Span(kind, start, end, index.line_of(start), text[start:end])


def _lex_fortran(text: str) -> list[Span]:
    index = LineIndex(text)
    spans = []
    for line_start in index.starts:
        newline = text.find("\n", line_start)
        line_end = len(text) if newline < 0 else newline
        if _DIRECTIVE.match(text, line_start, line_end):
            continue
        i = line_start
        while (m := _FORTRAN_NEXT.search(text, i, line_end)) is not None:
            i = m.start()
            if text[i] == "!":
                if not text.startswith("!$", i):
                    spans.append(_hash_span(text, i, line_end, index, "line"))
                break
            i = _FORTRAN_QUOTED[text[i]].match(text, i, line_end).end()
    return spans


def _par_string_comments(text: str, start: int, end: int, index: LineIndex) -> list[Span]:
    spans = []
    newline = text.find("\n", start, end)
    while newline >= 0:
        hash_ = text.find("#", newline + 1, end)
        if hash_ < 0:
            break
        spans.append(_hash_span(text, hash_, end, index))
        newline = text.find("\n", hash_, end)
    return spans


def scan(text: str, lang: Lang) -> tuple[list[Span], list[tuple[int, int]]]:
    pattern = _NEXT[lang]
    index = LineIndex(text)
    spans = []
    strings = []
    i = 0
    while (m := pattern.search(text, i)) is not None:
        i = m.start()
        c = text[i]
        if c == "#":
            span = _hash_span(text, i, len(text), index)
            spans.append(span)
            i = span.end
        elif c == "/":
            if text.startswith("/*", i):
                close = text.find("*/", i + 2)
                end = len(text) if close < 0 else close + 2
                spans.append(Span("block", i, end, index.line_of(i), text[i:end]))
                i = end
            else:
                i += 1
        else:
            quoted = _QUOTED[c].match(text, i)
            end = quoted.end()
            content_end = quoted.start(1)
            strings.append((i + 1, content_end))
            if lang is Lang.PAR:
                spans.extend(_par_string_comments(text, i + 1, content_end, index))
            i = end
    return spans, strings


def lex(text: str, lang: Lang) -> list[Span]:
    if lang is Lang.FORTRAN:
        return _lex_fortran(text)
    return scan(text, lang)[0]


def strings(text: str, lang: Lang) -> list[tuple[int, int]]:
    return scan(text, lang)[1]
