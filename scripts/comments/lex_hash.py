import re
from dataclasses import dataclass, field

from files import Lang
from lex_cxx import LineIndex, Span

_NEXT = {Lang.CCL: re.compile(r"""[#"'/]"""), Lang.PAR: re.compile(r'[#"]')}
_QUOTED = {
    '"': re.compile(r'"(?:[^"\\]|\\.)*("?)', re.S),
    "'": re.compile(r"'(?:[^'\\]|\\.)*('?)", re.S),
}


_FORTRAN_NEXT = re.compile(r"""[!"']""")
_FORTRAN_QUOTED_REST = {
    '"': re.compile(r'(?:[^"\n]|"")*("?)'),
    "'": re.compile(r"(?:[^'\n]|'')*('?)"),
}
_DIRECTIVE = re.compile(r"[ \t]*#")
_DIRECTIVE_NAME = re.compile(r"[ \t]*#[ \t]*(\w*)")
_BLANKS = re.compile(r"[ \t\r]*")
_EITHER = "either"


@dataclass
class _Conditional:
    entry: str | None
    ends: list[str | None] = field(default_factory=list)
    has_else: bool = False


def _quote_after_directive(line: str, quote: str | None, conditionals: list[_Conditional]) -> str | None:
    name = _DIRECTIVE_NAME.match(line).group(1)
    if name in ("if", "ifdef", "ifndef"):
        conditionals.append(_Conditional(quote))
        return quote
    if name not in ("elif", "elifdef", "elifndef", "else", "endif"):
        return quote
    if not conditionals:
        conditionals.append(_Conditional(_EITHER))
    c = conditionals.pop() if name == "endif" else conditionals[-1]
    c.ends.append(quote)
    c.has_else = c.has_else or name == "else"
    if name != "endif":
        return c.entry
    ends = c.ends if c.has_else else c.ends + [c.entry]
    return ends[0] if all(e == ends[0] for e in ends) else _EITHER


def _hash_span(text: str, start: int, stop: int, index: LineIndex, kind: str = "hash") -> Span:
    newline = text.find("\n", start, stop)
    end = stop if newline < 0 else newline
    if end > start and text[end - 1] == "\r":
        end -= 1
    return Span(kind, start, end, index.line_of(start), text[start:end])


def _string_continues(text: str, line_start: int, line_end: int) -> bool:
    return text[line_start:line_end].rstrip().endswith("&")


def _lex_fortran(text: str) -> list[Span]:
    index = LineIndex(text)
    spans = []
    quote = None
    conditionals: list[_Conditional] = []
    for line_start in index.starts:
        newline = text.find("\n", line_start)
        line_end = len(text) if newline < 0 else newline
        if _DIRECTIVE.match(text, line_start, line_end):
            quote = _quote_after_directive(text[line_start:line_end], quote, conditionals)
            continue
        i = line_start
        if quote is not None:
            first = _BLANKS.match(text, line_start, line_end).end()
            if first == line_end or text[first] == "!":
                if first < line_end and not text.startswith("!$", first):
                    spans.append(_hash_span(text, first, line_end, index, "line"))
                continue
            if quote == _EITHER:
                if not _string_continues(text, line_start, line_end):
                    quote = None
                continue
            if text[first] == "&":
                i = first + 1
            rest = _FORTRAN_QUOTED_REST[quote].match(text, i, line_end)
            if not rest.group(1):
                if not _string_continues(text, line_start, line_end):
                    quote = None
                continue
            quote = None
            i = rest.end()
        while (m := _FORTRAN_NEXT.search(text, i, line_end)) is not None:
            i = m.start()
            if text[i] == "!":
                if not text.startswith("!$", i):
                    spans.append(_hash_span(text, i, line_end, index, "line"))
                break
            rest = _FORTRAN_QUOTED_REST[text[i]].match(text, i + 1, line_end)
            if not rest.group(1):
                if _string_continues(text, line_start, line_end):
                    quote = text[i]
                break
            i = rest.end()
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
