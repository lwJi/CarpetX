from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from files import Lang
from tokens import Kind, LexError, Token

PLAIN = re.compile(r"(?P<newline>\r?\n)|(?P<space>(?:[ \t\f\v]|\r(?!\n))+)|(?P<code>[^ \t\f\v\r\n]+)")
NEXT_LINE = re.compile(r"\r?\n|\Z")
BLANKS = re.compile(r"[ \t\f\v\r]*")
FORTRAN_SPECIAL = re.compile(r"""[!"']""")
FORTRAN_DIRECTIVE = re.compile(r"[ \t\f\v]*#")
FORTRAN_SENTINEL = re.compile(r"!\$[A-Za-z]*")


@dataclass(frozen=True)
class Dialect:
    special: re.Pattern
    quotes: str = ""
    multiline: bool = True
    escapes: bool = False


DIALECTS = {
    Lang.INTERFACE: Dialect(re.compile(r"""[#"']"""), quotes="\"'"),
    Lang.PARAM: Dialect(re.compile(r'[#"]'), quotes='"'),
    Lang.SCHEDULE: Dialect(re.compile(r'[#"]|/\*'), quotes='"'),
    Lang.CONFIGURATION: Dialect(re.compile(r"#")),
    Lang.TEST: Dialect(re.compile(r"""[#"'\\]"""), quotes="\"'", multiline=False, escapes=True),
    Lang.PAR: Dialect(re.compile(r'[#"]'), quotes='"'),
    Lang.MAKE: Dialect(re.compile(r"[#$]")),
}
HINTS = {lang: ("#", "/*") if lang is Lang.SCHEDULE else ("#",) for lang in DIALECTS}
HINTS[Lang.FORTRAN] = ("!",)


class Out:
    def __init__(self, text: str):
        self.text = text
        self.items: list[list] = []
        self.strings: list[tuple[int, int]] = []

    def add(self, kind: Kind, start: int, end: int) -> None:
        if start >= end:
            return
        last = self.items[-1] if self.items else None
        if last is not None and last[0] is kind and kind in (Kind.CODE, Kind.SPACE) and last[2] == start:
            last[2] = end
        else:
            self.items.append([kind, start, end])

    def plain(self, start: int, end: int) -> None:
        for m in PLAIN.finditer(self.text, start, end):
            kind = Kind.NEWLINE if m.group("newline") else Kind.SPACE if m.group("space") else Kind.CODE
            self.add(kind, m.start(), m.end())

    def last_code(self) -> Optional[str]:
        for kind, start, end in reversed(self.items):
            if kind is Kind.CODE:
                return self.text[start:end]
            if kind not in (Kind.SPACE, Kind.NEWLINE, Kind.COMMENT):
                return None
        return None

    def tokens(self) -> list[Token]:
        found = []
        offset = 0
        line = 1
        for kind, start, end in self.items:
            piece = self.text[start:end]
            found.append(Token(kind, piece, offset, line))
            offset += len(piece.encode("utf-8", "surrogateescape"))
            line += piece.count("\n")
        return found


def line_end(text: str, start: int, stop: Optional[int] = None) -> int:
    stop = len(text) if stop is None else stop
    newline = text.find("\n", start, stop)
    if newline < 0:
        return stop
    return newline - 1 if newline > start and text[newline - 1] == "\r" else newline


def next_line(text: str, end: int) -> int:
    return NEXT_LINE.search(text, end).end()


def odd_backslashes_before(text: str, end: int, start: int = 0) -> bool:
    k = end
    while k > start and text[k - 1] == "\\":
        k -= 1
    return (end - k) % 2 == 1


def logical_line_end(text: str, start: int) -> int:
    end = line_end(text, start)
    while end < len(text) and odd_backslashes_before(text, end, start):
        end = line_end(text, next_line(text, end))
    return end


def string_end(text: str, start: int, dialect: Dialect) -> tuple[int, bool]:
    quote = text[start]
    i = start + 1
    n = len(text)
    while i < n:
        c = text[i]
        if c in "\r\n" and not dialect.multiline:
            return line_end(text, start), False
        if c == "\\":
            if not dialect.multiline and text[i + 1 : i + 2] in ("\r", "\n"):
                return line_end(text, start), False
            i += 2
            continue
        if c == quote:
            return i + 1, True
        i += 1
    return n, False


def par_string(out: Out, start: int, end: int, closed: bool) -> None:
    text = out.text
    content_end = end - 1 if closed else end
    first_line_end = line_end(text, start, content_end)
    if first_line_end == content_end or out.last_code() == "!DESC":
        out.add(Kind.CODE, start, end)
        return
    out.strings.append((start, end))
    out.add(Kind.CODE, start, first_line_end)
    i = first_line_end
    while i < content_end:
        hash_ = text.find("#", i, content_end)
        out.plain(i, content_end if hash_ < 0 else hash_)
        if hash_ < 0:
            break
        i = line_end(text, hash_, content_end)
        out.add(Kind.COMMENT, hash_, i)
    out.add(Kind.CODE, content_end, end)


def make_reference(out: Out, start: int) -> int:
    text = out.text
    opener = text[start + 1 : start + 2]
    if opener in ("", "\n", "\r"):
        out.add(Kind.CODE, start, start + 1)
        return start + 1
    if opener not in ("(", "{"):
        out.add(Kind.CODE, start, start + 2)
        return start + 2
    closer = ")" if opener == "(" else "}"
    stop = logical_line_end(text, start)
    depth = 0
    i = start + 1
    while i < stop:
        if text[i] == opener:
            depth += 1
        elif text[i] == closer:
            depth -= 1
            if depth == 0:
                i += 1
                break
        i += 1
    out.plain(start, i)
    return i


def lex_hashed(text: str, lang: Lang) -> Out:
    dialect = DIALECTS[lang]
    out = Out(text)
    i = 0
    n = len(text)
    while i < n:
        m = dialect.special.search(text, i)
        if m is None:
            out.plain(i, n)
            break
        j = m.start()
        out.plain(i, j)
        c = text[j]
        if c == "\\":
            end = j + 1 if text[j + 1 : j + 2] in ("", "\n", "\r") else j + 2
            out.add(Kind.CODE, j, end)
            i = end
        elif c == "$":
            i = make_reference(out, j)
        elif c == "#" and lang is Lang.MAKE and odd_backslashes_before(text, j):
            out.add(Kind.CODE, j, j + 1)
            i = j + 1
        elif c == "#":
            end = logical_line_end(text, j) if lang is Lang.MAKE else line_end(text, j)
            out.add(Kind.COMMENT, j, end)
            i = end
        elif c == "/":
            close = text.find("*/", j + 2)
            end = n if close < 0 else close + 2
            out.add(Kind.COMMENT, j, end)
            i = end
        else:
            end, closed = string_end(text, j, dialect)
            if lang is Lang.PAR:
                par_string(out, j, end, closed)
            else:
                out.add(Kind.CODE, j, end)
            i = end
    return out


def fortran_quote_end(text: str, start: int, stop: int, quote: str) -> Optional[int]:
    i = start
    while i < stop:
        if text[i] == quote:
            if i + 1 < stop and text[i + 1] == quote:
                i += 2
                continue
            return i + 1
        i += 1
    return None


def continues(text: str, start: int, stop: int) -> bool:
    return text[start:stop].rstrip().endswith("&")


def fortran_rest(out: Out, i: int, line_start: int, stop: int) -> Optional[str]:
    text = out.text
    while i < stop:
        m = FORTRAN_SPECIAL.search(text, i, stop)
        if m is None:
            out.plain(i, stop)
            return None
        j = m.start()
        out.plain(i, j)
        if text[j] == "!":
            out.add(Kind.COMMENT, j, stop)
            return None
        close = fortran_quote_end(text, j + 1, stop, text[j])
        if close is None:
            out.add(Kind.CODE, j, stop)
            return text[j] if continues(text, line_start, stop) else None
        out.add(Kind.CODE, j, close)
        i = close
    return None


def lex_fortran(text: str) -> Out:
    out = Out(text)
    n = len(text)
    quote: Optional[str] = None
    line_start = 0
    while line_start < n:
        stop = line_end(text, line_start)
        first = BLANKS.match(text, line_start, stop).end()
        if FORTRAN_DIRECTIVE.match(text, line_start, stop):
            out.plain(line_start, first)
            while stop < n and odd_backslashes_before(text, stop, first):
                stop = line_end(text, next_line(text, stop))
            out.add(Kind.DIRECTIVE, first, stop)
        elif first < stop and text.startswith("!$", first):
            out.plain(line_start, first)
            sentinel = FORTRAN_SENTINEL.match(text, first, stop).end()
            out.add(Kind.DIRECTIVE, first, sentinel)
            fortran_rest(out, sentinel, line_start, stop)
        elif quote is not None and (first == stop or text[first] == "!"):
            out.plain(line_start, first)
            out.add(Kind.COMMENT, first, stop)
        elif quote is not None:
            ampersand = text[first] == "&"
            begin = first if ampersand else line_start
            out.plain(line_start, begin)
            close = fortran_quote_end(text, first + 1 if ampersand else line_start, stop, quote)
            if close is None:
                out.add(Kind.CODE, begin, stop)
                if not continues(text, line_start, stop):
                    quote = None
            else:
                out.add(Kind.CODE, begin, close)
                quote = fortran_rest(out, close, line_start, stop)
        else:
            quote = fortran_rest(out, line_start, line_start, stop)
        end = next_line(text, stop)
        out.plain(stop, end)
        line_start = end
    return out


def par_strings(text: str) -> list[tuple[int, int]]:
    return lex_hashed(text, Lang.PAR).strings


def may_hold_comments(text: str, lang: Lang) -> bool:
    return any(hint in text for hint in HINTS[lang])


def lex(text: str, lang: Lang) -> list[Token]:
    out = lex_fortran(text) if lang is Lang.FORTRAN else lex_hashed(text, lang)
    tokens = out.tokens()
    if "".join(t.text for t in tokens) != text:
        raise LexError("token texts do not tile the file")
    return tokens
