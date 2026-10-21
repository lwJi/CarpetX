from __future__ import annotations

import bisect
import re
from typing import Optional

from tokens import Kind, LexError, Token

SPLICE = r"\\[ \t\f\v]*(?:\r\n|\r|\n)"
TOKEN = re.compile(
    rf"""
    (?P<newline>\r\n|\r|\n)
  | (?P<space>(?:\A﻿|[ \t\f\v]|{SPLICE})+)
  | (?P<comment>//(?:{SPLICE}|[^\r\n])*|/\*.*?(?:\*/|\Z))
  | (?P<raw>(?:u8|[uUL])?R"(?P<delimiter>[^ ()\\\t\v\f\r\n]{{0,16}})\(.*?\)(?P=delimiter)")
  | (?P<quoted>(?:u8|[uUL])?(?P<quote>["'])(?:{SPLICE}|\\.|(?!(?P=quote))[^\\\r\n])*(?P=quote)?)
  | (?P<number>\.?\d(?:[eEpP][+-]|'\w|[\w.])*)
  | (?P<name>[A-Za-z_$\x80-\U0010ffff][\w$\x80-\U0010ffff]*)
  | (?P<hash>\#|%:)
  | .
    """,
    re.S | re.X,
)
HEADER_NAME = re.compile(r"<[^>\r\n]*>")
INCLUDES = {"include", "include_next", "import"}
CONDITIONALS = {"if", "elif"}
HAS_INCLUDE = {"__has_include", "__has_include_next"}
LINE_END = re.compile(r"\r\n|\r|\n")


def expects_header_name(directive: list[str]) -> bool:
    if len(directive) == 2:
        return directive[1] in INCLUDES
    return (
        len(directive) >= 4
        and directive[1] in CONDITIONALS
        and directive[-2] in HAS_INCLUDE
        and directive[-1] == "("
    )


def lex(text: str) -> list[Token]:
    items: list[tuple[Kind, int, int]] = []
    at_line_start = True
    directive: Optional[list[str]] = None
    i = 0
    while i < len(text):
        m = TOKEN.match(text, i)
        kind, end = m.lastgroup, m.end()
        if kind == "newline":
            items.append((Kind.NEWLINE, i, end))
            at_line_start, directive = True, None
        elif kind == "space":
            items.append((Kind.SPACE, i, end))
        elif kind == "comment":
            items.append((Kind.COMMENT, i, end))
        else:
            header = HEADER_NAME.match(text, i) if directive and expects_header_name(directive) else None
            end = header.end() if header else end
            if at_line_start and kind == "hash":
                directive = []
            at_line_start = False
            if directive is not None:
                directive.append(text[i:end])
            items.append((Kind.CODE if directive is None else Kind.DIRECTIVE, i, end))
        i = end
    starts = [0] + [m.end() for m in LINE_END.finditer(text)]
    tokens = []
    offset = 0
    for kind, start, end in items:
        piece = text[start:end]
        tokens.append(Token(kind, piece, offset, bisect.bisect_right(starts, start)))
        offset += len(piece.encode("utf-8", "surrogateescape"))
    if "".join(t.text for t in tokens) != text:
        raise LexError("token texts do not tile the file")
    return tokens
