from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from files import Lang
from .rules import LINE_BREAK, lex, units
from .tokens import Kind, Token

CODE_KINDS = (Kind.CODE, Kind.DIRECTIVE)
QUOTED = re.compile(r'"(?:\\.|[^"\\])*"?|\'(?:\\.|[^\'\\])*\'?')
SCOPE_OPENER = re.compile(r'(?:inline )?namespace\b[^{}]*\{|extern "C(?:\+\+)?" \{')


@dataclass(frozen=True)
class Comment:
    first: int
    last: int
    end: int
    inline: bool
    units: tuple


@dataclass
class Layout:
    rows: tuple = ()
    units: tuple = ()
    comments: tuple = ()
    at: dict = field(default_factory=dict)


@dataclass
class Group:
    first: int
    last: int
    column: int
    inline: bool


def last_lines(tokens: list[Token]) -> list[int]:
    found = []
    for i, token in enumerate(tokens):
        if token.kind is Kind.NEWLINE:
            found.append(token.line)
        elif i + 1 < len(tokens):
            found.append(tokens[i + 1].line)
        else:
            found.append(token.line + len(LINE_BREAK.findall(token.text)))
    return found


def brace_steps(token: Token, lang: Lang) -> list[int]:
    if lang is Lang.CXX:
        return [1] if token.text == "{" else [-1] if token.text == "}" else []
    return [1 if c == "{" else -1 for c in QUOTED.sub("", token.text) if c in "{}"]


def group(marks: list[tuple[int, int, int]], has_code: list[bool]) -> list[Group]:
    groups: list[Group] = []
    for first, last, column in marks:
        inline = any(has_code[line] for line in range(first, last + 1))
        prev = groups[-1] if groups else None
        if prev is not None and first <= prev.last:
            prev.last, prev.column, prev.inline = max(prev.last, last), column, prev.inline or inline
        elif prev is not None and first == prev.last + 1 and not inline and (
            not prev.inline or column == prev.column
        ):
            prev.last = last
        else:
            groups.append(Group(first, last, column, inline))
    return groups


def span_end(g: Group, n: int, busy: list[bool], starts: set, steps: list[list[int]]) -> int:
    if g.inline:
        return g.last
    depth, end, line = 0, g.last, g.last + 1
    while line <= n:
        if depth == 0 and (not busy[line] or line in starts):
            break
        running = low = depth
        for step in steps[line]:
            running += step
            low = min(low, running)
        if low < 0:
            break
        depth, end, line = running, line, line + 1
    return end


def layout(text: str, lang: Lang) -> Layout:
    tokens = lex(text, lang) if text else []
    if not tokens:
        return Layout()
    found = tuple(units(text, lang, tokens))
    ends = last_lines(tokens)
    n = max(ends)
    pieces: list[list[str]] = [[] for _ in range(n + 2)]
    steps: list[list[int]] = [[] for _ in range(n + 2)]
    has_code = [False] * (n + 2)
    busy = [False] * (n + 2)
    marks: list[tuple[int, int, int]] = []
    logical: list[tuple[int, int, Optional[int]]] = []
    column, first_line, code_line = 0, None, None
    for token, last in zip(tokens, ends):
        if token.kind is Kind.NEWLINE:
            if first_line is not None:
                logical.append((first_line, token.line, code_line))
            column, first_line, code_line = 0, None, None
            continue
        first_line = token.line if first_line is None else first_line
        if token.kind is Kind.COMMENT:
            marks.append((token.line, last, column))
            for line in range(token.line, last + 1):
                busy[line] = True
        elif token.kind in CODE_KINDS:
            code_line = token.line if code_line is None else code_line
            for offset, piece in enumerate(LINE_BREAK.split(token.text)):
                if piece.strip():
                    pieces[token.line + offset].append(piece.strip())
                    has_code[token.line + offset] = busy[token.line + offset] = True
            if token.kind is Kind.CODE:
                steps[token.line].extend(brace_steps(token, lang))
        broken = LINE_BREAK.split(token.text)
        column = len(broken[-1]) if len(broken) > 1 else column + len(token.text)
    if first_line is not None:
        logical.append((first_line, ends[-1], code_line))

    directives = {u.line for u in found if u.key[0] == "directive"}
    for first, last, code in logical:
        if code in directives:
            for line in range(first, last + 1):
                pieces[line], steps[line] = [], []
    if lang is Lang.CXX:
        for line in range(1, n + 1):
            if SCOPE_OPENER.fullmatch(" ".join(pieces[line])):
                steps[line] = []

    groups = group(marks, has_code)
    starts = {g.first for g in groups if not g.inline}
    owner = {line: k for k, g in enumerate(groups) for line in range(g.first, g.last + 1)}
    members: list[list[int]] = [[] for _ in groups]
    for i, u in enumerate(found):
        if u.key[0] == "comment" and u.line in owner:
            members[owner[u.line]].append(i)
    comments = tuple(
        Comment(g.first, g.last, span_end(g, n, busy, starts, steps), g.inline, tuple(members[k]))
        for k, g in enumerate(groups)
    )
    keys: list[list[tuple]] = [[] for _ in range(n + 2)]
    for u in found:
        keys[u.line].append(u.key)
    rows = tuple((" ".join(pieces[line]), tuple(keys[line])) for line in range(1, n + 1))
    at = {line: c for c in comments for line in range(c.first, c.last + 1)}
    return Layout(rows, found, comments, at)
