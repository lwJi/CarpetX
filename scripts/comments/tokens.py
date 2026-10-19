from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable


class LexError(RuntimeError):
    pass


class Kind(Enum):
    CODE = "code"
    SPACE = "space"
    NEWLINE = "newline"
    COMMENT = "comment"
    DIRECTIVE = "directive"


@dataclass(frozen=True)
class Token:
    kind: Kind
    text: str
    start: int
    line: int


class Placement(Enum):
    WHOLE_LINE = "whole-line"
    TRAILING = "trailing"
    INLINE = "inline"
    GUARD = "constant guard"
    MESSAGE = "compile-time message"


DIRECTIVE_PLACEMENTS = (Placement.GUARD, Placement.MESSAGE)


@dataclass(frozen=True)
class Region:
    placement: Placement
    first: int
    last: int
    tokens: tuple

    @property
    def line(self) -> int:
        return self.tokens[0].line

    @property
    def comments(self) -> tuple:
        return tuple(t for t in self.tokens if t.kind is Kind.COMMENT)

    @property
    def text(self) -> str:
        return join(self.tokens)

    @property
    def code(self) -> str:
        pieces: list[str] = []
        for t in self.tokens:
            if t.kind in (Kind.CODE, Kind.DIRECTIVE):
                pieces.append(t.text)
            elif pieces and pieces[-1] != " ":
                pieces.append(" ")
        return "".join(pieces).strip()


def join(tokens: Iterable[Token]) -> str:
    return "".join(t.text for t in tokens)
