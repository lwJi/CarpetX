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


def join(tokens: Iterable[Token]) -> str:
    return "".join(t.text for t in tokens)
