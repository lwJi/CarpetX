from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from files import Lang

EXCERPT_WIDTH = 72

Covers = Callable[[str, Lang], bool]
Found = Callable[[str, str, Lang], list]
Judge = Callable[[str, str, str, str, Lang], object]
Meet = Callable[[object, object], object]
Render = Callable[[list[tuple[str, object]]], str]
RenderChange = Callable[[list[tuple[str, object]], bool], str]


@dataclass(frozen=True)
class Check:
    name: str
    rule: str
    covers: Covers
    found: Found
    judge: Judge
    meet: Meet
    render: Render
    render_change: RenderChange


def CHECKS() -> tuple[Check, ...]:
    from boundary.check import CHECK as boundary
    from comments.check import CHECK as comments

    return (comments, boundary)


def plural(count: int, noun: str) -> str:
    return f"{count} {noun}" + ("" if count == 1 else "s")


def excerpt(text: str) -> str:
    return text[:EXCERPT_WIDTH] + ("…" if len(text) > EXCERPT_WIDTH else "")
