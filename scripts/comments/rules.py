from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

import lex_clang
import lex_hash
from files import Lang
from regions import comment_regions, directive_regions
from tokens import DIRECTIVE_PLACEMENTS, Kind, Placement, Region, Token

CXX_HINT = re.compile(r"/[/*]|\\[ \t\f\v]*[\r\n]|_Pragma|(?:#|%:)[ \t\f\v]*(?:if|elif|warning|pragma)\b")
NAMESPACE_CLOSER = re.compile(r"// namespace(?: [A-Za-z_]\w*(?:::[A-Za-z_]\w*)*)?")
FORMATTER_MARKER = re.compile(r"// clang-format (?:off|on)|/\* clang-format (?:off|on) \*/")
URL = r"(?:<https?://[^\s<>]+>|https?://[^\s<>]+)"
ARXIV = r"arXiv:(?:\d{4}\.\d{4,5}|[a-z-]+(?:\.[A-Z]{2})?/\d{7})(?:v\d+)?"
DOI = r"doi:10\.\d{4,9}/\S+"
EQUATIONS = r"(?:\((?:eqs?\. ?)?\d+[a-z]?(?: ?[-–,] ?\d+[a-z]?)*\)|eqs?\. ?\d+[a-z]?(?: ?[-–,] ?\d+[a-z]?)*)"
CITATION = re.compile(
    rf"(?:(?:See|From|Taken from|cf\.) )?(?:{URL}|{ARXIV}|{DOI})(?: \[[\w.-]+\])?(?:,? {EQUATIONS})?\.?"
)
LINE_OPENER = re.compile(r"^(?://+|#+|!+)")
BLOCK_OPENER = re.compile(r"^/\*+")
BLOCK_CLOSER = re.compile(r"\*+/$")
BLOCK_LEADER = re.compile(r"^\*+")


@dataclass(frozen=True)
class Violation:
    region: Region
    shown: Optional[Token] = None

    @property
    def token(self) -> Token:
        if self.shown is not None:
            return self.shown
        if self.region.placement in DIRECTIVE_PLACEMENTS:
            return self.region.tokens[0]
        return self.region.comments[0]

    @property
    def line(self) -> int:
        return self.token.line


def is_formatter_marker(token: Token) -> bool:
    return bool(FORMATTER_MARKER.fullmatch(token.text.rstrip()))


def previous_code(tokens: list[Token], index: int) -> Optional[Token]:
    for k in range(index - 1, -1, -1):
        token = tokens[k]
        if token.kind is Kind.NEWLINE:
            return None
        if token.kind not in (Kind.SPACE, Kind.COMMENT):
            return token
    return None


def comment_lines(token: Token) -> list[str]:
    text = token.text.rstrip()
    if text.startswith("/*"):
        lines = BLOCK_CLOSER.sub("", BLOCK_OPENER.sub("", text)).splitlines()
        lines = lines[:1] + [BLOCK_LEADER.sub("", line.strip()) for line in lines[1:]]
    else:
        lines = LINE_OPENER.sub("", text).splitlines()
    return [line.strip() for line in lines]


def is_citation(region: Region) -> bool:
    lines = [line for token in region.comments for line in comment_lines(token) if line]
    return bool(lines) and all(CITATION.fullmatch(line) for line in lines)


def is_tool_marker(region: Region, tokens: list[Token]) -> bool:
    comments = region.comments
    if len(comments) != 1:
        return False
    [comment] = comments
    text = comment.text.rstrip()
    if FORMATTER_MARKER.fullmatch(text):
        return True
    if region.placement is not Placement.TRAILING:
        return False
    if text == "//":
        return True
    before = previous_code(tokens, region.first)
    return bool(NAMESPACE_CLOSER.fullmatch(text)) and before is not None and before.text == "}"


def allowed(region: Region, tokens: list[Token], lang: Lang) -> bool:
    if region.placement in DIRECTIVE_PLACEMENTS:
        return False
    if lang is Lang.CXX and is_tool_marker(region, tokens):
        return True
    return is_citation(region)


def regions_of(tokens: list[Token], lang: Lang) -> list[Region]:
    if lang is not Lang.CXX:
        return comment_regions(tokens)
    found = comment_regions(tokens, is_formatter_marker) + directive_regions(tokens)
    return sorted(found, key=lambda r: (r.first, r.placement in DIRECTIVE_PLACEMENTS))


def check(tokens: list[Token], lang: Lang) -> list[Violation]:
    return [Violation(r) for r in regions_of(tokens, lang) if not allowed(r, tokens, lang)]


def lex(text: str, lang: Lang) -> list[Token]:
    if lang is Lang.CXX:
        return lex_clang.lex(text)
    return lex_hash.lex(text, lang)


def may_hold_comments(text: str, lang: Lang) -> bool:
    if lang is Lang.CXX:
        return bool(CXX_HINT.search(text))
    return lex_hash.may_hold_comments(text, lang)


def check_text(text: str, lang: Lang) -> list[Violation]:
    if not may_hold_comments(text, lang):
        return []
    return check(lex(text, lang), lang)
