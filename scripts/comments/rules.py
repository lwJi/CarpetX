from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Optional

import lex_clang
import lex_hash
from files import Lang
from regions import directive_regions
from tokens import Kind, Token

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
LINE_BREAK = re.compile(r"\r\n|\r|\n")
LINE_OPENER = re.compile(r"^(?://+|#+|!+)")
CODE_KINDS = (Kind.CODE, Kind.DIRECTIVE)


@dataclass(frozen=True)
class Unit:
    line: int
    key: tuple
    text: str
    allowed: bool


def comment_words(text: str) -> list[str]:
    text = text.rstrip()
    if text.startswith("/*"):
        body = text[2:-2] if len(text) >= 4 and text.endswith("*/") else text[2:]
        lines = [line.strip().strip("*") for line in LINE_BREAK.split(body)]
    else:
        lines = LINE_BREAK.split(LINE_OPENER.sub("", text))
    return [" ".join(line.split()) for line in lines]


def is_tool_marker(token: Token, before: Optional[str]) -> bool:
    text = token.text.rstrip()
    if FORMATTER_MARKER.fullmatch(text):
        return True
    return bool(NAMESPACE_CLOSER.fullmatch(text)) and before is not None and before.endswith("}")


def comment_units(token: Token, before: Optional[str], lang: Lang) -> Iterable[Unit]:
    shown = LINE_BREAK.split(token.text.rstrip())
    marker = lang is Lang.CXX and is_tool_marker(token, before)
    for offset, words in enumerate(comment_words(token.text)):
        if words:
            allowed = marker or bool(CITATION.fullmatch(words))
            yield Unit(token.line + offset, ("comment", words), shown[offset].strip(), allowed)


def directive_units(tokens: list[Token]) -> Iterable[Unit]:
    for r in directive_regions(tokens):
        code = " ".join(t.text for t in r.tokens if t.kind in CODE_KINDS)
        yield Unit(r.line, ("directive", code), r.code, False)


def lex(text: str, lang: Lang) -> list[Token]:
    if lang is Lang.CXX:
        return lex_clang.lex(text)
    return lex_hash.lex(text, lang)


def may_hold_comments(text: str, lang: Lang) -> bool:
    if lang is Lang.CXX:
        return bool(CXX_HINT.search(text))
    return lex_hash.may_hold_comments(text, lang)


def units(text: str, lang: Lang) -> list[Unit]:
    if not may_hold_comments(text, lang):
        return []
    tokens = lex(text, lang)
    found: list[Unit] = []
    before: Optional[str] = None
    for token in tokens:
        if token.kind is Kind.NEWLINE:
            before = None
        elif token.kind is Kind.COMMENT:
            found.extend(comment_units(token, before, lang))
        elif token.kind in CODE_KINDS:
            before = token.text
    if lang is Lang.CXX:
        found.extend(directive_units(tokens))
    return sorted(found, key=lambda u: u.line)
