from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Optional

import lex_cxx
import lex_hash
from files import Lang
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
WORD = re.compile(r"[A-Za-z_]\w*|\.?\d[\w.']*|\S")
NUMBER = re.compile(r"\.?\d")
NAME = re.compile(r"[A-Za-z_]")
DEFINED = re.compile(r"\bdefined\s*(?:\(\s*[A-Za-z_]\w*\s*\)|[A-Za-z_]\w*)")
GUARD = re.compile(r"\s*(?:#|%:)\s*(?:if|elif)\b(.*)", re.S)
MESSAGE = re.compile(
    r"\s*(?:#|%:)\s*(?:warning\b|pragma\s+(?:message|GCC\s+warning)\b)"
    r'|.*?\b_Pragma\s*\(\s*(?:u8|u|U|L)?(?:R"[^(]*\(|")\s*(?:message|GCC\s+warning)\b',
    re.S,
)


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


def is_constant(condition: str) -> bool:
    words = WORD.findall(DEFINED.sub(" ", condition))
    return any(NUMBER.match(w) for w in words) and not any(NAME.match(w) for w in words)


def logical_lines(tokens: list[Token]) -> Iterable[tuple[int, str]]:
    line: Optional[int] = None
    pieces: list[str] = []
    for token in tokens + [Token(Kind.NEWLINE, "", -1, -1)]:
        if token.kind is Kind.NEWLINE:
            if line is not None:
                yield line, "".join(pieces)
            line, pieces = None, []
        elif token.kind in CODE_KINDS:
            line = token.line if line is None else line
            pieces.append(token.text)
        else:
            pieces.append(" ")


def directive_units(tokens: list[Token]) -> Iterable[Unit]:
    for line, code in logical_lines(tokens):
        guard = GUARD.match(code)
        if (guard and is_constant(guard.group(1))) or MESSAGE.match(code):
            yield Unit(line, ("directive", " ".join(WORD.findall(code))), " ".join(code.split()), False)


def lex(text: str, lang: Lang) -> list[Token]:
    if lang is Lang.CXX:
        return lex_cxx.lex(text)
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
