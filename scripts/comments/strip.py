from __future__ import annotations

import difflib
import re
import shutil
import subprocess
from dataclasses import dataclass
from typing import Optional

import lex_clang
import lex_hash
from files import Lang
from lex_clang import ToolMissing
from regions import continues_run, is_separator, placements
from rules import check, lex, may_hold_comments
from tokens import DIRECTIVE_PLACEMENTS, Kind, Placement, Token

UNIFDEF = "unifdef"
DIRECTIVE_HINT = re.compile(r"#|%:")
SPLICE = re.compile(r"\\[ \t\f\v]*(?:\r\n|\r|\n)")
TIGHT = {"(", ")", "[", "]", "{", "}", ",", ";"}
CLOSERS = {")", "]", ",", ";"}
NAME = re.compile(r"[A-Za-z_]\w*")
ACCESS_SPECIFIER = re.compile(r"[ \t]*(?:public|protected|private)[ \t]*:[ \t]*\r?\n?")
SPACED_BEFORE_PAREN = {
    "and", "case", "catch", "co_await", "co_return", "co_yield", "constexpr", "delete", "for", "if", "new", "not",
    "or", "return", "switch", "throw", "while", "xor",
}


class StripError(RuntimeError):
    pass


def unifdef() -> str:
    path = shutil.which(UNIFDEF)
    if path is None:
        raise ToolMissing(f"{UNIFDEF} not found on PATH")
    return path


def placeholder_tag(text: str) -> str:
    tag = "__comments_literal_"
    while tag in text:
        tag += "x"
    return tag


def unifdef_k(text: str, tokens: Optional[list[Token]] = None) -> str:
    if not DIRECTIVE_HINT.search(text):
        return text
    exe = unifdef()
    tokens = lex_clang.lex(text) if tokens is None else tokens
    tag = placeholder_tag(text)
    literals: list[str] = []
    pieces = []
    for t in tokens:
        if t.kind is Kind.CODE and ("'" in t.text or '"' in t.text):
            pieces.append(f"{tag}{len(literals)}_")
            literals.append(t.text)
        else:
            pieces.append(t.text)
    masked = "".join(pieces).encode("utf-8", "surrogateescape")
    proc = subprocess.run([exe, "-k"], input=masked, capture_output=True)
    if proc.returncode not in (0, 1) or proc.stderr.strip():
        why = proc.stderr.decode("utf-8", "replace").strip().split("\n")[0]
        raise StripError(f"{UNIFDEF} -k exited with status {proc.returncode}: {why}")
    folded = proc.stdout.decode("utf-8", "surrogateescape")
    return re.sub(re.escape(tag) + r"(\d+)_", lambda m: literals[int(m.group(1))], folded)


def lone_separators(tokens: list[Token], drop: set) -> set:
    whole = sorted(k for k, p in placements(tokens).items() if p is Placement.WHOLE_LINE)
    blocks: list[list[int]] = []
    for k in whole:
        if blocks and continues_run(tokens, blocks[-1][-1], k):
            blocks[-1].append(k)
        else:
            blocks.append([k])
    found = set()
    for block in blocks:
        for pos, k in enumerate(block):
            if not is_separator(tokens[k]):
                continue
            above = next((j for j in reversed(block[:pos]) if not is_separator(tokens[j])), None)
            below = next((j for j in block[pos + 1 :] if not is_separator(tokens[j])), None)
            if above is None or below is None or above in drop or below in drop:
                found.add(k)
    return found


def dropped(tokens: list[Token], lang: Lang) -> set:
    drop = set()
    for v in check(tokens, lang):
        if v.region.placement in DIRECTIVE_PLACEMENTS:
            continue
        drop.update(i for i in range(v.region.first, v.region.last + 1) if tokens[i].kind is Kind.COMMENT)
    if lang is Lang.CXX:
        drop |= lone_separators(tokens, drop)
    return drop


def closes_up(before: Token, after: Token) -> bool:
    if after.text in CLOSERS:
        return True
    if after.text in ("(", "[") and NAME.fullmatch(before.text):
        return before.text not in SPACED_BEFORE_PAREN
    return False


def gap(tokens: list[Token], drop: set, before: Optional[int], seg: list[int], after: Optional[int]) -> str:
    if not any(i in drop for i in seg):
        return "".join(tokens[i].text for i in seg)
    first = next(k for k, i in enumerate(seg) if i in drop)
    lead = "".join(tokens[i].text for i in seg[:first] if tokens[i].kind is Kind.SPACE)
    rest = [tokens[i].text for i in seg[first:] if tokens[i].kind is Kind.SPACE and i not in drop]
    if after is None:
        return ""
    if before is None:
        return lead
    if SPLICE.search(lead):
        return lead
    splices = [w for w in rest if SPLICE.search(w)]
    if splices:
        return lead + "".join(w.lstrip(" \t\f\v") for w in splices) if lead else "".join(splices)
    left, right = tokens[before], tokens[after]
    if Kind.DIRECTIVE in (left.kind, right.kind) or Kind.COMMENT in (left.kind, right.kind):
        return lead or "".join(rest) or " "
    if closes_up(left, right):
        return ""
    if not lead and left.text not in TIGHT and right.text not in TIGHT:
        return " "
    return lead


@dataclass
class Row:
    kind: str
    text: str
    start: int
    in_string: bool = False
    modified: bool = False


def rows_of(tokens: list[Token], drop: set) -> list[Row]:
    rows = []
    offset = 0
    current: list[int] = []

    def finish() -> None:
        nonlocal offset
        start = offset
        offset += sum(len(tokens[i].text) for i in current)
        significant = [i for i in current if tokens[i].kind in (Kind.CODE, Kind.DIRECTIVE)
                       or (tokens[i].kind is Kind.COMMENT and i not in drop)]
        if not any(i in drop for i in current):
            text = "".join(tokens[i].text for i in current)
            blank = all(tokens[i].kind in (Kind.SPACE, Kind.NEWLINE) for i in current) and not SPLICE.search(text)
            rows.append(Row("blank" if blank else "line", text, start))
            return
        if not significant:
            rows.append(Row("deleted", "", start))
            return
        pieces = []
        before: Optional[int] = None
        seg: list[int] = []
        for i in current:
            if i in significant:
                pieces.append(gap(tokens, drop, before, seg, i))
                pieces.append(tokens[i].text)
                before, seg = i, []
            elif tokens[i].kind is Kind.NEWLINE:
                pieces.append(gap(tokens, drop, before, seg, None))
                pieces.append(tokens[i].text)
                seg = []
            else:
                seg.append(i)
        if seg:
            pieces.append(gap(tokens, drop, before, seg, None))
        rows.append(Row("line", "".join(pieces), start, modified=True))

    for i, t in enumerate(tokens):
        current.append(i)
        if t.kind is Kind.NEWLINE:
            finish()
            current = []
    if current:
        finish()
    return rows


def is_access_specifier(row: Optional[Row]) -> bool:
    return row is not None and ACCESS_SPECIFIER.fullmatch(row.text) is not None


def opens_block(row: Optional[Row], lang: Lang) -> bool:
    if row is None or row.text.rstrip().endswith("{"):
        return True
    return lang is Lang.CXX and is_access_specifier(row)


def closes_block(row: Optional[Row]) -> bool:
    return row is None or row.text.lstrip().startswith("}")


def newline_of(row: Optional[Row]) -> str:
    return "\r\n" if row is not None and row.text.endswith("\r\n") else "\n"


def junction(span: list[Row], previous: Optional[Row], following: Optional[Row], lang: Lang) -> list[Row]:
    runs: list[list[Row]] = [[]]
    for r in span:
        if r.kind == "deleted":
            runs.append([])
        else:
            runs[-1].append(r)
    top, bottom = opens_block(previous, lang), closes_block(following)
    if top and bottom:
        kept = min(runs[0], runs[-1], key=len)
    elif top:
        kept = runs[0]
    elif bottom:
        kept = runs[-1]
    else:
        kept = max(runs, key=len)
    if lang is Lang.CXX and not kept and not top and is_access_specifier(following):
        kept = [Row("blank", newline_of(previous), span[0].start)]
    return kept


def tidy(rows: list[Row], lang: Lang) -> str:
    kept: list[Row] = []
    previous: Optional[Row] = None
    i = 0
    while i < len(rows):
        row = rows[i]
        if row.in_string or row.kind == "line":
            if lang is Lang.CXX and row.modified and row.text.lstrip().startswith("}"):
                while kept and kept[-1].kind == "blank" and not kept[-1].in_string:
                    kept.pop()
            if row.kind != "deleted":
                kept.append(row)
                previous = row
            i += 1
            continue
        j = i
        while j < len(rows) and not rows[j].in_string and rows[j].kind != "line":
            j += 1
        span = rows[i:j]
        following = rows[j] if j < len(rows) else None
        if any(r.kind == "deleted" for r in span):
            kept.extend(junction(span, previous, following, lang))
        else:
            kept.extend(span)
        i = j
    return "".join(r.text for r in kept)


def removed_lines(text: str, folded: str) -> list[int]:
    a, b = text.splitlines(keepends=True), folded.splitlines(keepends=True)
    starts = [0]
    for line in b:
        starts.append(starts[-1] + len(line))
    found = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if op in ("delete", "replace") and i2 - i1 > j2 - j1:
            found.append(starts[j1])
    return found


def strip_tokens(
    tokens: list[Token], lang: Lang, strings: list[tuple[int, int]] = (), removed: list[int] = ()
) -> str:
    drop = dropped(tokens, lang)
    if not drop and not removed:
        return "".join(t.text for t in tokens)
    rows = rows_of(tokens, drop)
    for row in rows:
        row.in_string = any(start < row.start < end for start, end in strings)
    for offset in sorted(removed, reverse=True):
        at = next((k for k, row in enumerate(rows) if row.start >= offset), len(rows))
        rows.insert(at, Row("deleted", "", offset))
    return tidy(rows, lang)


def strip(text: str, lang: Lang) -> str:
    tokens = None
    removed: list[int] = []
    if lang is Lang.CXX and DIRECTIVE_HINT.search(text):
        tokens = lex_clang.lex(text)
        folded = unifdef_k(text, tokens)
        if folded != text:
            removed = removed_lines(text, folded)
            text, tokens = folded, lex_clang.lex(folded)
    if tokens is None:
        if not may_hold_comments(text, lang):
            return text
        tokens = lex(text, lang)
    strings = lex_hash.par_strings(text) if lang is Lang.PAR else []
    return strip_tokens(tokens, lang, strings, removed)
