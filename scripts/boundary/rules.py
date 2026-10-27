from __future__ import annotations

import os
import re
import sys
from collections import Counter
from dataclasses import dataclass
from typing import Optional

TOOL = os.path.dirname(os.path.abspath(__file__))
COMMENTS = os.path.join(os.path.dirname(TOOL), "comments")
if COMMENTS not in sys.path:
    sys.path.append(COMMENTS)

from files import Lang, changes_since, covered, is_regular, read_text, show  # noqa: E402

MARKER = "CARPETX_INTERNAL"
PRIVATE_DIR = "CarpetX/src/"
PUBLIC_DIR = "CarpetX/src/include/"
MAKE_DEFN = "CarpetX/src/make.code.defn"
CHECKED = (Lang.CXX, Lang.MAKE)
MARKER_WORD = re.compile(r"CARPETX_INTERNAL\b")
GUARD = re.compile(r"^\s*#\s*ifndef\s+CARPETX_INTERNAL\s*$")
INCLUDE = re.compile(r'^\s*#\s*include\s*(?:"([^"]*)"|<([^>]*)>)')
PRIVATE_PATH = "CarpetX/src"
VALIDITY_FILES = ("CarpetX/src/valid.cxx", "CarpetX/src/valid.hxx")
VALIDITY_WORD = re.compile(
    r"(\b(error_if_invalid|warn_if_invalid|poison_invalid_g[fa]|check_valid_g[fa])"
    r"|(\.|->)(set_all|set_int|set_outer|set_ghosts|set_invalid|set_valid))\s*\("
)
SYNC_FILES = (
    "CarpetX/src/schedule.hxx",
    "CarpetX/src/schedule.cxx",
    "CarpetX/src/sync_restrict.cxx",
    "CarpetX/src/sync_restrict_internal.hxx",
)
DIRECTIONS_WORD = re.compile(r"\bdirections\b")
FLESH_SYNC_OVERLOAD = "SyncGroupsOnActiveLevels"
VARINDEX_WORD = re.compile(r"\b(firstvarindex|CCTK_FirstVarIndexI\s*\((?:[^()]|\([^()]*\))*\))\s*\+")
RECORD_VAR_FILE = "CarpetX/src/driver.hxx"
RECORD_VAR = re.compile(r"\bvar_t\s*\{\s*firstvarindex\s*\+")
WINDOW_WORD = re.compile(r"\bactive_levels_t\b")
INDEX_PARAM = re.compile(r"\bint\s+(gi|vi|tl|patch|level|min_level|max_level|min_patch|max_patch)\b")
INDEX_FIXED = {
    "CarpetX/src/include/carpetx.hxx": re.compile(r"\b(level_t|window_t|levels|narrow)\s*\("),
    "CarpetX/src/schedule.hxx": re.compile(
        r"\b(QueryGroupStorageB|GroupStorageIncrease|GroupStorageDecrease|SyncGroupsOnActiveLevels)\s*\("
    ),
    "CarpetX/src/driver.hxx": re.compile(
        r"\b(ErrorEst|MakeNewLevelFromScratch|MakeNewLevelFromCoarse|RemakeLevel|ClearLevel)\s*\("
    ),
}
CODEC_FILES = (
    "CarpetX/src/io_openpmd_checkpoint.cxx",
    "CarpetX/src/io_openpmd_names.hxx",
    "CarpetX/src/io_silo_checkpoint.cxx",
    "CarpetX/src/io_silo_names.hxx",
)
CODEC_WORD = re.compile(
    r'\bghext->|\bdriver\s*\(\s*\)|\bGHExt\b|\bLevelData\b|\bGroupData\b|^\s*#\s*include\s*["<]driver\.hxx[">]'
)
STATEMENT_END = re.compile(r"[;{}]")
EXCERPT_WIDTH = 72

RULE = (
    "A thorn reaches CarpetX only through its public headers (see agent_docs/carpetx-public-api.md).\n"
    "include: an #include whose path reaches into CarpetX/src; include <carpetx.hxx> instead.\n"
    "marker: CARPETX_INTERNAL belongs only in CarpetX/src/make.code.defn and in the\n"
    "#ifndef CARPETX_INTERNAL guard of a private CarpetX header.\n"
    "validity: a validity kernel or a why_valid_t setter is named only in CarpetX/src/valid.cxx\n"
    "and valid.hxx; every other site opens a write scope or calls require or invalidate.\n"
    "directions: the sync layer takes a window and a list of gi_t; only the flesh-facing\n"
    "SyncGroupsOnActiveLevels receives the flesh's directions array, and discards it.\n"
    "varindex: a flesh variable index is made from a group-relative one only by GroupRecord::var;\n"
    "call record.var(vi) instead of adding to firstvarindex or CCTK_FirstVarIndexI(gi).\n"
    "window: window_t is the one \"which levels\" type; active_levels_t is gone, and a window is\n"
    "built only by active_window(), all_levels(), window_t::one_level, window_t::levels and narrow.\n"
    "index: a CarpetX header names a group, variable, time level, patch or level by its type\n"
    "(gi_t, vi_t, tl_t, patch_t, level_t, window_t), never a bare int, except in the signatures\n"
    "the flesh and AMReX fix, the level_t and window_t constructors, and window_t::levels and narrow.\n"
    "codec: a checkpoint codec (io_*_checkpoint.cxx and its io_*_names.hxx) sees only the checkpoint\n"
    "record and its own library: no ghext->, driver(), GHExt, LevelData, GroupData or driver.hxx."
)


@dataclass(frozen=True)
class Finding:
    line: int
    rule: str
    text: str


def is_private(name: str) -> bool:
    return name.startswith(PRIVATE_DIR) and not name.startswith(PUBLIC_DIR)


def marker_allowed(name: str, lang: Lang, line: str) -> bool:
    if name == MAKE_DEFN:
        return True
    return is_private(name) and lang is Lang.CXX and bool(GUARD.match(line))


def directions_allowed(name: str, statement: str) -> bool:
    return name in SYNC_FILES and FLESH_SYNC_OVERLOAD in statement


def varindex_allowed(name: str, line: str) -> bool:
    return name == RECORD_VAR_FILE and bool(RECORD_VAR.search(line))


def index_allowed(name: str, statement: str) -> bool:
    fixed = INDEX_FIXED.get(name)
    return fixed is not None and bool(fixed.search(statement))


def findings(name: str, text: str, lang: Optional[Lang]) -> list[Finding]:
    if lang not in CHECKED:
        return []
    found = []
    statement = ""
    for number, line in enumerate(text.splitlines(), 1):
        statement += line + "\n"
        if MARKER_WORD.search(line) and not marker_allowed(name, lang, line):
            found.append(Finding(number, "marker", line.strip()))
        if lang is Lang.CXX and not is_private(name):
            match = INCLUDE.match(line)
            if match and PRIVATE_PATH in (match.group(1) or match.group(2) or ""):
                found.append(Finding(number, "include", line.strip()))
        if lang is Lang.CXX and is_private(name) and name not in VALIDITY_FILES and VALIDITY_WORD.search(line):
            found.append(Finding(number, "validity", line.strip()))
        if name in SYNC_FILES and DIRECTIONS_WORD.search(line) and not directions_allowed(name, statement):
            found.append(Finding(number, "directions", line.strip()))
        if lang is Lang.CXX and name.startswith(PRIVATE_DIR) and VARINDEX_WORD.search(line) and not varindex_allowed(name, line):
            found.append(Finding(number, "varindex", line.strip()))
        if lang is Lang.CXX and name.startswith(PRIVATE_DIR) and WINDOW_WORD.search(line):
            found.append(Finding(number, "window", line.strip()))
        if (
            lang is Lang.CXX
            and name.startswith(PRIVATE_DIR)
            and name.endswith(".hxx")
            and INDEX_PARAM.search(line)
            and not index_allowed(name, statement)
        ):
            found.append(Finding(number, "index", line.strip()))
        if lang is Lang.CXX and name in CODEC_FILES and CODEC_WORD.search(line):
            found.append(Finding(number, "codec", line.strip()))
        if STATEMENT_END.search(line):
            statement = ""
    return found


def new_findings(before: list[Finding], after: list[Finding]) -> list[Finding]:
    old = Counter((f.rule, f.text) for f in before)
    found = []
    for f in after:
        if old[(f.rule, f.text)] > 0:
            old[(f.rule, f.text)] -= 1
        else:
            found.append(f)
    return found


def since_findings(root: str, base: str) -> list[tuple[str, Finding]]:
    found = []
    for name, old in sorted(changes_since(root, base).items()):
        lang = covered(name, root)
        if lang not in CHECKED:
            continue
        path = os.path.join(root, name)
        after = read_text(path) if is_regular(path) else ""
        before = findings(old, show(root, base, old), lang) if old is not None else []
        found.extend((name, f) for f in new_findings(before, findings(name, after, lang)))
    return found


def plural(count: int, noun: str) -> str:
    return f"{count} {noun}" + ("" if count == 1 else "s")


def summary(found: list[tuple[str, Finding]]) -> str:
    includes = sum(1 for _, f in found if f.rule == "include")
    markers = sum(1 for _, f in found if f.rule == "marker")
    validity = sum(1 for _, f in found if f.rule == "validity")
    directions = sum(1 for _, f in found if f.rule == "directions")
    varindex = sum(1 for _, f in found if f.rule == "varindex")
    window = sum(1 for _, f in found if f.rule == "window")
    index = sum(1 for _, f in found if f.rule == "index")
    codec = sum(1 for _, f in found if f.rule == "codec")
    if not found:
        return (
            "✓ boundary: no private CarpetX includes, no stray CARPETX_INTERNAL, "
            "no validity call outside valid.cxx, no directions in the sync layer, "
            "no firstvarindex addition outside GroupRecord::var, no active_levels_t, "
            "no bare index int in a CarpetX header, no hierarchy access in a checkpoint codec"
        )
    parts = (
        ([plural(includes, "private include")] if includes else [])
        + ([plural(markers, "stray marker")] if markers else [])
        + ([plural(validity, "validity call") + " outside valid.cxx"] if validity else [])
        + ([plural(directions, "directions parameter") + " in the sync layer"] if directions else [])
        + ([plural(varindex, "firstvarindex addition") + " outside GroupRecord::var"] if varindex else [])
        + ([plural(window, "active_levels_t use")] if window else [])
        + ([plural(index, "bare index int") + " in a CarpetX header"] if index else [])
        + ([plural(codec, "hierarchy use") + " in a checkpoint codec"] if codec else [])
    )
    return "✗ boundary: " + ", ".join(parts)


def excerpt(text: str) -> str:
    return text[:EXCERPT_WIDTH] + ("…" if len(text) > EXCERPT_WIDTH else "")


def render(found: list[tuple[str, Finding]]) -> str:
    if not found:
        return summary(found)
    found = sorted(found, key=lambda e: (e[0], e[1].line))
    places = [f"{name}:{f.line}" for name, f in found]
    width = max(len(p) for p in places)
    out = [f"{place.ljust(width)}   {f.rule.ljust(8)}   {excerpt(f.text)}" for place, (_, f) in zip(places, found)]
    out.extend(RULE.split("\n"))
    out.append(summary(found))
    return "\n".join(out)
