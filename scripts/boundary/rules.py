from __future__ import annotations

import os
import re
import sys
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Optional

TOOL = os.path.dirname(os.path.abspath(__file__))
COMMENTS = os.path.join(os.path.dirname(TOOL), "comments")
if COMMENTS not in sys.path:
    sys.path.append(COMMENTS)

from files import Lang, changes_since, covered, is_regular, lang_of, read_text, show  # noqa: E402

MARKER = "CARPETX_INTERNAL"
PRIVATE_DIR = "CarpetX/src/"
PUBLIC_DIR = "CarpetX/src/include/"
MAKE_DEFN = "CarpetX/src/make.code.defn"
LAYERS = ("util", "traversal", "hierarchy", "comm", "regrid", "ops", "io", "schedule", "api", "flesh")


def layer_path(layer: str, basename: str) -> str:
    assert layer in LAYERS, layer
    return f"{PRIVATE_DIR}{layer}/{basename}"


def layer_of(name: str) -> Optional[str]:
    if not name.startswith(PRIVATE_DIR):
        return None
    parts = name[len(PRIVATE_DIR) :].split("/")
    return parts[0] if len(parts) == 2 and parts[0] in LAYERS else None


CHECKED = (Lang.CXX, Lang.MAKE)
MARKER_WORD = re.compile(r"CARPETX_INTERNAL\b")
GUARD = re.compile(r"^\s*#\s*ifndef\s+CARPETX_INTERNAL\s*$")
INCLUDE_INTO_SRC = re.compile(r'^\s*#\s*include\s*(?:"[^"]*CarpetX/src[^"]*"|<[^>]*CarpetX/src[^>]*>)')
VALIDITY_FILES = (layer_path("hierarchy", "valid.cxx"), layer_path("hierarchy", "valid.hxx"))
VALIDITY_WORD = re.compile(
    r"(\b(error_if_invalid|warn_if_invalid|poison_invalid_g[fa]|check_valid_g[fa])"
    r"|(\.|->)(set_all|set_int|set_outer|set_ghosts|set_invalid|set_valid))\s*\("
)
TRAVERSAL_FILES = (layer_path("traversal", "traversal.cxx"), layer_path("traversal", "traversal.hxx"))
TRAVERSAL_WORD = re.compile(r"\b(active_levels|tail_bin|recovery_traversal|last_routine|recover_iteration)\b")
DOOR_FILE = layer_path("hierarchy", "hierarchy.cxx")
DOOR_WORD = re.compile(r"\bghext\b")
SYNC_FILES = (
    layer_path("comm", "comm.hxx"),
    layer_path("comm", "sync_restrict.cxx"),
)
DIRECTIONS_WORD = re.compile(r"\bdirections\b")
FLESH_SYNC_OVERLOAD = "SyncGroupsOnActiveLevels"
VARINDEX_WORD = re.compile(r"\b(firstvarindex|CCTK_FirstVarIndexI\s*\((?:[^()]|\([^()]*\))*\))\s*\+")
RECORD_VAR_FILE = layer_path("hierarchy", "driver.hxx")
RECORD_VAR = re.compile(r"\bvar_t\s*\{\s*firstvarindex\s*\+")
WINDOW_WORD = re.compile(r"\bactive_levels_t\b")
INDEX_PARAM = re.compile(r"\bint\s+(gi|vi|tl|patch|level|min_level|max_level|min_patch|max_patch)\b")
INDEX_FIXED = {
    PUBLIC_DIR + "carpetx.hxx": re.compile(r"\b(level_t|window_t|levels|narrow)\s*\("),
    layer_path("schedule", "schedule.hxx"): re.compile(
        r"\b(QueryGroupStorageB|GroupStorageIncrease|GroupStorageDecrease)\s*\("
    ),
    layer_path("comm", "comm.hxx"): re.compile(r"\b(SyncGroupsOnActiveLevels)\s*\("),
    layer_path("hierarchy", "driver.hxx"): re.compile(
        r"\b(ErrorEst|MakeNewLevelFromScratch|MakeNewLevelFromCoarse|RemakeLevel|ClearLevel)\s*\("
    ),
}
CODEC_FILES = (
    layer_path("io", "io_openpmd_checkpoint.cxx"),
    layer_path("io", "io_openpmd_names.hxx"),
    layer_path("io", "io_silo_checkpoint.cxx"),
    layer_path("io", "io_silo_names.hxx"),
)
CODEC_WORD = re.compile(
    r'\bghext->|\bdriver\s*\(\s*\)|\bGHExt\b|\bLevelData\b|\bGroupData\b|^\s*#\s*include\s*["<]hierarchy/driver\.hxx[">]'
)
STATEMENT_END = re.compile(r"[;{}]")
EXCERPT_WIDTH = 72
PREAMBLE = "A thorn reaches CarpetX only through its public headers (see agent_docs/carpetx-public-api.md)."

Files = Callable[[str, Lang], bool]
Allowed = Callable[[str, str, str], bool]


@dataclass(frozen=True)
class Finding:
    line: int
    rule: str
    text: str


@dataclass(frozen=True)
class Rule:
    name: str
    files: Files
    word: re.Pattern
    allowed: Allowed
    text: str
    noun: str
    where: str
    clean: str

    def phrase(self, noun: str) -> str:
        return f"{noun} {self.where}" if self.where else noun

    def count(self, n: int) -> str:
        return self.phrase(plural(n, self.noun))


def is_private(name: str) -> bool:
    return name.startswith(PRIVATE_DIR) and not name.startswith(PUBLIC_DIR)


def checked(name: str, lang: Lang) -> bool:
    return True


def public_cxx(name: str, lang: Lang) -> bool:
    return lang is Lang.CXX and not is_private(name)


def private_cxx(name: str, lang: Lang) -> bool:
    return lang is Lang.CXX and is_private(name)


def carpetx_cxx(name: str, lang: Lang) -> bool:
    return lang is Lang.CXX and name.startswith(PRIVATE_DIR)


def carpetx_header(name: str, lang: Lang) -> bool:
    return carpetx_cxx(name, lang) and name.endswith(".hxx")


def codec_cxx(name: str, lang: Lang) -> bool:
    return lang is Lang.CXX and name in CODEC_FILES


def in_files(names: tuple[str, ...]) -> Callable[..., bool]:
    return lambda name, *_: name in names


def is_name(path: str) -> Callable[..., bool]:
    return lambda name, *_: name == path


def never(name: str, line: str, statement: str) -> bool:
    return False


def marker_allowed(name: str, line: str, statement: str) -> bool:
    if name == MAKE_DEFN:
        return True
    return is_private(name) and lang_of(name) is Lang.CXX and bool(GUARD.match(line))


def directions_allowed(name: str, line: str, statement: str) -> bool:
    return FLESH_SYNC_OVERLOAD in statement


def varindex_allowed(name: str, line: str, statement: str) -> bool:
    return name == RECORD_VAR_FILE and bool(RECORD_VAR.search(line))


def index_allowed(name: str, line: str, statement: str) -> bool:
    fixed = INDEX_FIXED.get(name)
    return fixed is not None and bool(fixed.search(statement))


RULES = (
    Rule(
        "include",
        files=public_cxx,
        word=INCLUDE_INTO_SRC,
        allowed=never,
        text="include: an #include whose path reaches into CarpetX/src; include <carpetx.hxx> instead.",
        noun="private include",
        where="",
        clean="no private CarpetX includes",
    ),
    Rule(
        "marker",
        files=checked,
        word=MARKER_WORD,
        allowed=marker_allowed,
        text="marker: CARPETX_INTERNAL belongs only in CarpetX/src/make.code.defn and in the\n"
        "#ifndef CARPETX_INTERNAL guard of a private CarpetX header.",
        noun="stray marker",
        where="",
        clean="no stray CARPETX_INTERNAL",
    ),
    Rule(
        "validity",
        files=private_cxx,
        word=VALIDITY_WORD,
        allowed=in_files(VALIDITY_FILES),
        text="validity: a validity kernel or a why_valid_t setter is named only in CarpetX/src/hierarchy/valid.cxx\n"
        "and valid.hxx; every other site opens a write scope or calls require or invalidate.",
        noun="validity call",
        where="outside valid.cxx",
        clean="no validity call outside valid.cxx",
    ),
    Rule(
        "directions",
        files=in_files(SYNC_FILES),
        word=DIRECTIONS_WORD,
        allowed=directions_allowed,
        text="directions: the sync layer takes a window and a list of gi_t; only the flesh-facing\n"
        "SyncGroupsOnActiveLevels receives the flesh's directions array, and discards it.",
        noun="directions parameter",
        where="in the sync layer",
        clean="no directions in the sync layer",
    ),
    Rule(
        "varindex",
        files=carpetx_cxx,
        word=VARINDEX_WORD,
        allowed=varindex_allowed,
        text="varindex: a flesh variable index is made from a group-relative one only by GroupRecord::var;\n"
        "call record.var(vi) instead of adding to firstvarindex or CCTK_FirstVarIndexI(gi).",
        noun="firstvarindex addition",
        where="outside GroupRecord::var",
        clean="no firstvarindex addition outside GroupRecord::var",
    ),
    Rule(
        "window",
        files=carpetx_cxx,
        word=WINDOW_WORD,
        allowed=never,
        text='window: window_t is the one "which levels" type; active_levels_t is gone, and a window is\n'
        "built only by active_window(), all_levels(), window_t::one_level, window_t::levels and narrow.",
        noun="active_levels_t use",
        where="",
        clean="no active_levels_t",
    ),
    Rule(
        "traversal",
        files=private_cxx,
        word=TRAVERSAL_WORD,
        allowed=in_files(TRAVERSAL_FILES),
        text="traversal: a traversal slot (active_levels, tail_bin, recovery_traversal, last_routine,\n"
        "recover_iteration) is named only in CarpetX/src/traversal/traversal.cxx and traversal.hxx; a driver\n"
        "phase opens a scope, parameter recovery calls record_recovered_iteration() once, and a reader calls\n"
        "scheduled_window(), scheduled_tail_bin(), in_recovery_traversal(), scheduled_routine() or\n"
        "recovered_iteration().",
        noun="traversal use",
        where="outside traversal.cxx",
        clean="no active_levels outside traversal.cxx",
    ),
    Rule(
        "index",
        files=carpetx_header,
        word=INDEX_PARAM,
        allowed=index_allowed,
        text="index: a CarpetX header names a group, variable, time level, patch or level by its type\n"
        "(gi_t, vi_t, tl_t, patch_t, level_t, window_t), never a bare int, except in the signatures\n"
        "the flesh and AMReX fix, the level_t and window_t constructors, and window_t::levels and narrow.",
        noun="bare index int",
        where="in a CarpetX header",
        clean="no bare index int in a CarpetX header",
    ),
    Rule(
        "codec",
        files=codec_cxx,
        word=CODEC_WORD,
        allowed=never,
        text="codec: a checkpoint codec (io_*_checkpoint.cxx and its io_*_names.hxx) sees only the checkpoint\n"
        "record and its own library: no ghext->, driver(), GHExt, LevelData, GroupData or hierarchy/driver.hxx.",
        noun="hierarchy use",
        where="in a checkpoint codec",
        clean="no hierarchy access in a checkpoint codec",
    ),
    Rule(
        "door",
        files=private_cxx,
        word=DOOR_WORD,
        allowed=is_name(DOOR_FILE),
        text="door: the hierarchy pointer ghext is named only in CarpetX/src/hierarchy/hierarchy.cxx; every other\n"
        "file calls driver().",
        noun="raw hierarchy name",
        where="outside hierarchy.cxx",
        clean="the hierarchy has one door",
    ),
)
RULE = "\n".join([PREAMBLE] + [rule.text for rule in RULES])


def findings(name: str, text: str, lang: Optional[Lang]) -> list[Finding]:
    if lang not in CHECKED:
        return []
    rules = [rule for rule in RULES if rule.files(name, lang)]
    found = []
    statement = ""
    for number, line in enumerate(text.splitlines(), 1):
        statement += line + "\n"
        found.extend(
            Finding(number, rule.name, line.strip())
            for rule in rules
            if rule.word.search(line) and not rule.allowed(name, line, statement)
        )
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


def series(items: list[str]) -> str:
    return ", ".join(items[:-1]) + ", and " + items[-1]


def summary(found: list[tuple[str, Finding]]) -> str:
    if not found:
        return "✓ boundary: " + series([rule.clean for rule in RULES])
    counts = Counter(f.rule for _, f in found)
    return "✗ boundary: " + ", ".join(rule.count(counts[rule.name]) for rule in RULES if counts[rule.name])


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
