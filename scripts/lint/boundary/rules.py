from __future__ import annotations

import re
import textwrap
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Optional

from checks import excerpt, plural
from files import Lang, lang_of

MARKER = "CARPETX_INTERNAL"
PRIVATE_DIR = "CarpetX/src/"
PUBLIC_DIR = "CarpetX/src/include/"
MAKE_DEFN = "CarpetX/src/make.code.defn"
LAYERS = ("util", "traversal", "hierarchy", "comm", "regrid", "ops", "io", "schedule", "api", "flesh")
HIERARCHY_LAYERS = LAYERS[LAYERS.index("hierarchy") :]
UNDER_HIERARCHY = LAYERS[: LAYERS.index("hierarchy")]
ROOT_HEADER = "internal.hxx"


def layer_path(layer: str, basename: str) -> str:
    assert layer in LAYERS, layer
    return f"{PRIVATE_DIR}{layer}/{basename}"


def layer_of(name: str) -> Optional[str]:
    if not name.startswith(PRIVATE_DIR):
        return None
    parts = name[len(PRIVATE_DIR) :].split("/")
    return parts[0] if len(parts) == 2 and parts[0] in LAYERS else None


def below(layer: str) -> tuple[str, ...]:
    assert layer in LAYERS, layer
    return LAYERS[: LAYERS.index(layer) + 1]


def layers_of(names: tuple[str, ...]) -> tuple[str, ...]:
    found = tuple(dict.fromkeys(layer_of(name) for name in names))
    assert None not in found, names
    return found


def dirs(layers: tuple[str, ...], conjunction: str) -> str:
    return f" {conjunction} ".join(f"{layer}/" for layer in layers)


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
TRAVERSAL_WORD = re.compile(
    r"\b(active_levels|tail_bin|recovery_traversal|last_routine|recover_iteration|integrator_config|integrator_sealed)\b"
)
DOOR_FILE = layer_path("hierarchy", "hierarchy.cxx")
DOOR_WORD = re.compile(r"\bghext\b")
DRIVER_CALL = re.compile(r"\bdriver\s*\(\s*\)")
MUTABLE_DRIVER_CALL = re.compile(r"\bmutable_driver\s*\(\s*\)")
WRITE_LAYER = "hierarchy"
GHEXT_OR_DRIVER_CALL = re.compile(f"{DOOR_WORD.pattern}|{DRIVER_CALL.pattern}|{MUTABLE_DRIVER_CALL.pattern}")
LAYER_INCLUDE = re.compile(r'^\s*#\s*include\s*(?:"([^"]*)"|<((?:' + "|".join(LAYERS) + r')/[^>]*)>)')
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
INDEX_SOURCE_LAYERS = ("io",)
STORAGE_OVERLOADS = ("QueryGroupStorageB", "GroupStorageIncrease", "GroupStorageDecrease")
AMREX_CALLBACKS = ("ErrorEst", "MakeNewLevelFromScratch", "MakeNewLevelFromCoarse", "RemakeLevel", "ClearLevel")
INDEX_FIXED = {
    PUBLIC_DIR + "carpetx.hxx": ("level_t", "window_t", "levels", "narrow"),
    layer_path("schedule", "schedule.hxx"): STORAGE_OVERLOADS,
    layer_path("schedule", "storage.cxx"): STORAGE_OVERLOADS,
    layer_path("flesh", "startup.cxx"): ("ArrayGroupSizeB", "GroupDynamicData"),
    layer_path("comm", "comm.hxx"): (FLESH_SYNC_OVERLOAD,),
    layer_path("comm", "sync_restrict.cxx"): (FLESH_SYNC_OVERLOAD,),
    layer_path("hierarchy", "driver.hxx"): AMREX_CALLBACKS,
    layer_path("regrid", "regrid.cxx"): AMREX_CALLBACKS,
}
INDEX_FIXED_CALL = {name: re.compile(r"\b(" + "|".join(calls) + r")\s*\(") for name, calls in INDEX_FIXED.items()}
RECORDS_LAYERS = ("io",)
RECORDS_CALLS = (
    "NumGroups",
    "GroupTypeI",
    "GroupData",
    "GroupDimI",
    "NumVarsInGroupI",
    "FirstVarIndexI",
    "GroupIndexFromVarI",
)
RECORDS_WORD = re.compile(r"\bCCTK_(" + "|".join(RECORDS_CALLS) + r")\s*\(")
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
RULE_WIDTH = 104
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
    home: tuple[str, ...] = ()

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


def index_files(name: str, lang: Lang) -> bool:
    return carpetx_header(name, lang) or (carpetx_cxx(name, lang) and layer_of(name) in INDEX_SOURCE_LAYERS)


def records_files(name: str, lang: Lang) -> bool:
    return carpetx_cxx(name, lang) and layer_of(name) in RECORDS_LAYERS


def codec_cxx(name: str, lang: Lang) -> bool:
    return lang is Lang.CXX and name in CODEC_FILES


def in_files(names: tuple[str, ...]) -> Callable[..., bool]:
    return lambda name, *_: name in names


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
    fixed = INDEX_FIXED_CALL.get(name)
    return fixed is not None and bool(fixed.search(statement))


def layer_allowed(name: str, line: str, statement: str) -> bool:
    quoted, angled = LAYER_INCLUDE.search(line).groups()
    path = quoted if quoted is not None else angled
    if path == ROOT_HEADER:
        return True
    parts = path.split("/")
    layer = layer_of(name)
    return len(parts) > 1 and ".." not in parts and layer is not None and parts[0] in below(layer)


def door_allowed(name: str, line: str, statement: str) -> bool:
    if DOOR_WORD.search(line) and name != DOOR_FILE:
        return False
    if MUTABLE_DRIVER_CALL.search(line) and layer_of(name) != WRITE_LAYER:
        return False
    return not DRIVER_CALL.search(line) or layer_of(name) in HIERARCHY_LAYERS


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
        home=layers_of(VALIDITY_FILES),
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
        home=layers_of(SYNC_FILES),
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
        home=layers_of((RECORD_VAR_FILE,)),
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
        "recover_iteration, and the integrator's integrator_config and integrator_sealed) is named only in\n"
        "CarpetX/src/traversal/traversal.cxx and traversal.hxx; a driver phase opens a scope, parameter recovery\n"
        "calls record_recovered_iteration() once, register_integrator calls record_integrator() once, and a\n"
        "reader calls scheduled_window(), scheduled_tail_bin(), in_recovery_traversal(), scheduled_routine(),\n"
        "recovered_iteration() or integrator(), whose first call seals the integrator.",
        noun="traversal use",
        where="outside traversal.cxx",
        clean="no active_levels outside traversal.cxx",
        home=layers_of(TRAVERSAL_FILES),
    ),
    Rule(
        "index",
        files=index_files,
        word=INDEX_PARAM,
        allowed=index_allowed,
        text=textwrap.fill(
            f"index: a CarpetX header or a source in {dirs(INDEX_SOURCE_LAYERS, 'or')} names a group, variable, "
            "time level, patch or level by its type (gi_t, vi_t, tl_t, patch_t, level_t, window_t), never a bare int, "
            "except in the signatures the flesh and AMReX fix, the level_t and window_t constructors, and "
            "window_t::levels and narrow.",
            width=RULE_WIDTH,
        ),
        noun="bare index int",
        where=f"in a CarpetX header or {dirs(INDEX_SOURCE_LAYERS, 'or')} source",
        clean=f"no bare index int in a CarpetX header or {dirs(INDEX_SOURCE_LAYERS, 'or')} source",
        home=INDEX_SOURCE_LAYERS,
    ),
    Rule(
        "records",
        files=records_files,
        word=RECORDS_WORD,
        allowed=never,
        text=textwrap.fill(
            f"records: the flesh's group table (CCTK_{', '.join(RECORDS_CALLS)}) is read by hierarchy/, which "
            "builds the records from it, and by flesh/, where its ints arrive, never by "
            f"{dirs(RECORDS_LAYERS, 'or')}; a file there asks driver().record(gi), record_of(var) or records().",
            width=RULE_WIDTH,
        ),
        noun="group-table call",
        where=f"in {dirs(RECORDS_LAYERS, 'or')}",
        clean=f"no group-table call in {dirs(RECORDS_LAYERS, 'or')}",
        home=RECORDS_LAYERS,
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
        home=layers_of(CODEC_FILES),
    ),
    Rule(
        "layer",
        files=private_cxx,
        word=LAYER_INCLUDE,
        allowed=layer_allowed,
        text=textwrap.fill(
            "layer: a private CarpetX file includes internal.hxx and the headers of its own layer or a lower one, "
            f'each as "<layer>/<name>.hxx"; the layers, lowest first, are {", ".join(LAYERS)}.',
            width=RULE_WIDTH,
        ),
        noun="include",
        where="outside the layer order",
        clean="every internal include at or below its own layer",
    ),
    Rule(
        "door",
        files=private_cxx,
        word=GHEXT_OR_DRIVER_CALL,
        allowed=door_allowed,
        text="door: the hierarchy pointer ghext is named only in CarpetX/src/hierarchy/hierarchy.cxx; every other\n"
        f"file calls driver(), and only from {HIERARCHY_LAYERS[0]}/ or a layer above it, never from "
        f"{dirs(UNDER_HIERARCHY, 'or')};\n"
        f"driver() is const, and mutable_driver() is called only in {WRITE_LAYER}/: another layer changes the\n"
        f"hierarchy through the operations of {WRITE_LAYER}/operations.hxx.",
        noun="hierarchy use",
        where="the door does not allow",
        clean=f"the hierarchy has one door that {dirs(UNDER_HIERARCHY, 'and')} never use",
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



def series(items: list[str]) -> str:
    return ", ".join(items[:-1]) + ", and " + items[-1]


def summary(found: list[tuple[str, Finding]]) -> str:
    if not found:
        return "✓ boundary: " + series([rule.clean for rule in RULES])
    counts = Counter(f.rule for _, f in found)
    return "✗ boundary: " + ", ".join(rule.count(counts[rule.name]) for rule in RULES if counts[rule.name])


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
