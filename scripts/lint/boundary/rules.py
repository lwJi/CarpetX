from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Iterator, Optional

from checks import excerpt, plural, wrap
from files import Lang, lang_of

MARKER = "CARPETX_INTERNAL"
PRIVATE_DIR = "CarpetX/src/"
PUBLIC_DIR = "CarpetX/src/include/"
MAKE_DEFN = "CarpetX/src/make.code.defn"
LAYERS = ("util", "traversal", "hierarchy", "comm", "regrid", "ops", "io", "schedule", "api", "flesh")
READS = {
    "util": (),
    "traversal": ("util",),
    "hierarchy": ("util", "traversal"),
    "comm": ("util", "traversal", "hierarchy"),
    "regrid": ("hierarchy", "comm"),
    "ops": ("util", "traversal", "hierarchy"),
    "io": ("util", "traversal", "hierarchy", "comm", "ops"),
    "schedule": ("traversal", "hierarchy", "comm", "io"),
    "api": ("util", "hierarchy", "comm", "schedule"),
    "flesh": ("util", "traversal", "hierarchy", "comm", "regrid", "ops", "io", "schedule", "api"),
}
assert tuple(READS) == LAYERS, tuple(READS)
assert all(targets == tuple(target for target in LAYERS[: LAYERS.index(layer)] if target in targets) for layer, targets in READS.items()), READS
READS_TEXT = "; ".join(f"{layer}/ reads {', '.join(f'{target}/' for target in READS[layer]) or 'nothing'}" for layer in LAYERS)
READS_HOME = "scripts/lint/boundary/rules.py"
OWNS = {
    "util": "leaves that know nothing of the grid. Nothing here names the hierarchy, a level, a window or the scheduler.",
    "traversal": "the run's phase, one slot per fact with one writer. A fact about the phase that `comm/`, `ops/` or `io/` must read without depending on `schedule/` is a slot here; a fact about the grid is not.",
    "hierarchy": "the data model and the hierarchy's views of itself. What a checkpoint carries is a field of `LevelData` or `GroupData`, or an RK band, which `comm/` owns and hands out through `checkpoint_bands` and `restore_bands`; what `comm/` builds on a level is in the slot `buffers`; a change to the hierarchy that another layer needs is a member of one of five writers, by what it writes: `clock_writer_t`, `structure_writer_t`, `level_writer_t`, `registry_writer_t` or `scratch_writer_t`. A writer has no public constructor, copy or move, and writes through the `GHExt &` it holds, so no member can be static. `create_hierarchy` makes them once and returns four in `hierarchy_writers_t`, which owns the hierarchy: destroying it is the only way to destroy the hierarchy, and `hierarchy/` keeps none, only the pointer `driver()` reads; the level writer is a private member of the structure writer, which lends it to the AMReX core of each patch it adds and builds a recovered level through it.",
    "comm": "moving data between ranks, between levels, and into ghost and boundary zones. It never decides when: `schedule/` and `api/` ask. A body names no `cGH` but the global one, and takes the iteration from `current_iteration()`.",
    "regrid": "the bodies of the AMReX callbacks, never data-model code: what a level holds and how it is built belongs to `hierarchy/`.",
    "ops": "operations on grid data that other thorns ask for. Their flesh entry points are in `flesh/`.",
    "io": "output, checkpoint, recovery and the file reader's input. When to do each is decided by the flesh's bins and by `schedule/`.",
    "schedule": "the driver's loop and clocks, and `CallFunction`, which every routine runs through. How a level is stored, synced or written is decided below it. Its header declares only what `flesh/` registers, what `flesh/` wraps to lend a writer, which is `Initialise`, `Evolve` and the four storage functions that record storage, and what `api/` hands to the flesh. A writer is an argument of each function that uses one, and none is kept between calls.",
    "api": "the bodies of the public headers. An operation names itself with `operation_scope_t`, checks its arguments, and calls the layer that owns the work. A thorn calls in with no flesh entry in its path, so `api/` holds the two writers it uses, `registry_writer_t` and `scratch_writer_t`, in its one slot: `adopt_writers` fills it and refuses while it is full, `return_writers` empties it at shutdown, and `registry_writer` and `scratch_writer` refuse while it is empty.",
    "flesh": 'every `extern "C"`, and the four writers of the hierarchy that `create_hierarchy` returns, `hierarchy_writers_t`, which it keeps in one slot and lends: as arguments to the entries of `schedule/` it wraps, and once to `api/` in `SetupGH`. The set owns the hierarchy, so `CarpetX_Shutdown` destroys the hierarchy by emptying the slot, after `api/` has returned its two. It never holds `level_writer_t`: the structure writer lends that to the AMReX core in `regrid/` when `InitGH` adds a patch. It owns no other data: an entry converts the flesh\'s arguments and calls the layer that does the work.',
}
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


def includable(layer: str) -> tuple[str, ...]:
    assert layer in LAYERS, layer
    return READS[layer] + (layer,)


def linkable(layer: str) -> tuple[str, ...]:
    assert layer in LAYERS, layer
    found = {layer}
    for target in READS[layer]:
        found.update(linkable(target))
    return tuple(target for target in LAYERS if target in found)


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
BINDING_FILES = (layer_path("hierarchy", "binding.cxx"),)
BINDING_WORD = re.compile(r"(\.|->)\s*data\s*\[[^\]]*\]\s*\[[^\]]*\]\s*=(?!=)")
GHOST_ONLY_FILES = (
    layer_path("comm", "comm.hxx"),
    layer_path("comm", "sync_restrict.cxx"),
    layer_path("api", "api.cxx"),
    layer_path("flesh", "compile_checks.cxx"),
)
GHOST_ONLY_WORD = re.compile(r"\bSyncGhostsOnly\s*\(")
TRAVERSAL_FILES = (layer_path("traversal", "traversal.cxx"), layer_path("traversal", "traversal.hxx"))
TRAVERSAL_WORD = re.compile(
    r"\b(active_levels|tail_bin|recovery_traversal|last_routine|recover_iteration)\b"
)
DOOR_FILE = layer_path("hierarchy", "hierarchy.cxx")
DOOR_WORD = re.compile(r"\bghext\b")
DRIVER_CALL = re.compile(r"\bdriver\s*\(\s*\)")
GHEXT_OR_DRIVER_CALL = re.compile(f"{DOOR_WORD.pattern}|{DRIVER_CALL.pattern}")
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
MODEL_FILE = layer_path("hierarchy", "driver.hxx")
MODEL_WORD = re.compile(
    r'\bmutable\b|\bextern\b|^\s*#\s*include\s*[<"]carpetx_rk\.hxx[>"]'
    r"|\bamrex::(Interpolater|BCRec|FluxRegister)\b"
)
WINDOW_WORD = re.compile(r"\bactive_levels_t\b")
ALIAS_WORD = re.compile(r"\bmake_alias\b")
INDEX_PARAM = re.compile(r"\b(?:int|CCTK_INT)\s+(gi|vi|tl|patch|level|min_level|max_level|min_patch|max_patch|stage|component|tile)\b")
STORAGE_OVERLOADS = ("QueryGroupStorageB", "GroupStorageIncrease", "GroupStorageDecrease")
AMREX_CALLBACKS = ("ErrorEst", "MakeNewLevelFromScratch", "MakeNewLevelFromCoarse", "RemakeLevel", "ClearLevel")
INDEX_FIXED = {
    PUBLIC_DIR + "carpetx.hxx": ("level_t", "tile_t", "window_t", "levels", "narrow"),
    layer_path("schedule", "schedule.hxx"): STORAGE_OVERLOADS,
    layer_path("schedule", "storage.cxx"): STORAGE_OVERLOADS,
    layer_path("hierarchy", "handles.cxx"): ("levels", "narrow"),
    layer_path("flesh", "startup.cxx"): ("ArrayGroupSizeB", "GroupDynamicData"),
    layer_path("flesh", "aliased.cxx"): ("CarpetX_GetDomainSpecification", "CarpetX_GetBoundarySizesAndTypes"),
    layer_path("hierarchy", "group_tags.cxx"): ("CCTK_TraverseString",),
    layer_path("comm", "comm.hxx"): (FLESH_SYNC_OVERLOAD,),
    layer_path("comm", "sync_restrict.cxx"): (FLESH_SYNC_OVERLOAD,),
    layer_path("regrid", "regrid.hxx"): AMREX_CALLBACKS,
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
PROTOTYPE_END = re.compile(r"\)\s*(?:noexcept\s*)?;\s*$")
NOT_A_DECLARATION = re.compile(r"\s*(?:$|#|//|(?:public|protected|private)\s*:(?!:))")
DECLARATION_START = re.compile(r"[A-Za-z_]")
FILE_LOCAL_DECLARATION = re.compile(r"static_assert\b|(?:extern\s+)?template\s+(?![\s<])")
STATIC_WORD = re.compile(r"\bstatic\b")
EXTERN_C = re.compile(r'\bextern\s+"C"')
OPERATOR_NAME = re.compile(r"\boperator\s*[^\s\w(]*")
TEMPLATE_ARGS = re.compile(r"<[^<>]*>")
NOT_A_DECLARATOR = re.compile(r"[^\w\s:*&]")
DECLARED_WORD = re.compile(r"[A-Za-z_]\w*(?:\s*::\s*[A-Za-z_]\w*)*")
STATEMENT_KEYWORDS = ("return", "throw", "else", "do", "case", "new", "delete", "co_return", "co_yield", "co_await")
HEADER_SUFFIXES = (".h", ".hh", ".hpp", ".hxx")
USING_DIRECTIVE = re.compile(r"^\s*using\s+namespace\b")
GUARD_PREFIX = "CARPETX_CARPETX_"
GUARD_MACRO = re.compile(r"^\s*#\s*(?:ifndef|define)\s+(" + GUARD_PREFIX + r"\w+)")
INCLUDE_LINE = re.compile(r"^\s*#\s*include\b")
DIRECTIVE = re.compile(r"^\s*#")
GUARD_WORD = re.compile(f"{GUARD_MACRO.pattern}|{INCLUDE_LINE.pattern}")
ENTRY_LAYER = "flesh"
STATEMENT_END = re.compile(r"[;{}]")
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
    whole: Optional[Callable[[str, str], list[int]]] = None

    def phrase(self, noun: str) -> str:
        return f"{noun} {self.where}" if self.where else noun

    def count(self, n: int) -> str:
        return self.phrase(plural(n, self.noun))

    def paragraph(self) -> str:
        return wrap(f"{self.name}: {self.text}")


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


def carpetx_source(name: str, lang: Lang) -> bool:
    return carpetx_cxx(name, lang) and name.endswith(".cxx")


def carpetx_header(name: str, lang: Lang) -> bool:
    return carpetx_cxx(name, lang) and name.endswith(HEADER_SUFFIXES)


def model_files(name: str, lang: Lang) -> bool:
    return lang is Lang.CXX and name == MODEL_FILE


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
    return len(parts) > 1 and ".." not in parts and layer is not None and parts[0] in includable(layer)


def declaration_head(statement: str) -> str:
    lines = statement.split("\n")
    while lines and NOT_A_DECLARATION.match(lines[0]):
        lines.pop(0)
    return "\n".join(lines)


def declared_function(head: str, in_header: bool = False) -> Optional[str]:
    before, paren, _ = head.partition("(")
    if not paren:
        return None
    before = EXTERN_C.sub("", before)
    operator = OPERATOR_NAME.search(before)
    before = OPERATOR_NAME.sub("operator", before)
    while in_header and TEMPLATE_ARGS.search(before):
        before = TEMPLATE_ARGS.sub("", before)
    words = DECLARED_WORD.findall(before)
    if NOT_A_DECLARATOR.search(before) or len(words) < 2 or words[0] in STATEMENT_KEYWORDS:
        return None
    return re.sub(r"\s+", "", operator.group(0)) if operator else words[-1]


def names_a_function(head: str) -> bool:
    return declared_function(head) is not None


def file_local(head: str) -> bool:
    return bool(FILE_LOCAL_DECLARATION.match(head)) or bool(STATIC_WORD.search(head.partition("(")[0]))


def prototype_allowed(name: str, line: str, statement: str) -> bool:
    head = declaration_head(statement)
    return not (DECLARATION_START.match(head) and names_a_function(head) and not file_local(head))


def guard_of(name: str) -> str:
    assert name.startswith(PRIVATE_DIR), name
    return "CARPETX_" + re.sub(r"[/.]", "_", "CarpetX/" + name[len(PRIVATE_DIR) :]).upper()


def guard_opening(name: str, text: str) -> list[int]:
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        m = GUARD_MACRO.match(line)
        if not DIRECTIVE.match(line) or INCLUDE_LINE.match(line) or (m is not None and m.group(1) != guard_of(name)):
            return []
        return [] if re.fullmatch(rf"\s*#\s*ifndef\s+{guard_of(name)}\s*", line) else [number]
    return []


def guard_allowed(name: str, line: str, statement: str) -> bool:
    m = GUARD_MACRO.match(line)
    if m:
        return m.group(1) == guard_of(name)
    return not is_private(name) or ROOT_HEADER in statement or guard_of(name) not in statement


def entry_allowed(name: str, line: str, statement: str) -> bool:
    return layer_of(name) == ENTRY_LAYER


def door_allowed(name: str, line: str, statement: str) -> bool:
    if DOOR_WORD.search(line) and name != DOOR_FILE:
        return False
    return not DRIVER_CALL.search(line) or layer_of(name) in HIERARCHY_LAYERS


RULES = (
    Rule(
        "include",
        files=public_cxx,
        word=INCLUDE_INTO_SRC,
        allowed=never,
        text="an #include whose path reaches into CarpetX/src; include <carpetx.hxx> instead.",
        noun="private include",
        where="",
        clean="no private CarpetX includes",
    ),
    Rule(
        "marker",
        files=checked,
        word=MARKER_WORD,
        allowed=marker_allowed,
        text="CARPETX_INTERNAL belongs only in CarpetX/src/make.code.defn and in the #ifndef CARPETX_INTERNAL guard of "
        "a private CarpetX header.",
        noun="stray marker",
        where="",
        clean="no stray CARPETX_INTERNAL",
    ),
    Rule(
        "validity",
        files=private_cxx,
        word=VALIDITY_WORD,
        allowed=in_files(VALIDITY_FILES),
        text="a validity kernel or a why_valid_t setter is named only in CarpetX/src/hierarchy/valid.cxx and "
        "valid.hxx; every other site opens a write scope or calls require or invalidate.",
        noun="validity call",
        where="outside valid.cxx",
        clean="no validity call outside valid.cxx",
        home=layers_of(VALIDITY_FILES),
    ),
    Rule(
        "binding",
        files=private_cxx,
        word=BINDING_WORD,
        allowed=in_files(BINDING_FILES),
        text="a cGH data[var][tl] entry is assigned only in CarpetX/src/hierarchy/binding.cxx; every other site calls "
        "cctkGH_binding.",
        noun="pointer-table write",
        where="outside binding.cxx",
        clean="no pointer-table write outside binding.cxx",
        home=layers_of(BINDING_FILES),
    ),
    Rule(
        "directions",
        files=in_files(SYNC_FILES),
        word=DIRECTIONS_WORD,
        allowed=directions_allowed,
        text="the sync layer takes a window and a list of gi_t; only the flesh-facing SyncGroupsOnActiveLevels "
        "receives the flesh's directions array, and discards it.",
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
        text="a flesh variable index is made from a group-relative one only by GroupRecord::var; call record.var(vi) "
        "instead of adding to firstvarindex or CCTK_FirstVarIndexI(gi).",
        noun="firstvarindex addition",
        where="outside GroupRecord::var",
        clean="no firstvarindex addition outside GroupRecord::var",
        home=layers_of((RECORD_VAR_FILE,)),
    ),
    Rule(
        "model",
        files=model_files,
        word=MODEL_WORD,
        allowed=never,
        text=f"the data model, {MODEL_FILE}, has no mutable member, declares no extern variable, does not include "
        "<carpetx_rk.hxx> and names no amrex::Interpolater, amrex::BCRec or amrex::FluxRegister, so that a const hierarchy writes nothing but comm/'s buffers and what its cGH pointers reach, and every fact "
        "about the hierarchy, the epoch included, is a GHExt member changed through one of the five writers, four lent by flesh/ and the level writer by the structure writer; what comm/ "
        "builds on a level lives in comm/'s level_buffers_t, which LevelData holds through one pointer to a type it "
        "only forward-declares, and the integrator record is a GHExt value.",
        noun="mutable, extern, integrator include or comm/ object",
        where="in the data model",
        clean="no mutable, extern, integrator include or comm/ object in the data model",
        home=layers_of((MODEL_FILE,)),
    ),
    Rule(
        "window",
        files=carpetx_cxx,
        word=WINDOW_WORD,
        allowed=never,
        text='window_t is the one "which levels" type; active_levels_t is gone, and a window is built only by '
        "active_window(), all_levels(), window_t::one_level, window_t::levels and narrow.",
        noun="active_levels_t use",
        where="",
        clean="no active_levels_t",
    ),
    Rule(
        "alias",
        files=carpetx_cxx,
        word=ALIAS_WORD,
        allowed=never,
        text="amrex::make_alias is named nowhere in CarpetX: AMReX's alias constructor takes a const MultiFab and "
        "returns a mutable one over the same data, which would turn what driver() hands out into a destination; a "
        "write site takes its destination from the write scope it opened, and a view of some components passes "
        "scomp and ncomp to the AMReX call.",
        noun="make_alias use",
        where="in CarpetX",
        clean="no make_alias in CarpetX",
    ),
    Rule(
        "traversal",
        files=private_cxx,
        word=TRAVERSAL_WORD,
        allowed=in_files(TRAVERSAL_FILES),
        text="a traversal slot (active_levels, tail_bin, recovery_traversal, last_routine, recover_iteration) is named "
        "only in CarpetX/src/traversal/traversal.cxx and traversal.hxx; a driver phase opens a scope, parameter "
        "recovery calls record_recovered_iteration() once, and a reader calls scheduled_window(), "
        "scheduled_tail_bin(), in_recovery_traversal(), scheduled_routine() or recovered_iteration().",
        noun="traversal use",
        where="outside traversal.cxx",
        clean="no active_levels outside traversal.cxx",
        home=layers_of(TRAVERSAL_FILES),
    ),
    Rule(
        "index",
        files=carpetx_cxx,
        word=INDEX_PARAM,
        allowed=index_allowed,
        text="every CarpetX source and header names a group, variable, time level, patch, level, RK stage, box or "
        "tile by its type (gi_t, vi_t, tl_t, patch_t, level_t, window_t, stage_t, component_t, tile_t), never a bare "
        "int or CCTK_INT, except in the signatures the flesh and AMReX fix, the level_t, tile_t and window_t "
        "constructors, and window_t::levels and narrow; a loop over a patch's levels walks level_handles(), one over "
        "the patches patch_handles(), one over the RK stages rk_stages, one over the boxes of a MultiFab "
        "components_of(), the tile under an MFIter is tile_of(mfi), and .index() unwraps a handle where it meets "
        "AMReX or the flesh.",
        noun="bare index int",
        where="in CarpetX",
        clean="no bare index int in CarpetX",
    ),
    Rule(
        "records",
        files=records_files,
        word=RECORDS_WORD,
        allowed=never,
        text=f"the flesh's group table (CCTK_{', '.join(RECORDS_CALLS)}) is read by hierarchy/, which builds the "
        "records from it, and by flesh/, where its ints arrive, never by "
        f"{dirs(RECORDS_LAYERS, 'or')}; a file there asks driver().record(gi), record_of(var) or records().",
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
        text="a checkpoint codec (io_*_checkpoint.cxx and its io_*_names.hxx) sees only the checkpoint record and its "
        "own library: no ghext->, driver(), GHExt, LevelData, GroupData or hierarchy/driver.hxx.",
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
        text="a private CarpetX file includes internal.hxx and the headers of its own layer or of a layer its own "
        f'reads, each as "<layer>/<name>.hxx": {READS_TEXT}. A layer reads only layers before it in that list, and a '
        f"new read is an entry of READS in {READS_HOME}, which holds no entry that no file of the layer includes.",
        noun="include",
        where="its layer does not read",
        clean="every internal include from its own layer or one it reads",
    ),
    Rule(
        "prototype",
        files=carpetx_source,
        word=PROTOTYPE_END,
        allowed=prototype_allowed,
        text="a CarpetX .cxx declares no function it does not define, since a prototype reaches past the header that "
        "the layer rule judges; a declaration at column 0 that ends in ); and names a plain type and function is "
        "one, and the symbols subcommand judges the rest, templates included, over the build's objects. A function "
        "another file defines is declared in a header of that file's layer and included, one this file defines is "
        "defined above its first use, and a forward declaration only this file needs is static. A namespace-scope "
        "object is initialised with = or braces, never with (), which reads as a prototype.",
        noun="prototype",
        where="in a CarpetX .cxx",
        clean="no prototype in a CarpetX .cxx",
    ),
    Rule(
        "using",
        files=carpetx_header,
        word=USING_DIRECTIVE,
        allowed=never,
        text="a CarpetX header adds no using namespace, since every file that includes it, and every layer above its "
        "own, would then see that whole namespace; a header spells what it takes from Arith, Loop or another "
        "namespace qualified, or declares a name it uses throughout, as driver.hxx declares using Loop::dim;, and a "
        ".cxx may say using namespace for its own body.",
        noun="using-directive",
        where="in a CarpetX header",
        clean="no using-directive in a CarpetX header",
    ),
    Rule(
        "guard",
        files=carpetx_header,
        word=GUARD_WORD,
        allowed=guard_allowed,
        text=f"a CarpetX header opens with the guard that spells its path, {GUARD_PREFIX}<LAYER>_<NAME>_HXX, on its "
        "first line, so that a copied header cannot hide behind the guard of its original, and a private one "
        f"includes {ROOT_HEADER} before any other header.",
        noun="header opening",
        where="out of order",
        clean="every header opens with its own guard and internal.hxx",
        whole=guard_opening,
    ),
    Rule(
        "entry",
        files=private_cxx,
        word=EXTERN_C,
        allowed=entry_allowed,
        text=f'every extern "C" definition of CarpetX is in {ENTRY_LAYER}/, one file per way the flesh calls in: a '
        "PROVIDES FUNCTION in aliased.cxx, a scheduled function in scheduled.cxx, and an overload above its "
        "registration in CarpetX_Startup in startup.cxx; a compile-time check of another layer's signature is in "
        "compile_checks.cxx, which sits at the top so that it may include any header.",
        noun="extern C entry",
        where=f"outside {ENTRY_LAYER}/",
        clean=f"no extern C entry outside {ENTRY_LAYER}/",
        home=(ENTRY_LAYER,),
    ),
    Rule(
        "door",
        files=private_cxx,
        word=GHEXT_OR_DRIVER_CALL,
        allowed=door_allowed,
        text="the hierarchy pointer ghext is named only in CarpetX/src/hierarchy/hierarchy.cxx; every other file "
        f"calls driver(), and only from {HIERARCHY_LAYERS[0]}/ or a layer above it, never from "
        f"{dirs(UNDER_HIERARCHY, 'or')}.",
        noun="hierarchy use",
        where="the door does not allow",
        clean=f"the hierarchy has one door that {dirs(UNDER_HIERARCHY, 'and')} never use",
    ),
    Rule(
        "ghost-only",
        files=private_cxx,
        word=GHOST_ONLY_WORD,
        allowed=in_files(GHOST_ONLY_FILES),
        text="SyncGhostsOnly, the same-level ghost-only copy, is called only by the public sync_ghosts_only in CarpetX/src/api/api.cxx; "
        "a sync inside the driver opens a scope that says what it fills and uses the scheduled SYNC or the "
        "prolongation.",
        noun="driver-internal ghost-only sync",
        where="outside api.cxx",
        clean="no driver-internal ghost-only sync",
        home=layers_of(GHOST_ONLY_FILES),
    ),
)
RULE = "\n".join([wrap(PREAMBLE)] + [rule.paragraph() for rule in RULES])


def findings(name: str, text: str, lang: Optional[Lang]) -> list[Finding]:
    if lang not in CHECKED:
        return []
    rules = [rule for rule in RULES if rule.files(name, lang)]
    found = []
    whole = sorted((number, rule.name) for rule in rules if rule.whole for number in rule.whole(name, text))
    for number, line, statement in statements(text):
        found.extend(Finding(number, rule, line.strip()) for at, rule in whole if at == number)
        found.extend(
            Finding(number, rule.name, line.strip())
            for rule in rules
            if rule.word.search(line) and not rule.allowed(name, line, statement)
        )
    return found


def statements(text: str) -> Iterator[tuple[int, str, str]]:
    statement = ""
    for number, line in enumerate(text.splitlines(), 1):
        statement += line + "\n"
        yield number, line, statement
        if STATEMENT_END.search(line):
            statement = ""


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
