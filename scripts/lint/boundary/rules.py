from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Iterator, Optional

from checks import excerpt, plural, wrap
from comments.lex_cxx import lex
from comments.tokens import Kind
from files import Lang, lang_of

MARKER = "CARPETX_INTERNAL"
PRIVATE_DIR = "CarpetX/src/"
PUBLIC_DIR = "CarpetX/src/include/"
MAKE_DEFN = "CarpetX/src/make.code.defn"
READS = {
    "util": (),
    "traversal": ("util",),
    "config": ("util",),
    "hierarchy": ("util", "traversal", "config"),
    "comm": ("util", "config", "hierarchy"),
    "regrid": ("hierarchy", "comm"),
    "ops": ("util", "hierarchy"),
    "io": ("util", "config", "hierarchy", "comm", "ops"),
    "schedule": ("traversal", "config", "hierarchy", "comm", "io"),
    "api": ("util", "config", "hierarchy", "comm", "schedule"),
    "flesh": ("util", "config", "hierarchy", "comm", "regrid", "ops", "io", "schedule", "api"),
}
LAYERS = tuple(READS)
assert all(targets == tuple(target for target in LAYERS[: LAYERS.index(layer)] if target in targets) for layer, targets in READS.items()), READS
READS_TEXT = "; ".join(f"{layer}/ reads {', '.join(f'{target}/' for target in READS[layer]) or 'nothing'}" for layer in LAYERS)
READS_HOME = "scripts/lint/boundary/rules.py"
OWNS = {
    "util": "leaves that know nothing of the grid. Nothing here names the hierarchy, a level, a window or the scheduler.",
    "traversal": "the run's phase, one slot per fact, each written only by a scope that `schedule/` opens. Its readers are `schedule/` and the public `active_window()`; a lower layer is told what it needs of the phase in a call. A fact about the whole run or about the grid is not a slot here.",
    "config": "what the run is configured with, and the index and stage types to say it in. Nothing here calls `driver()`.",
    "hierarchy": "the data model and what it cannot be built or written without. Nothing here only reads the model: a query belongs to the layer that owns its subject.",
    "comm": "filling grid points whose values other grid points or the boundary conditions determine: ghosts from another rank or a coarser level, coarse cells from finer ones, and the outer boundary; with the operators it fills with and what it keeps on a level to do so. A stem either decides what is filled and records its validity, or applies and records nothing. It also owns three questions a fill must answer before it runs: which groups the integrator fills, where a level stands in its parent's step, and which groups have a flux register. It never decides when: `schedule/` and `api/` ask. A body names no `cGH` but the global one and changes nothing of the clock.",
    "regrid": "the bodies of the AMReX callbacks, never data-model code: what a level holds and how it is built belongs to `hierarchy/`.",
    "ops": "operations on grid data that other thorns ask for. Their flesh entry points are in `flesh/`.",
    "io": "output, checkpoint, recovery and the file reader's input, and the format of the data model's metadata. When to do each is decided by the flesh's bins and by `schedule/`. It keeps one fact between two of the flesh's calls: the iteration of the checkpoint the parameters were recovered from, which the grid structure is then recovered from.",
    "schedule": "the driver's loop and clocks, and `CallFunction`, which every routine runs through, with its checksums. It opens the scopes of the run's phase, and every sync of `comm/` runs through its `SyncRequested`, which refuses one the open bin does not allow; `SyncScheduled` is the flesh's sync, on the levels of the open traversal. How a level is stored, synced or written is decided below it. Its header declares only what `flesh/` registers or wraps, what `api/` hands to the flesh, and `SyncScheduled` and `SyncRequested`, which `flesh/` and `api/` call. A writer is an argument of each function that uses one, and none is kept between calls.",
    "api": "the bodies of the public headers. An operation names itself with `operation_scope_t`, checks its arguments, and calls the layer that owns the work. It keeps the writers it uses in one slot, and of the clock it changes only the stage time and step, for the length of one `run_at`.",
    "flesh": 'every `extern "C"`, and `hierarchy_writers_t`, the writers that own the hierarchy, which it lends to `schedule/` and `api/`. It owns no other data: an entry converts the flesh\'s arguments and calls the layer that does the work.',
}
STEMS = {
    "config": {
        "groups": "what a group was declared as: its kind and its tags",
        "boundaries": "the symmetries and default boundaries the parameters ask for",
        "grid_sizes": "the largest grid each level may have",
        "indices": "ranges over the public index types",
        "stages": "RK stages and the refinement of time",
    },
    "hierarchy": {
        "model": "`GHExt`, its nested types, and the door `driver()`",
        "writers": "the writers of the hierarchy; `create_hierarchy` makes them once and returns them in `hierarchy_writers_t`, which owns the hierarchy",
        "clock": "the only place the model's run clock and the stage it shows are opened, and the only writes to the cGH clock fields: `cctkGH_clock` publishes every change to the flesh's cGH and to every copy of it; and the check that a cGH a scheduled function returns still shows it",
        "valid": "the validity state a time level stores",
        "scopes": "the only way to change that state: write, require and invalidate",
        "binding": "the only writes to the cGH pointer table",
        "modes": "which cGH a routine sees, and entering and leaving a mode",
        "loops": "visiting levels and tiles",
        "grid": "describing one tile to a kernel",
        "handles": "the bodies of the public handles, all in one place",
    },
    "comm": {
        "sync": "the sync a routine or thorn asks for: ghosts from the same level and the coarser one, then the outer boundary",
        "restrict": "coarse cells from the finer level's",
        "ghost_prolongation": "which levels, groups and time levels are prolongated into, the validity that requires of the level it reads, and the validity it records",
        "outer_boundary": "which groups and time levels get the outer boundary, and the validity that records; the thorn routine it calls, and the refusal of a sync started from inside that call",
        "rk_ghosts": "filling coarse-fine ghosts between parent steps from the RK bands",
        "reflux": "flux registers: which groups have one, what accumulates in it, and correcting the coarse level from it",
        "fillpatch": "carrying data between two `MultiFab`s with an operator",
        "boundary_conditions": "applying a group's boundary conditions to one `MultiFab`",
        "prolongate_3d_rf2": "the prolongation operators, and the choice of one per group",
        "buffers": "what `comm/` keeps on a level, and its lifetime",
        "rk_bands": "what a level holds of its parent's RK step, and when it may be read",
        "checkpoint_bands": "the bands of a level a checkpoint carries, each under its tag",
        "timelevels": "which time levels a request covers",
        "group_selection": "which of the groups asked for an exchange acts on",
        "level_runs": "splitting a window into runs of levels a predicate treats alike",
        "level_pairs": "visiting each level of a window that has a child in it",
        "integrator": "what the registered integrator integrates, and in how many stages",
        "substeps": "where a level stands in its parent's step",
    },
}
assert set(STEMS) <= set(LAYERS), STEMS
STEMS_HOME = READS_HOME
GRANTS = {
    "written_by": ("Writer",),
    "GHExt": ("hierarchy_writers_t",),
    "GHExt::run_clock_t": ("cctkGH_clock",),
    "GHExt::shown_stage_t": ("cctkGH_clock",),
    "GHExt::registry_t": ("registry_writer_t",),
    "GHExt::level_set_t": ("level_writer_t",),
    "GHExt::patches_t": ("structure_writer_t",),
    "GHExt::PatchData::core_t": ("structure_writer_t",),
    "GHExt::PatchData::levels_t": ("level_writer_t",),
    "GHExt::PatchData::LevelData::clock_t": ("clock_writer_t",),
    "GHExt::PatchData::LevelData::GroupData::timelevel_t": ("write_scope_t",),
    "GHExt::PatchData::LevelData::GroupData::scratch_pool_t": ("scratch_writer_t",),
    "GHExt::PatchData::LevelData::GroupData::ring_t": ("cctkGH_binding",),
    "GHExt::GlobalData::ArrayGroupData::ring_t": ("cctkGH_binding",),
    "GHExt::GlobalData::ArrayGroupData::timelevel_t": ("array_write_scope_t",),
    "clock_writer_t": ("hierarchy_writers_t",),
    "stage_writer_t": ("hierarchy_writers_t",),
    "level_writer_t": ("structure_writer_t",),
    "structure_writer_t": ("hierarchy_writers_t",),
    "registry_writer_t": ("hierarchy_writers_t",),
    "scratch_writer_t": ("hierarchy_writers_t",),
    "hierarchy_writers_t": ("create_hierarchy",),
    "cctkGH_binding": ("clock_writer_t",),
    "cctkGH_clock": ("clock_writer_t", "stage_writer_t"),
    "why_valid_t": ("write_scope_t", "array_write_scope_t", "operator<<"),
    "write_scope_t": ("open_write", "write_scopes_t"),
    "array_write_scope_t": ("open_array_write",),
    "GridPtrDesc1": ("operator<<",),
}
GRANTS_HOME = READS_HOME
GRANT_LAYER = "hierarchy"
GRANT_WRAPPER = "written_by"
assert GRANT_WRAPPER in GRANTS, GRANTS
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


FAMILY = re.compile(r"(.+?)_impl(?:_.+)?")


def row_of(stem: str) -> str:
    family = FAMILY.fullmatch(stem)
    return family.group(1) if family else stem


def includable(layer: str) -> tuple[str, ...]:
    assert layer in LAYERS, layer
    return READS[layer] + (layer,)


def carried(layer: str, header_reads: dict[str, tuple[str, ...]]) -> set[str]:
    found = {layer}
    for target in header_reads.get(layer, ()):
        assert target in READS[layer], (layer, target)
        found.update(carried(target, header_reads))
    return found


def linkable(layer: str, header_reads: dict[str, tuple[str, ...]]) -> tuple[str, ...]:
    assert layer in LAYERS, layer
    found = {layer}
    for target in READS[layer]:
        found.update(carried(target, header_reads))
    return tuple(target for target in LAYERS if target in found)


def layers_of(names: tuple[str, ...]) -> tuple[str, ...]:
    found = tuple(dict.fromkeys(layer_of(name) for name in names))
    assert None not in found, names
    return found


def dirs(layers: tuple[str, ...], conjunction: str) -> str:
    names = [f"{layer}/" for layer in layers]
    return f" {conjunction} ".join(filter(None, (", ".join(names[:-1]), names[-1])))


CHECKED = (Lang.CXX, Lang.MAKE)
MARKER_WORD = re.compile(r"CARPETX_INTERNAL\b")
GUARD = re.compile(r"^\s*#\s*ifndef\s+CARPETX_INTERNAL\s*$")
INCLUDE_INTO_SRC = re.compile(r'^\s*#\s*include\s*(?:"[^"]*CarpetX/src[^"]*"|<[^>]*CarpetX/src[^>]*>)')
VALIDITY_FILES = (layer_path("hierarchy", "scopes.cxx"), layer_path("hierarchy", "scopes.hxx"), layer_path("hierarchy", "valid.hxx"))
VALIDITY_WORD = re.compile(
    r"(\b(error_if_invalid|warn_if_invalid|poison_invalid_g[fa]|check_valid_g[fa])"
    r"|(\.|->)(set_all|set_int|set_outer|set_ghosts|set_invalid|set_valid))\s*\("
)
BINDING_FILES = (layer_path("hierarchy", "binding.cxx"),)
BINDING_WORD = re.compile(r"(\.|->)\s*data\s*\[[^\]]*\]\s*\[[^\]]*\]\s*=(?!=)")
CLOCK_FILES = (layer_path("hierarchy", "clock.cxx"),)
CLOCK_FIELD = r"(\.|->)\s*cctk_(iteration|time|delta_time|timefac)\b"
CLOCK_WORD = re.compile(rf"{CLOCK_FIELD}\s*([-+*/]?=(?!=)|\+\+|--)|(\+\+|--)\s*[\w.()\->]*{CLOCK_FIELD}")
GHOST_ONLY_FILES = (
    layer_path("comm", "sync.hxx"),
    layer_path("comm", "sync.cxx"),
    layer_path("api", "api.cxx"),
    layer_path("flesh", "compile_checks.cxx"),
)
GHOST_ONLY_WORD = re.compile(r"\bSyncGhostsOnly\s*\(")
REQUESTED_SYNC_FILES = (layer_path("comm", "sync.hxx"), layer_path("comm", "sync.cxx"))
REQUESTED_SYNC_CALLER = layer_path("schedule", "clocks.cxx")
REQUESTED_SYNC_WORD = re.compile(r"\bSyncGroups\s*\(")
APPLIES_LAYER = "comm"
APPLIES_STEMS = ("fillpatch", "boundary_conditions", "prolongate_3d_rf2")
assert set(APPLIES_STEMS) <= set(STEMS[APPLIES_LAYER]), APPLIES_STEMS
APPLIES_WORD = re.compile(r'^\s*#\s*include\s*[<"](?:hierarchy/scopes|' + APPLIES_LAYER + r'/(\w+))\.hxx[>"]')
TRAVERSAL_FILES = (layer_path("traversal", "traversal.cxx"), layer_path("traversal", "traversal.hxx"))
TRAVERSAL_WORD = re.compile(
    r"\b(active_levels|tail_bin|last_routine)\b"
)
PHASE_LAYER = "hierarchy"
PHASE_READER = layer_path(PHASE_LAYER, "handles.cxx")
PHASE_INCLUDE = re.compile(r'^\s*#\s*include\s*[<"]traversal/traversal\.hxx[>"]')
DOOR_FILE = layer_path("hierarchy", "model.cxx")
DRIVER_CALL = re.compile(r"\bdriver\s*\(\s*\)")
LAYER_INCLUDE = re.compile(r'^\s*#\s*include\s*(?:"([^"]*)"|<((?:' + "|".join(LAYERS) + r')/[^>]*)>)')
VARINDEX_WORD = re.compile(r"\b(firstvarindex|CCTK_FirstVarIndexI\s*\((?:[^()]|\([^()]*\))*\))\s*\+")
RECORD_VAR_FILE = layer_path("hierarchy", "model.hxx")
RECORD_VAR = re.compile(r"\bvar_t\s*\{\s*firstvarindex\s*\+")
MODEL_FILE = layer_path("hierarchy", "model.hxx")
MODEL_STATE_FILE = layer_path("hierarchy", "valid.hxx")
MODEL_FILES = (MODEL_FILE, MODEL_STATE_FILE)
MODEL_WORD = re.compile(
    r'\bmutable\b|\bextern\b|^\s*#\s*include\s*[<"]carpetx_rk\.hxx[>"]'
    r"|\bamrex::(Interpolater|BCRec|FluxRegister)\b"
    r'|^\s*#\s*include\s*[<"]yaml-cpp/|^\s*#\s*include\s*[<"]zlib\.h[>"]'
    r'|^\s*#\s*include\s*[<"]hierarchy/(?!valid\.hxx[>"])'
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
    layer_path("config", "groups.cxx"): ("CCTK_TraverseString",),
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
    r'\bghext->|\bdriver\s*\(\s*\)|\bGHExt\b|\bLevelData\b|\bGroupData\b|^\s*#\s*include\s*["<]hierarchy/model\.hxx[">]'
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
FRIEND = re.compile(r"\bfriend\b")
CLASS_KEYWORDS = ("struct", "class", "union")
ACCESS_KEYWORDS = ("public", "protected", "private")
IDENTIFIER = re.compile(r"[A-Za-z_]\w*")
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
    return lang is Lang.CXX and name in MODEL_FILES


def records_files(name: str, lang: Lang) -> bool:
    return carpetx_cxx(name, lang) and layer_of(name) in RECORDS_LAYERS


def codec_cxx(name: str, lang: Lang) -> bool:
    return lang is Lang.CXX and name in CODEC_FILES


def hierarchy_header(name: str, lang: Lang) -> bool:
    return carpetx_header(name, lang) and layer_of(name) == GRANT_LAYER


def in_files(names: tuple[str, ...]) -> Callable[..., bool]:
    return lambda name, *_: name in names


def in_rows(layer: str, stems: tuple[str, ...]) -> Files:
    assert set(stems) <= set(STEMS[layer]), (layer, stems)
    return lambda name, lang: carpetx_cxx(name, lang) and layer_of(name) == layer and row_of(name.rsplit("/", 1)[1].split(".", 1)[0]) in stems


def never(name: str, line: str, statement: str) -> bool:
    return False


def applies_allowed(name: str, line: str, statement: str) -> bool:
    stem = APPLIES_WORD.search(line).group(1)
    return stem is not None and row_of(stem) in APPLIES_STEMS


def marker_allowed(name: str, line: str, statement: str) -> bool:
    if name == MAKE_DEFN:
        return True
    return is_private(name) and lang_of(name) is Lang.CXX and bool(GUARD.match(line))


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
    return layer_of(name) in HIERARCHY_LAYERS


def requested_sync_allowed(name: str, line: str, statement: str) -> bool:
    return name in REQUESTED_SYNC_FILES or name == REQUESTED_SYNC_CALLER


def unqualified(head: list[tuple[str, int]]) -> list[tuple[str, int]]:
    while head:
        if head[0][0] in ACCESS_KEYWORDS and head[1:2] and head[1][0] == ":":
            head = head[2:]
        elif head[0][0] == "template":
            at = 1
            depth = 0
            while at < len(head) and head[1][0] == "<":
                depth += head[at][0] == "<"
                depth -= head[at][0] == ">"
                at += 1
                if depth <= 0:
                    break
            head = head[at:]
        else:
            break
    return head


def friend_name(words: list[str]) -> str:
    if "(" in words:
        call = words.index("(")
        if "operator" in words[:call]:
            at = words.index("operator") + 1
            if words[at : at + 2] == ["(", ")"]:
                return "operator()"
            return "operator" + "".join(words[at : words.index("(", at)])
        return words[call - 1]
    depth = 0
    name = ""
    for word in words:
        depth += word == "<"
        depth -= word == ">"
        if depth == 0 and IDENTIFIER.fullmatch(word):
            name = word
    return name


def class_named(head: list[tuple[str, int]]) -> Optional[str]:
    words = [word for word, _ in head]
    if words[:1] and words[0] in CLASS_KEYWORDS and words[1:2] and IDENTIFIER.fullmatch(words[1]) and "(" not in words:
        return words[1]
    return None


def aliased(words: list[str]) -> Optional[tuple[str, str]]:
    name, wrapped, writer = words[1:2], words[5:6], words[7:8]
    if words[:1] + words[2:5] + words[6:7] + words[8:] != ["using", "=", GRANT_WRAPPER, "<", ",", ">"]:
        return None
    if not all(IDENTIFIER.fullmatch(word) for word in name + wrapped + writer):
        return None
    return wrapped[0], writer[0]


def friends(text: str) -> Iterator[tuple[int, str, str]]:
    scopes: list[Optional[str]] = []
    head: list[tuple[str, int]] = []
    for token in lex(text):
        if token.kind is not Kind.CODE:
            continue
        word = token.text
        if word in (";", "{"):
            declaration = unqualified(head)
            alias = aliased([w for w, _ in declaration]) if word == ";" else None
            if declaration[:1] and declaration[0][0] == "friend":
                yield declaration[0][1], "::".join(scope for scope in scopes if scope), friend_name([w for w, _ in declaration[1:]])
            elif alias:
                yield declaration[0][1], "::".join([scope for scope in scopes if scope] + [alias[0]]), alias[1]
            if word == "{":
                scopes.append(class_named(declaration))
            head = []
        elif word == "}":
            if scopes:
                scopes.pop()
            head = []
        else:
            head.append((word, token.line))


def misspelled(text: str) -> list[int]:
    words = [(token.text, token.line) for token in lex(text) if token.kind is Kind.CODE]
    named = [at for at, (word, _) in enumerate(words[:-1]) if word == GRANT_WRAPPER and words[at + 1][0] == "<"]
    spelled = [[word for word, _ in words[max(at - 3, 0) : at + 7]] for at in named]
    return [words[at][1] for at, around in zip(named, spelled) if around[9:] != [";"] or not aliased(around[:9])]


def ungranted(name: str, text: str) -> list[int]:
    unlisted = [line for line, owner, friend in friends(text) if friend not in GRANTS.get(owner, ())]
    return sorted(unlisted + [line for line in misspelled(text) if line not in unlisted])


def granted(name: str, line: str, statement: str) -> bool:
    return True


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
        text="a validity kernel or a why_valid_t setter is named only in CarpetX/src/hierarchy/scopes.cxx, scopes.hxx and "
        "valid.hxx; every other site opens a write scope or calls require or invalidate.",
        noun="validity call",
        where="outside scopes.cxx",
        clean="no validity call outside scopes.cxx",
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
        "clock",
        files=private_cxx,
        word=CLOCK_WORD,
        allowed=in_files(CLOCK_FILES),
        text="a cGH clock field (cctk_iteration, cctk_time, cctk_delta_time, cctk_timefac) is assigned only in "
        "CarpetX/src/hierarchy/clock.cxx, which publishes the model's run clock to every cGH; every other site asks "
        "clock_writer_t, or the stage_writer_t lent to api/, and reads the fields.",
        noun="clock-field write",
        where="outside clock.cxx",
        clean="no clock-field write outside clock.cxx",
        home=layers_of(CLOCK_FILES),
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
        text=f"{MODEL_FILE}, the header that declares the data model, has no mutable member, declares no extern variable, does not include "
        "<carpetx_rk.hxx> and names no amrex::Interpolater, amrex::BCRec or amrex::FluxRegister, so that a const hierarchy writes nothing but comm/'s buffers and what its cGH pointers reach, of which the rules binding and clock leave the pointer table and the clock fields to one file each, and every fact "
        f"about the hierarchy, the epoch included, is a GHExt member changed only by a holder the table GRANTS in {GRANTS_HOME} names; what comm/ "
        "builds on a level lives in comm/'s level_buffers_t, which LevelData holds through one pointer to a type it "
        "only forward-declares, and the integrator record is a GHExt value. It includes no yaml-cpp header, no <zlib.h> and no header of "
        f"its own layer but {MODEL_STATE_FILE}, the validity state it stores, so that a file that includes the data model receives no other concern with it: "
        "a file that opens a write scope includes hierarchy/scopes.hxx, and the format of the metadata is in io/. "
        f"{MODEL_STATE_FILE} is held to the same words.",
        noun="mutable, extern, comm/ object or other concern's include",
        where="in the data model's headers",
        clean="no mutable, extern, comm/ object or other concern's include in the data model's headers",
        home=layers_of((MODEL_FILE,)),
    ),
    Rule(
        "grant",
        files=hierarchy_header,
        word=FRIEND,
        allowed=granted,
        text=f"a friend line in a header of {GRANT_LAYER}/ is an entry of the table GRANTS in {GRANTS_HOME}, under the type it "
        "sits in, spelled with the types that type is nested in, and naming the class or function it befriends, so that "
        f"granting a holder the private members of a type is an edit of that table; an alias using name = {GRANT_WRAPPER}<T, W>; is the "
        f"friend line of W under T, spelled the same way, because only W opens it for writing, and {GRANT_WRAPPER} is given arguments in no other form "
        "in such a header, so that no spelling hides a writer; the tests refuse an entry no friend line backs.",
        noun="friend line",
        where="GRANTS does not list",
        clean=f"every friend line in {GRANT_LAYER}/ listed in GRANTS",
        home=(GRANT_LAYER,),
        whole=ungranted,
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
        text="a traversal slot (active_levels, tail_bin, last_routine) is named only in "
        "CarpetX/src/traversal/traversal.cxx and traversal.hxx; a driver phase opens a scope, and a reader calls "
        "scheduled_window(), scheduled_tail_bin() or scheduled_routine().",
        noun="traversal use",
        where="outside traversal.cxx",
        clean="no active_levels outside traversal.cxx",
        home=layers_of(TRAVERSAL_FILES),
    ),
    Rule(
        "phase",
        files=in_rows(PHASE_LAYER, tuple(STEMS[PHASE_LAYER])),
        word=PHASE_INCLUDE,
        allowed=in_files((PHASE_READER,)),
        text=f"in {PHASE_LAYER}/ only handles.cxx, which defines the public active_window(), includes "
        f"traversal/traversal.hxx; any other file of {PHASE_LAYER}/ is told what it needs of the phase in a call.",
        noun="phase include",
        where=f"in {PHASE_LAYER}/ outside handles.cxx",
        clean=f"no phase include in {PHASE_LAYER}/ outside handles.cxx",
        home=(PHASE_LAYER,),
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
        text=f"the flesh's group table (CCTK_{', '.join(RECORDS_CALLS)}) is read by config/ and hierarchy/, which build the "
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
        "own library: no ghext->, driver(), GHExt, LevelData, GroupData or hierarchy/model.hxx.",
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
        "namespace qualified, or declares a name it uses throughout, as config/indices.hxx declares using Loop::dim;, and a "
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
        word=DRIVER_CALL,
        allowed=door_allowed,
        text=f"driver() is called only from {HIERARCHY_LAYERS[0]}/ or a layer above it, never from "
        f"{dirs(UNDER_HIERARCHY, 'or')}. The pointer it reads is defined in the unnamed namespace of {DOOR_FILE}, "
        "so the compiler refuses its name in every other file.",
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
    Rule(
        "requested-sync",
        files=private_cxx,
        word=REQUESTED_SYNC_WORD,
        allowed=requested_sync_allowed,
        text=f"SyncGroups, the sync of comm/, is called outside comm/sync.cxx only in {REQUESTED_SYNC_CALLER}, by SyncRequested, "
        "which refuses a sync in an end-of-step bin that does not hold every level and then calls SyncGroups; the "
        "flesh's sync entry calls SyncScheduled, and the public CarpetX::sync and every other sync of the driver call "
        "SyncRequested.",
        noun="SyncGroups call",
        where="outside schedule/clocks.cxx",
        clean="no SyncGroups call outside schedule/clocks.cxx",
        home=layers_of(REQUESTED_SYNC_FILES + (REQUESTED_SYNC_CALLER,)),
    ),
    Rule(
        "applies",
        files=in_rows(APPLIES_LAYER, APPLIES_STEMS),
        word=APPLIES_WORD,
        allowed=applies_allowed,
        text=f"a stem of {APPLIES_LAYER}/ that applies ({', '.join(APPLIES_STEMS)}) writes into the MultiFab it is handed and "
        f"records nothing: it includes neither hierarchy/scopes.hxx nor a header of {APPLIES_LAYER}/ outside those stems, so it "
        "can open no write scope, require no validity, and call no stem that does; the stem that decides what is filled "
        "does both.",
        noun="write-scope or deciding-stem include",
        where=f"in a stem of {APPLIES_LAYER}/ that applies",
        clean=f"no write-scope or deciding-stem include in a stem of {APPLIES_LAYER}/ that applies",
        home=(APPLIES_LAYER,),
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
