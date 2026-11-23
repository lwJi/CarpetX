from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass

from .doc import PUBLIC_HEADERS, defined, prototypes, read
from .rules import RULES
from files import PROJECT, is_thorn

GUIDE = "agent_docs/carpetx-public-api.md"
RK_GUIDE = "agent_docs/carpetx-rk-api.md"
THORN = "TestCarpetXAPI"
SOURCES = f"{THORN}/src"
TESTS = f"{THORN}/test"
ERRORS = f"{THORN}/test/errors"
PARAM = f"{THORN}/param.ccl"
DRIVER_PARAM = "CarpetX/param.ccl"
MARKER_HEADER = "CarpetX/src/internal.hxx"
DRIVER_SOURCES = "CarpetX/src"
MODE_ROW = re.compile(r'^\s*"([\w-]+)"\s*::\s*"([^"]*)"\s*$', re.M)
MODE_DEFAULT = re.compile(r'^\}\s*"([\w-]+)"\s*$', re.M)
PARFILE_MODE = re.compile(rf'^\s*{THORN}::mode\s*=\s*"([\w-]+)"', re.M)
PARAMETER = re.compile(r"^\s*(?:BOOLEAN|INT|REAL|KEYWORD|STRING|CCTK_INT|CCTK_REAL)\s+(\w+)", re.M)
WORD = re.compile(r"\w+")
SPAN = re.compile(r"`([^`]*)`")
QUALIFIED = re.compile(r"\b(\w+)::(\w+)")
FOREIGN_QUALIFIED = re.compile(r"\b(?:CCTK_\w*|cctk\w*|std|amrex|Loop|Array4\w*)(?:::\w+)+")
CALLED = re.compile(r"\b(\w+)\s*[({]")
TYPED = re.compile(r"\b(\w+_t)\b")
BARE = re.compile(r"^\w+$")
DECLARED = re.compile(r"\b(?:struct|class|enum class|using)\s+(?:\[\[nodiscard\]\]\s+)?(\w+)\b(?!\s*;)")
METHOD = re.compile(r"\b([A-Za-z_]\w*)\s*\((?:[^;{}()]|\([^;{}()]*\))*\)\s*(?:const\s*)?(?:&&?\s*)?(?:noexcept\s*)?(?:=\s*(?:default|delete)\s*)?[;{]")
KEYWORDS = {"if", "switch", "for", "while", "return", "sizeof", "static_assert", "decltype", "alignas", "catch", "operator"}
FOREIGN = ("CCTK_", "cctk", "std", "amrex", "Loop", "Array4")


@dataclass(frozen=True)
class Row:
    text: str
    held_by: tuple[str, ...]


@dataclass(frozen=True)
class Section:
    title: str
    rows: tuple[Row, ...]
    exercised_by: tuple[str, ...] = ()
    refusals: tuple[str, ...] = ()


TITLE = "# CarpetX's public API"
WRITTEN = "`python3 scripts/lint doc --write` writes this file from `scripts/lint/boundary/guide.py`. A row says only what a header cannot: a signature is read in the header."
BOUNDARY: tuple[str, ...] = (
    WRITTEN,
    f"A thorn reaches CarpetX through `carpetx.hxx`, and the time integrator also through `carpetx_rk.hxx`, whose contract is in `{RK_GUIDE}`. "
    f"`{THORN}` is a client of both: its `interface.ccl` and `configuration.ccl` are what a client declares, `{SOURCES}/` is a worked use of the calls below, "
    f"and each parfile of `{ERRORS}/` is a call CarpetX refuses, with the text it aborts with in the `.expect` file beside it. "
    "Every call is available between the driver's SetupGH and its Shutdown, and a bad argument or a stale handle is a `CCTK_VERROR`.",
)
RK_TITLE = "# CarpetX's integrator API"
RK_BOUNDARY: tuple[str, ...] = (
    WRITTEN,
    f"`carpetx_rk.hxx` is for the one thorn that is the time integrator. `{GUIDE}` covers `carpetx.hxx` and the validity regions named here, and `{THORN}` is a client of both.",
)
UNEXERCISED = ("restrict_no_poison", "scratch")
GENERAL: tuple[Section, ...] = (
    Section(
        "Indices and handles",
        (
            Row(
                "`vi_t` is a variable's place in its group and `var_t` the flesh's variable index. "
                "`window_t` is a half-open range of levels on a half-open range of patches: `levels` spans every patch, `narrow` keeps the patches of its window, and `coarsest` and `finest` are on its first patch. "
                "`component_t` is a box of a level, AMReX's global box index and the flesh's `cctk_component`; `tile_t` is one tile of the tiled traversal over this rank's boxes and carries the `component_t` it lies in. "
                "`loop_parallel` hands its kernel the `level_t`, the `tile_t` and the local cGH.",
                ("hierarchy/handles.cxx:variable", "hierarchy/handles.cxx:levels", "hierarchy/handles.cxx:narrow", "hierarchy/handles.cxx:coarsest", "hierarchy/handles.cxx:finest", "hierarchy/handles.cxx:loop_parallel", "hierarchy/model.hxx:level", "hierarchy/model.hxx:get_local_cctkGH", "config/indices.hxx:tile_of"),
            ),
        ),
        exercised_by=("compile_checks.cxx", "TestCarpetXAPI_Fill"),
        refusals=("level_missing", "timelevel_missing", "parent_of_level_0", "window_empty", "window_out_of_bounds", "window_outside_traversal", "hierarchy_absent"),
    ),
    Section(
        "Data, validity and the write scope",
        (
            Row(
                "A write scope sets the validity of the group at `tl` to exactly `leaves_valid` when it opens and poisons the regions it does not promise, so the scope must not read them; when it closes, the promised regions are scanned for poison and NaN. "
                "While it is open, a communication operation, `invalidate` or a scheduled function touching the group is refused. "
                "`require_valid` also runs that scan over `where`. A validity bit is set only by a write scope or a communication operation.",
                ("api/api.cxx:write", "api/api.cxx:require_valid", "api/api.cxx:invalidate", "api/api.cxx:require_no_open_write", "hierarchy/scopes.cxx:write_scope_t", "schedule/callfunction.cxx:CallFunction"),
            ),
            Row(
                "The halo has two ghost regions with a bit each: the same-level ghosts, which another box of the level or a periodic image covers, and the coarse-fine ghosts, which no box of the level covers and only an interpolation from the parent fills. "
                "A ghost fill marks only the region it copies; the boundary condition reads the interior and both ghost regions, so it runs after the last ghost fill, requires all three valid, and alone marks the outer boundary. "
                "`make_valid_ghosts()` and a schedule clause's everywhere name both.",
                ("include/carpetx.hxx:make_valid_ghosts", "schedule/clauses.cxx:decode_clauses", "hierarchy/model.cxx:LevelData", "hierarchy/scopes.cxx:loop_ghost_halves", "hierarchy/scopes.cxx:check_valid_gf", "comm/outer_boundary.cxx:fill_outer_boundary"),
            ),
            Row(
                "`scratch(group)` is shaped like `data(group, tl_t{0})` and has no validity bits; a regrid frees it, after which a `scratch_t` is an error.",
                ("api/api.cxx:scratch", "api/api.cxx:resolve_scratch", "api/api.cxx:scratch_t"),
            ),
        ),
        exercised_by=("TestCarpetXAPI_Fill",),
        refusals=("write_twice", "write_unfilled", "require_invalid", "invalidate_during_write", "schedule_during_write"),
    ),
    Section(
        "Communication",
        (
            Row(
                "An operation without a `tl` acts on every time level but the oldest. "
                "`sync` is the scheduled SYNC, subcycling under `use_subcycling`, and sets the same-level, coarse-fine and outer bits it fills; `sync_ghosts_only` is its same-level copy alone and sets the same-level bit, and `prolongate_ghosts` its coarse-to-fine interpolation alone and sets the coarse-fine bit, neither applying the boundary condition; `prolongate_aligned` is `prolongate_ghosts` that under subcycling fills only levels time-aligned with their parent. "
                "`apply_outer_boundary` is the boundary condition alone: it requires the interior and both ghost regions, writes the outer region and sets the outer bit, so a thorn that has written a group everywhere and wants the driver's boundary condition on it calls this and no ghost fill. "
                "`restrict_no_poison` leaves the coarse ghost and outer regions as they were, where the scheduled restriction invalidates and poisons them. `synchronize` waits for the device.",
                ("api/api.cxx:sync", "api/api.cxx:sync_ghosts_only", "api/api.cxx:prolongate_ghosts", "api/api.cxx:prolongate_aligned", "api/api.cxx:apply_outer_boundary", "comm/sync.cxx:SyncGhostsOnly", "comm/ghost_prolongation.cxx:ProlongateGhosts_impl", "comm/outer_boundary.cxx:ApplyOuterBoundary", "api/api.cxx:restrict_no_poison", "comm/restrict.cxx:Restrict", "comm/restrict.cxx:RestrictNoPoison", "hierarchy/loops.cxx:synchronize"),
            ),
        ),
        exercised_by=("TestCarpetXAPI_Fill",),
        refusals=("sync_during_write", "prolongate_unwritten_parent", "sync_invalid_coarse", "boundary_before_sync"),
    ),
    Section(
        "The level hierarchy and the parameters",
        (
            Row(
                "The BoxArray and DistributionMapping of a level are those of any group's MultiFab on it, and the refinement ratio is 2. "
                "Driver parameters are read through the flesh, `SHARES: Driver` or `CCTK_ParameterGet`.",
                ("hierarchy/model.cxx:PatchData", "hierarchy/model.cxx:GroupData", "config/grid_sizes.cxx:per_level_max_grid_sizes", "io/selection.cxx:select_groups"),
            ),
        ),
        exercised_by=("TestCarpetXAPI_Fill",),
        refusals=("too_many_levels", "output_unknown_variable"),
    ),
    Section(
        "The clock a routine sees",
        (
            Row(
                "`cctk_iteration`, `cctk_time`, `cctk_delta_time` and `cctk_timefac` are the run clock that CarpetX holds, and every cGH shows it. "
                "A thorn reads them and never assigns them: when a scheduled function returns, a cGH that disagrees with the run clock stops the run.",
                ("hierarchy/clock.cxx:publish_clock", "hierarchy/clock.cxx:require_current_clock", "schedule/callfunction.cxx:CallFunction"),
            ),
        ),
        exercised_by=("TestCarpetXAPI_Fill",),
        refusals=("clock_stale",),
    ),
)
RK_SECTIONS: tuple[Section, ...] = (
    Section(
        "The integrator's contract",
        (
            Row(
                "Exactly one thorn, the time integrator, publishes what it integrates and reports the events of its RK step. "
                "Registration is `AT WRAGH`, once. "
                "A registration is built by `with_dense_output`, which names the method whose dense output fills the refinement boundary under subcycling and takes that method's weights, "
                "or by `with_weights`, which writes `b`, the final-update weight of every stage, and has `dense_output` `none`. "
                "A dense output is never written beside a `b`; the method is named, not inferred from the stage count, and `none` cannot run under subcycling. "
                "Nonempty weights sum to 1; a method that never updates the state registers no integrator.",
                ("api/api_rk.cxx:register_integrator", "api/api_rk.cxx:require_integrator", "api/api_rk.cxx:check_weights", "api/api_rk.cxx:check_pair", "hierarchy/model.cxx:record_integrator"),
            ),
            Row(
                "`rk_dense_output_method` states the method each dense output was written for as a whole Butcher tableau: its stage count, its order, `a`, its weights `b`, and its stage times `c`, in the order the stages arrive through `rhs_computed`. "
                "The dense output's weights reduce to that `b` at the end of the parent step, which CarpetX asserts at build, and the kernel takes the stage values of the finer level from that `a` and `c`. "
                "An integrator that registers a dense output is that method: CarpetX takes the registered weights from this table, so they are not written twice.",
                ("comm/rk_ghosts.cxx:rk_dense_output_weights", "comm/rk_ghosts.cxx:rk_dense_output_weights_end_at_method_b", "comm/rk_ghosts.cxx:rk_fine_stage", "api/api_rk.cxx:register_integrator"),
            ),
            Row(
                "`run_schedule_group_at` sets `cctk_time` and `cctk_delta_time` on every cGH for the traversal and then restores both; pass `cctkGH->cctk_delta_time` unless the group is to see a step of another size.",
                ("api/api_rk.cxx:run_schedule_group_at",),
            ),
            Row(
                "`open_rk_step` is called once the state holds u(t_n), outside any OpenMP parallel region. "
                "`rhs_computed` feeds the flux registers with the stage's weight times that level's `CCTK_DELTA_TIME`. "
                "`state_updated` says the state holds the next stage value: under subcycling it fills the coarse-fine ghosts of every refined level in the window and marks them valid; "
                "the same-level ghosts and the outer boundary of the integrated groups stay invalid until the thorn's SYNC in ODESolvers_PostStep. Without subcycling the last `state_updated` applies the flux registers.",
                ("api/api_rk.cxx:open_rk_step", "api/api_rk.cxx:require_serial_context", "api/api_rk.cxx:rhs_computed", "api/api_rk.cxx:state_updated", "comm/rk_ghosts.cxx:FillRKBoundary", "api/api_rk.cxx:rk_step_t"),
            ),
            Row(
                "Under subcycling, on a refined level at iteration > 0 the SYNC of an integrated group keeps the coarse-fine ghosts that `state_updated` or the driver's refill after a recovery wrote; elsewhere it fills every ghost itself. "
                "After a recovery the driver refills the integrated groups' refinement boundary before `CCTK_POST_RECOVER_VARIABLES`; the integrator schedules nothing for it.",
                ("comm/sync.cxx:SyncGroupsSubcycling", "comm/rk_ghosts.cxx:RecoverRKBoundary"),
            ),
        ),
        exercised_by=("TestCarpetXAPI_Register", "TestCarpetXAPI_Fill", "TestCarpetXAPI_Step"),
        refusals=(
            "register_at_startup",
            "register_at_paramcheck",
            "register_after_use",
            "register_twice",
            "weights_sum_mismatch",
            "reflux_without_weights",
            "subcycling_without_dense_output",
            "integrated_not_checkpointed",
            "unregistered_rk_call",
            "rk_before_freeze",
            "rk_open_invalid_state",
            "rhs_invalid",
            "rhs_repeated",
            "state_before_rhs",
            "rhs_after_last_stage",
            "step_closed_early",
            "ghosts_before_sync",
            "outer_before_sync",
            "rk_fill_before_parent_step",
            "rk_fill_stale_parent_step",
        ),
    ),
)
SECTIONS = GENERAL + RK_SECTIONS


def words(text: str) -> set[str]:
    return set(WORD.findall(text))


def foreign(name: str) -> bool:
    return name.startswith(FOREIGN)


def names_in(text: str) -> set[str]:
    found: set[str] = set()
    for span in SPAN.findall(text):
        if BARE.match(span):
            found.add(span)
            continue
        rest = FOREIGN_QUALIFIED.sub(" ", span)
        for scope, member in QUALIFIED.findall(rest):
            found.update((scope, member))
        found.update(CALLED.findall(rest))
        found.update(TYPED.findall(rest))
    return {name for name in found if not foreign(name)}


def public_names(root: str) -> set[str]:
    return set.union(*(words(read(os.path.join(root, header))) for header in PUBLIC_HEADERS))


def declared_names(root: str) -> set[str]:
    found = {name for names in prototypes(root).values() for name in names}
    for header in PUBLIC_HEADERS:
        text = read(os.path.join(root, header))
        found.update(name for name in DECLARED.findall(text) if name != "namespace")
        found.update(name for name in METHOD.findall(text) if name not in KEYWORDS and not foreign(name))
    return found


def bare_names(root: str) -> set[str]:
    known = {rule.name for rule in RULES}
    known.update(PARAMETER.findall(read(os.path.join(root, DRIVER_PARAM))))
    known.update(words(read(os.path.join(root, MARKER_HEADER))))
    return known


def test_sources(root: str) -> dict[str, str]:
    folder = os.path.join(root, SOURCES)
    return {name: read(os.path.join(folder, name)) for name in sorted(os.listdir(folder)) if name.endswith(".cxx")}


def hold_names(root: str, title: str, text: str, public: set[str], bare: set[str]) -> None:
    unknown = sorted(name for name in names_in(text) if name not in public and not (BARE.match(name) and name in bare) and not is_thorn(root, name))
    if unknown:
        raise ValueError(f"{', '.join(unknown)} named in the rows of {title!r} in scripts/lint/boundary/guide.py but in no public header")


def hold_exercised(root: str, section: Section, sources: dict[str, str]) -> None:
    for anchor in section.exercised_by:
        if anchor.endswith(".cxx"):
            if anchor not in sources:
                raise ValueError(f"{anchor} is not a file of {SOURCES}/, where a row of {section.title!r} in scripts/lint/boundary/guide.py says it is")
        elif not any(defined(text, anchor) for text in sources.values()):
            raise ValueError(f"{anchor} is defined in no file of {SOURCES}/, where a row of {section.title!r} in scripts/lint/boundary/guide.py says it is")


def hold_bound(root: str, section: Section, row: Row) -> None:
    if not row.held_by:
        raise ValueError(f"a row of {section.title!r} in scripts/lint/boundary/guide.py is held by no function of {DRIVER_SOURCES}/")
    for anchor in row.held_by:
        file, _, name = anchor.partition(":")
        path = os.path.join(root, DRIVER_SOURCES, file)
        if not name or not os.path.isfile(path):
            raise ValueError(f"{anchor} in a row of {section.title!r} in scripts/lint/boundary/guide.py is not a file of {DRIVER_SOURCES}/ and a function")
        if not defined(read(path), name):
            raise ValueError(f"{name} is not defined in {DRIVER_SOURCES}/{file}, where a row of {section.title!r} in scripts/lint/boundary/guide.py says it is")


def hold_unexercised(root: str, declared: set[str], sources: dict[str, str]) -> None:
    used = set.union(*(words(text) for text in sources.values()))
    unknown = [name for name in UNEXERCISED if name not in declared]
    if unknown:
        raise ValueError(f"{', '.join(unknown)} in UNEXERCISED in scripts/lint/boundary/guide.py but declared in no public header")
    stale = [name for name in UNEXERCISED if name in used]
    if stale:
        raise ValueError(f"{', '.join(stale)} in UNEXERCISED in scripts/lint/boundary/guide.py but used in {SOURCES}/: remove it from the list")
    mentioned = set.union(*(names_in(row.text) for section in SECTIONS for row in section.rows))
    missing = sorted(name for name in mentioned & declared if name not in used and name not in UNEXERCISED)
    if missing:
        raise ValueError(f"{', '.join(missing)} named in the rows of scripts/lint/boundary/guide.py but used nowhere in {SOURCES}/: add a test or list it in UNEXERCISED")
    silent = [name for name in UNEXERCISED if name not in mentioned]
    if silent:
        raise ValueError(f"{', '.join(silent)} in UNEXERCISED in scripts/lint/boundary/guide.py but named in none of its rows: remove it from the list")
    if list(UNEXERCISED) != sorted(UNEXERCISED):
        raise ValueError("UNEXERCISED in scripts/lint/boundary/guide.py is not sorted")


def modes(root: str) -> tuple[str, dict[str, str]]:
    text = read(os.path.join(root, PARAM))
    default = MODE_DEFAULT.search(text)
    if default is None:
        raise ValueError(f"{PARAM} names no default mode")
    return default.group(1), dict(MODE_ROW.findall(text))


def refusals(root: str) -> dict[str, tuple[str, str]]:
    default, described = modes(root)
    found: dict[str, tuple[str, str]] = {}
    seen: dict[str, str] = {}
    for par in sorted(glob.glob(os.path.join(root, ERRORS, "*.par"))):
        name = os.path.basename(par)[: -len(".par")]
        expect = par[: -len(".par")] + ".expect"
        if not os.path.isfile(expect):
            raise ValueError(f"{ERRORS}/{name}.par has no {name}.expect beside it")
        m = PARFILE_MODE.search(read(par))
        mode = m.group(1) if m else default
        if mode not in described:
            raise ValueError(f"{ERRORS}/{name}.par sets {THORN}::mode = {mode!r}, which {PARAM} does not offer")
        if mode != default:
            if mode in seen:
                raise ValueError(f"{ERRORS}/{name}.par and {seen[mode]}.par both set {THORN}::mode = {mode!r}")
            seen[mode] = name
        found[name] = (described[mode] if mode != default else "", read(expect).strip())
    tested: set[str] = set()
    for par in sorted(glob.glob(os.path.join(root, TESTS, "*.par"))):
        name = os.path.basename(par)[: -len(".par")]
        m = PARFILE_MODE.search(read(par))
        mode = m.group(1) if m else default
        if mode not in described:
            raise ValueError(f"{TESTS}/{name}.par sets {THORN}::mode = {mode!r}, which {PARAM} does not offer")
        if mode in seen:
            raise ValueError(f"{TESTS}/{name}.par and {ERRORS}/{seen[mode]}.par both set {THORN}::mode = {mode!r}: a mode is a test or a refusal")
        tested.add(mode)
    unused = sorted(mode for mode in described if mode != default and mode not in seen and mode not in tested)
    if unused:
        raise ValueError(f"{PARAM} offers {THORN}::mode {', '.join(unused)}, which no parfile of {TESTS}/ or {ERRORS}/ sets")
    return found


def hold_refusals(found: dict[str, tuple[str, str]]) -> None:
    claimed: dict[str, str] = {}
    for section in SECTIONS:
        for name in section.refusals:
            if name in claimed:
                raise ValueError(f"{name} is a refusal of both {claimed[name]!r} and {section.title!r} in scripts/lint/boundary/guide.py")
            if name not in found:
                raise ValueError(f"{name} is a refusal of {section.title!r} in scripts/lint/boundary/guide.py but {ERRORS}/{name}.par does not exist")
            claimed[name] = section.title
    unclaimed = sorted(name for name in found if name not in claimed)
    if unclaimed:
        raise ValueError(f"{', '.join(unclaimed)} in {ERRORS}/ but a refusal of no section in scripts/lint/boundary/guide.py")


def hold_all(root: str) -> None:
    public = public_names(root)
    bare = bare_names(root)
    sources = test_sources(root)
    for paragraph in BOUNDARY + RK_BOUNDARY:
        hold_names(root, "the preamble", paragraph, public, bare)
    for section in SECTIONS:
        for row in section.rows:
            hold_names(root, section.title, row.text, public, bare)
            hold_bound(root, section, row)
        hold_exercised(root, section, sources)
    hold_unexercised(root, declared_names(root), sources)
    hold_refusals(refusals(root))


def page(title: str, paragraphs: tuple[str, ...], sections: tuple[Section, ...]) -> str:
    parts = [title, ""]
    for paragraph in paragraphs:
        parts.extend([paragraph, ""])
    for section in sections:
        parts.extend([f"## {section.title}", ""])
        for row in section.rows:
            parts.extend([row.text, ""])
    return "\n".join(parts)


def render(root: str = PROJECT) -> str:
    hold_all(root)
    return page(TITLE, BOUNDARY, GENERAL)


def render_rk(root: str = PROJECT) -> str:
    hold_all(root)
    return page(RK_TITLE, RK_BOUNDARY, RK_SECTIONS)
