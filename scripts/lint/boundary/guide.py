from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass

from .doc import PUBLIC_HEADERS, defined, prototypes, read
from .rules import RULES
from files import PROJECT, is_thorn

GUIDE = "agent_docs/carpetx-public-api.md"
THORN = "TestCarpetXAPI"
SOURCES = f"{THORN}/src"
ERRORS = f"{THORN}/test/errors"
PARAM = f"{THORN}/param.ccl"
DRIVER_PARAM = "CarpetX/param.ccl"
MARKER_HEADER = "CarpetX/src/internal.hxx"
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
class Section:
    title: str
    text: tuple[str, ...]
    exercised_by: tuple[str, ...] = ()
    refusals: tuple[str, ...] = ()


TITLE = "# CarpetX's public API"
PREAMBLE = (
    f"`python3 scripts/lint doc --write` writes this file from the rows of `scripts/lint/boundary/guide.py`, which it holds to the public headers, to `{SOURCES}/` and to the parfiles of `{ERRORS}/` "
    f"and their modes in `{PARAM}`: a name the rows set in code must be in a header, a parfile must be named under one section, and a refusal's text is its expect file; "
    "`scripts/lint/boundary/test_rules.py` fails when the file differs from what it writes, and the edit hook refuses an edit to it."
)
BOUNDARY: tuple[str, ...] = (
    "A thorn reaches CarpetX through two headers that `CarpetX/interface.ccl` exports from `CarpetX/src/include/`: `carpetx.hxx` for everything a general thorn needs, and `carpetx_rk.hxx` for the time integrator's contract. "
    "Every other header under `CarpetX/src/` is private: it includes `internal.hxx`, which stops with `#error` unless `CARPETX_INTERNAL` is defined, and only CarpetX's own `src/make.code.defn` defines it. "
    "The public headers include nothing under `CarpetX/src/`, so a client cannot reach the driver's internals transitively either.",
    "`python3 scripts/lint check --only boundary` holds this with two of its rules: `include` reports an `#include` outside `CarpetX/src/` whose path reaches into it, and `marker` reports a `CARPETX_INTERNAL` outside CarpetX's `src/make.code.defn` and the private headers' guard. "
    "Its other rules hold the layers inside CarpetX; `agent_docs/carpetx-internals.md`, the map of `CarpetX/src/`, says where each is at home, `python3 scripts/lint check --help` prints what every rule holds, and `python3 scripts/lint --help` says how the hooks and the Lint workflow run the check. "
    "The data model behind the headers has no public form: the public counterpart of a private name is the operation CarpetX performs with it.",
    f"`{THORN}` is a client of both headers. `{SOURCES}/compile_checks.cxx` holds the static assertions that pin the types, `{SOURCES}/api_test.cxx` exercises the calls, and `{ERRORS}/` holds a parfile for every refusal named below, each listed under its section with what the run does and the text it aborts with.",
)
UNEXERCISED_INTRO = f"What this file says of a name {THORN} never uses is held by nothing but the headers. Those names are: "
UNEXERCISED = ("Interval", "Timer", "delta_iteration", "iteration", "loop_coarse_to_fine", "loop_parallel", "num_patches", "rat64", "restrict_no_poison", "scratch", "symmetries")
SECTIONS: tuple[Section, ...] = (
    Section(
        "Using the headers",
        (
            "In `interface.ccl`, `USES INCLUDE HEADER: carpetx.hxx` (and `carpetx_rk.hxx` for an integrator); in `configuration.ccl`, `REQUIRES CarpetX`, which also provides the AMReX, Loop and Arith headers that `carpetx.hxx` includes; in C++, `#include <carpetx.hxx>` and qualify every name with `CarpetX::`. "
            "The headers add no using-directive at any scope. `Loop::GridDescBase` is a Loop type and comes from `<loop.hxx>`, not from CarpetX.",
            "The header is the list of calls; this file states only what a signature cannot. Every call is available between the driver's SetupGH and its Shutdown, and a bad argument is a `CCTK_VERROR`, never garbage.",
        ),
    ),
    Section(
        "Indices and handles",
        (
            "An index is a type: `patch_t` names a patch, `gi_t` a Cactus group, `tl_t` a time level, `vi_t` a variable by its place in its group, and `var_t` a variable as the flesh numbers it. "
            "Each is built explicitly from the flesh's `int`, `patch_t{cctk_patch}`, `gi_t{CCTK_GroupIndex(\"…\")}`, `tl_t{0}` or `var_t{CCTK_VarIndex(\"…\")}`, and read back only through `index()`, where a value leaves for the flesh (`CCTK_FullGroupName(gi.index())`). "
            "An index does not convert to an `int` or from one implicitly, nor to another kind of index, and it has no arithmetic; two of a kind compare equal and order. "
            "The one named step is `tl_t::next()`, the following time level, which is refused where it is used if the group has no such level. "
            "`variable(var)` splits a flesh variable index into `variable_t{gi, vi}`, its group and its place there; `valid` and `reduce` take that `vi_t`.",
            "A handle is made of indices and resolved on every call, so a stale one names a level that no longer exists and the call reports it instead of touching freed memory. "
            "`level_t{patch_t{cctk_patch}, cctk_level}` names a refinement level of a patch; it has no public fields, is read back through `patch()` and `index()`, and has no arithmetic: `level.parent()` is the next coarser level and an error on level 0, and `level.child()` is the next finer one, empty on the finest level of its patch. "
            "`group_t{level, gi}` names a grid-function group on a level (`gi` is a `gi_t`). "
            "`window_t` is a half-open range of levels on a half-open range of patches, both for traversal (`loop_serially`, `loop_coarse_to_fine`, and `loop_parallel`, whose kernel receives the component's cGH) and as the \"which levels\" argument of every operation. "
            "It has no public fields and is built only by `active_window()`, the window of the current traversal and an error outside one, `all_levels()`, the whole hierarchy, `window_t::one_level(level)`, `window_t::levels(min_level, max_level)` over every patch, and `window.narrow(min_level, max_level)` over the same patches as `window`. "
            "Each checks the bounds against the hierarchy when it is called, so a bad window is an error that names the line that built it, not the operation that later uses it. "
            "The two ints of `levels` and `narrow` are the bounds of a range, not levels, and `window.level_bounds()` and `window.patch_bounds()` read them back as `{min, max}` pairs of ints. "
            "`window.coarsest()` and `window.finest()` are its lowest and highest level on its first patch, and an error on an empty window. `num_levels(patch)` takes a `patch_t` and is per patch.",
        ),
        exercised_by=("compile_checks.cxx", "TestCarpetXAPI_Fill"),
        refusals=("level_missing", "timelevel_missing", "parent_of_level_0", "window_empty", "window_out_of_bounds", "window_outside_traversal", "hierarchy_absent"),
    ),
    Section(
        "Data, validity and the write scope",
        (
            "`data(group, tl)` is the `const amrex::MultiFab &` of a time level; AMReX's const overloads give `Array4<const T>`. "
            "Writing goes through a scope: `auto w = write(group, tl, leaves_valid, \"what\"); w.mfab()`. "
            "When it opens, every variable of the group at `tl` gets exactly `leaves_valid` as its validity, and the regions it does not promise are poisoned, so the scope must not read them. "
            "When it is destroyed, the promised regions are scanned for poison and NaN. "
            "While it is open, a second `write` on the same group and time level, any communication operation or `invalidate` on the group, and scheduling a function that reads, writes or syncs the group are all errors.",
            "`valid(group, tl, vi)` reads the bits. `require_valid(group, tl, where, \"why\")` raises an error with CarpetX's explanation if any variable of the group lacks a bit of `where`, and runs the poison and NaN scan. "
            "`invalidate(window, groups, tl, \"why\")` clears every bit of every variable at `tl`. "
            "No thorn outside CarpetX sets a validity bit: the write scope and the communication operations are the only way a bit is set.",
            "`scratch(group)` is a move-only buffer with the boxes, distribution, components and ghosts of `data(group, tl_t{0})`, drawn from a per-group pool that CarpetX keeps between calls. "
            "There is no init or free call; the destructor returns the buffer. The pool dies with the level on a regrid, and using a `scratch_t` after that is an error, not a dangling pointer. It has no validity bits.",
        ),
        exercised_by=("TestCarpetXAPI_Fill",),
        refusals=("write_twice", "write_unfilled", "require_invalid", "invalidate_during_write", "schedule_during_write"),
    ),
    Section(
        "Communication",
        (
            "Each operation takes a window and a list of grid-function groups as a `std::span<const gi_t>`, so a `std::vector<gi_t>` or a `std::array<gi_t, N>` converts in place (a braced list does not; write `std::array{gi}` for one group), and is an error while a write scope is open on any of them.",
            "- `sync(window, groups)` is a scheduled SYNC: every time level but the oldest, the subcycling variant when `use_subcycling`, and it sets the ghost and outer bits it fills.",
            "- `sync_ghosts_only(window, groups)`: the same-level copy only, on every time level but the oldest; `sync_ghosts_only(window, groups, tl)` does that one time level.",
            "- `prolongate_ghosts(window, groups)`: coarse-to-fine interpolation only, on every time level but the oldest; marks the ghost and outer bits of the levels it filled. `prolongate_ghosts(window, groups, tl)` does that one time level.",
            "- `prolongate_aligned(window, groups)` and `prolongate_aligned(window, groups, tl)`: as `prolongate_ghosts`, but under subcycling only into levels time-aligned with their parent.",
            "- `restrict_no_poison(window, groups)`: fine-to-coarse over every level pair in the window.",
            "- `synchronize()` waits for the device.",
        ),
        exercised_by=("TestCarpetXAPI_Fill",),
        refusals=("sync_during_write", "prolongate_unwritten_parent", "sync_invalid_coarse"),
    ),
    Section(
        "Queries, reductions and parameters",
        (
            "`reduce(gi, vi, tl)` and `reduction<T, D>`, `Timer` and `Interval`, `rat64` (the type of `iteration(level)` and `delta_iteration(level)`), `geometry(level)`, `symmetries(level)`, `numvars`, `indextype` and `nghostzones` are what their names say. "
            "The BoxArray and DistributionMapping of a level are those of any group's MultiFab on it, and the refinement ratio is 2 between every pair of levels. "
            "Driver parameters are read through the flesh (`SHARES: Driver` in `param.ccl`, or `CCTK_ParameterGet`), not through the headers, and CarpetX checks its own.",
        ),
        exercised_by=("TestCarpetXAPI_Fill",),
        refusals=("too_many_levels", "output_unknown_variable"),
    ),
    Section(
        "The integrator's contract: `carpetx_rk.hxx`",
        (
            "Exactly one thorn, the time integrator, publishes what it integrates and reports the events of its RK step; CarpetX derives everything else.",
            "Registration is once, `AT WRAGH`: `register_integrator(rk_config_t{dense_output, b, integrated})`. "
            "The driver freezes the record when its WRAGH traversal ends, so a registration before SetupGH or after WRAGH is refused, and so is an RK operation before the freeze. "
            "`dense_output` names the method whose dense output fills the refinement boundary under subcycling (`ssprk3`, `rk4` or `none`); it is named, not inferred from the stage count, so a 3-stage method other than SSPRK3 registers `none` and cannot run under subcycling. "
            "`b` holds the final-update weight of every stage, so `b.size()` is the stage count; an implicit method registers an empty `b` and never feeds a flux register, and all-zero weights are allowed for a method that never updates the state. "
            "`integrated` holds `rk_pair_t{state, rhs}` pairs of `gi_t`. CarpetX checks the record against its own parameters and refuses a bad one. A driver without a registration still runs; only `open_rk_step` needs one.",
            "`run_schedule_group_at(cctkGH, \"G\", t, delta_time)` sets `cctk_time` and `cctk_delta_time` on the given cGH and on every driver cGH, traverses the group, and restores both. "
            "`delta_time` is the raw `cctk_delta_time` the group sees, so pass `cctkGH->cctk_delta_time` unless the step itself changes, as an IMEX half step does.",
            "The step is `auto step = open_rk_step(window)`, opened once the state holds u(t_n) and outside any OpenMP parallel region. It stores the old state of every registered pair with storage on each level of the window and requires their interior valid. "
            "Then `rhs_computed()` and `state_updated()` alternate, `b.size()` times each, and the destructor checks that every stage ran. "
            "`rhs_computed` requires the interior of every registered right-hand side valid on every level of the window, stores the stage band, and feeds the flux registers with the stage's weight times that level's `CCTK_DELTA_TIME`. "
            "`state_updated` says the state now holds the next stage value: under subcycling CarpetX copies the same-level ghosts of every refined level in the window, fills its refinement boundary from its own clocks, scans the ghosts, and marks the integrated groups' ghosts valid, and their outer boundary where every face has a symmetry or boundary; without subcycling the last `state_updated` applies the flux registers. "
            "A call out of order, a call after the last stage, and a step destroyed early are errors naming the window and the stage; no stage index is passed, so stages cannot run out of order.",
            "Two promises CarpetX makes the integrator in return, which no parfile below holds. Under subcycling the SYNC of an integrated group on a refined level at iteration > 0 copies only the same-level ghosts and requires the ghost bit set already, by `state_updated` or by the driver's refill after a recovery; finding it unset is an error, and on level 0 the SYNC fills and marks every ghost itself. "
            "After a recovery the driver refills the integrated groups' refinement boundary itself before it traverses `CCTK_POST_RECOVER_VARIABLES`; the integrator schedules nothing for it.",
        ),
        exercised_by=("TestCarpetXAPI_Register", "TestCarpetXAPI_Fill"),
        refusals=(
            "register_at_startup",
            "register_at_paramcheck",
            "register_after_use",
            "register_twice",
            "dense_output_stage_mismatch",
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
        ),
    ),
)


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


def hold_unexercised(root: str, declared: set[str], sources: dict[str, str]) -> None:
    used = set.union(*(words(text) for text in sources.values()))
    unknown = [name for name in UNEXERCISED if name not in declared]
    if unknown:
        raise ValueError(f"{', '.join(unknown)} in UNEXERCISED in scripts/lint/boundary/guide.py but declared in no public header")
    stale = [name for name in UNEXERCISED if name in used]
    if stale:
        raise ValueError(f"{', '.join(stale)} in UNEXERCISED in scripts/lint/boundary/guide.py but used in {SOURCES}/: remove it from the list")
    mentioned = set.union(*(names_in(paragraph) for section in SECTIONS for paragraph in section.text))
    missing = sorted(name for name in mentioned & declared if name not in used and name not in UNEXERCISED)
    if missing:
        raise ValueError(f"{', '.join(missing)} named in the rows of scripts/lint/boundary/guide.py but used nowhere in {SOURCES}/: add a test or list it in UNEXERCISED")
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
    unused = sorted(mode for mode in described if mode != default and mode not in seen)
    if unused:
        raise ValueError(f"{PARAM} offers {THORN}::mode {', '.join(unused)}, which no parfile of {ERRORS}/ sets")
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


def refusal_lines(section: Section, found: dict[str, tuple[str, str]]) -> list[str]:
    lines = []
    for name in section.refusals:
        what, says = found[name]
        lines.append(f"- `{name}`: {what + ', with' if what else 'CarpetX aborts the run with'} `{says}`")
    return lines


def closing(section: Section, found: dict[str, tuple[str, str]]) -> list[str]:
    if not section.exercised_by and not section.refusals:
        return []
    anchors = [f"`{SOURCES}/{anchor}`" if anchor.endswith(".cxx") else f"`{anchor}`" for anchor in section.exercised_by]
    parts = []
    if anchors:
        parts.append(f"{' and '.join(anchors)} {'exercises' if len(anchors) == 1 else 'exercise'} this section.")
    if section.refusals:
        parts.append(f"Its refusals, each a parfile of `{ERRORS}/`:")
    return [" ".join(parts), ""] + refusal_lines(section, found)


def render(root: str = PROJECT) -> str:
    public = public_names(root)
    bare = bare_names(root)
    sources = test_sources(root)
    for paragraph in BOUNDARY:
        hold_names(root, "the preamble", paragraph, public, bare)
    for section in SECTIONS:
        for paragraph in section.text:
            hold_names(root, section.title, paragraph, public, bare)
        hold_exercised(root, section, sources)
    hold_unexercised(root, declared_names(root), sources)
    found = refusals(root)
    hold_refusals(found)
    parts = [TITLE, "", PREAMBLE, ""]
    for paragraph in BOUNDARY:
        parts.extend([paragraph, ""])
    parts.extend([UNEXERCISED_INTRO + ", ".join(f"`{name}`" for name in UNEXERCISED) + ".", ""])
    for section in SECTIONS:
        parts.extend([f"## {section.title}", ""])
        bullets = False
        for paragraph in section.text:
            bullet = paragraph.startswith("- ")
            if bullets and not bullet:
                parts.append("")
            parts.append(paragraph)
            if not bullet:
                parts.append("")
            bullets = bullet
        if bullets:
            parts.append("")
        tail = closing(section, found)
        if tail:
            parts.extend(tail + [""])
    return "\n".join(parts)
