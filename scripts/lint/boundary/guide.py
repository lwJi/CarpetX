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
    f"`python3 scripts/lint doc --write` writes this file from `scripts/lint/boundary/guide.py`, which holds every row to the public headers, to `{SOURCES}/` and to `{ERRORS}/`; the edit hook refuses an edit here. "
    "A row says only what a header cannot: a signature is read in the header, and a refusal in its parfile."
)
BOUNDARY: tuple[str, ...] = (
    "A thorn reaches CarpetX through the two headers `CarpetX/interface.ccl` exports from `CarpetX/src/include/`: `carpetx.hxx` for a general thorn, and `carpetx_rk.hxx` for the time integrator. "
    "Every other header under `CarpetX/src/` includes `internal.hxx`, which refuses a compilation that is not CarpetX's own; `python3 scripts/lint check --only boundary` holds this, `python3 scripts/lint check --help` says what each rule holds, and `agent_docs/carpetx-internals.md` maps the inside.",
    f"`{THORN}` is a client of both headers: its `interface.ccl` and `configuration.ccl` are what a client declares, and `{SOURCES}/` exercises every name below but those on the last line of this preamble. "
    f"A refusal named under a section is a parfile of `{ERRORS}/`: the description of its mode in `{PARAM}` says what the run does, and its `.expect` file is the text CarpetX aborts with. "
    "Every call is available between the driver's SetupGH and its Shutdown, and a bad argument is a `CCTK_VERROR`, never garbage.",
)
UNEXERCISED_INTRO = "Held by nothing but the headers: "
UNEXERCISED = ("restrict_no_poison", "scratch")
SECTIONS: tuple[Section, ...] = (
    Section(
        "Indices and handles",
        (
            "`vi_t` is a variable's place in its group and `var_t` the flesh's variable index. "
            "A handle is made of indices and resolved on every call, so a stale one is reported, never dereferenced. "
            "`window_t` is a half-open range of levels on a half-open range of patches: `levels` spans every patch, `narrow` keeps the patches of its window, and `coarsest` and `finest` are on its first patch. "
            "A window checks its bounds when it is built, so the error names the line that built it.",
        ),
        exercised_by=("compile_checks.cxx", "TestCarpetXAPI_Fill"),
        refusals=("level_missing", "timelevel_missing", "parent_of_level_0", "window_empty", "window_out_of_bounds", "window_outside_traversal", "hierarchy_absent"),
    ),
    Section(
        "Data, validity and the write scope",
        (
            "A write scope sets the validity of the group at `tl` to exactly `leaves_valid` when it opens and poisons the regions it does not promise, so the scope must not read them; when it closes, the promised regions are scanned for poison and NaN. "
            "While it is open, a communication operation, `invalidate` or a scheduled function touching the group is refused. "
            "`require_valid` also runs that scan over `where`. A validity bit is set only by a write scope or a communication operation.",
            "`scratch(group)` is shaped like `data(group, tl_t{0})` and comes from a per-group pool its destructor returns it to; the pool dies with its level on a regrid, after which a `scratch_t` is an error, not a dangling pointer. It has no validity bits.",
        ),
        exercised_by=("TestCarpetXAPI_Fill",),
        refusals=("write_twice", "write_unfilled", "require_invalid", "invalidate_during_write", "schedule_during_write"),
    ),
    Section(
        "Communication",
        (
            "An operation without a `tl` acts on every time level but the oldest. "
            "`sync` is the scheduled SYNC, subcycling under `use_subcycling`, and sets the ghost and outer bits it fills; `sync_ghosts_only` is its same-level copy alone and `prolongate_ghosts` its coarse-to-fine interpolation alone; `prolongate_aligned` is `prolongate_ghosts` that under subcycling fills only levels time-aligned with their parent. "
            "`restrict_no_poison` leaves the coarse ghost and outer regions as they were, where the scheduled restriction invalidates and poisons them. `synchronize` waits for the device.",
        ),
        exercised_by=("TestCarpetXAPI_Fill",),
        refusals=("sync_during_write", "prolongate_unwritten_parent", "sync_invalid_coarse"),
    ),
    Section(
        "The hierarchy and the parameters",
        (
            "The BoxArray and DistributionMapping of a level are those of any group's MultiFab on it, and the refinement ratio is 2. "
            "Driver parameters are read through the flesh, `SHARES: Driver` or `CCTK_ParameterGet`, and CarpetX checks its own.",
        ),
        exercised_by=("TestCarpetXAPI_Fill",),
        refusals=("too_many_levels", "output_unknown_variable"),
    ),
    Section(
        "The integrator's contract: `carpetx_rk.hxx`",
        (
            "Exactly one thorn, the time integrator, publishes what it integrates and reports the events of its RK step; CarpetX derives everything else, and runs without a registration until `open_rk_step`. "
            "Registration is `AT WRAGH`, once; the record freezes when that traversal ends. "
            "`dense_output` names the method whose dense output fills the refinement boundary under subcycling; it is named, not inferred from the stage count, and `none` cannot run under subcycling. "
            "`b` holds the final-update weight of every stage, all zero for a method that never updates the state.",
            "`run_schedule_group_at` sets `cctk_time` and `cctk_delta_time` on the given cGH and on every driver cGH for the traversal; pass `cctkGH->cctk_delta_time` unless the step itself changes, as an IMEX half step does.",
            "`open_rk_step` is called once the state holds u(t_n), outside any OpenMP parallel region, and stores the old state of every registered pair with storage on each level of the window. "
            "`rhs_computed` stores the stage band and feeds the flux registers with the stage's weight times that level's `CCTK_DELTA_TIME`. "
            "`state_updated` says the state holds the next stage value: under subcycling CarpetX copies the same-level ghosts of every refined level in the window, fills its refinement boundary from its own clocks, scans the ghosts, and marks the integrated groups' ghosts valid, and their outer boundary where every face has a symmetry or boundary; without subcycling the last `state_updated` applies the flux registers.",
            "Two promises no parfile holds. Under subcycling the SYNC of an integrated group on a refined level at iteration > 0 copies only the same-level ghosts and requires the ghost bit set already, by `state_updated` or by the driver's refill after a recovery; on level 0 it fills and marks every ghost itself. "
            "After a recovery the driver refills the integrated groups' refinement boundary before `CCTK_POST_RECOVER_VARIABLES`; the integrator schedules nothing for it.",
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


def closing(section: Section) -> list[str]:
    if not section.refusals:
        return []
    return ["Refusals: " + ", ".join(f"`{name}`" for name in section.refusals) + ".", ""]


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
        for paragraph in section.text:
            parts.extend([paragraph, ""])
        parts.extend(closing(section))
    return "\n".join(parts)
