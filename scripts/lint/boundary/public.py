from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

from .doc import API, MAP, PUBLIC_HEADERS, read
from .doc import render as render_map
from .rules import DECLARATION_START, PRIVATE_DIR, PUBLIC_DIR, STATEMENT_END, declaration_head, declared_function, statements
from files import PROJECT

SOURCE = "scripts/lint/boundary/public.py"
CLIENT = "TestCarpetXAPI"
MODES = f"{CLIENT}/param.ccl"
ERRORS = f"{CLIENT}/test/errors/"
TESTS = f"{CLIENT}/test/"
INTERFACE = "CarpetX/interface.ccl"

TYPE = re.compile(r"^(?:template\s*<[^>]*>\s*)?(struct|class|enum(?:\s+class)?)\s+(?:\[\[\w+\]\]\s*)?(\w+)\s*(?::(?!:)|[{;])")
ALIAS = re.compile(r"^using\s+(\w+)\s*=")
NAMESPACE = re.compile(r"^(?:inline\s+)?namespace\b")
ACCESS = re.compile(r"^\s*(public|protected|private)\s*:(?!:)", re.MULTILINE)
FRIEND = re.compile(r"\bfriend\b")
QUIET = re.compile(r'"(?:[^"\\]|\\.)*"|//[^\n]*')
WORD = re.compile(r"[A-Za-z_]\w*")
SPAN = re.compile(r"`([^`\n]+)`")
QUALIFIED = re.compile(r"\b([A-Za-z_]\w*)::([A-Za-z_]\w*)")
INCLUDE = re.compile(r'^\s*#\s*include\s*[<"]([^>"]+)[>"]', re.MULTILINE)
MODE_BLOCK = re.compile(r'^KEYWORD mode\b[^\n]*\n\{(.*?)^\}\s*"([^"]+)"', re.MULTILINE | re.DOTALL)
MODE_ROW = re.compile(r'^\s*"([^"]+)"\s*::\s*"([^"]*)"', re.MULTILINE)
MODE_SET = re.compile(rf'^\s*{CLIENT}::mode\s*=\s*"([^"]*)"', re.MULTILINE)
NAMESPACE_NAME = "CarpetX"


@dataclass
class Frame:
    kind: str
    name: str = ""
    access: str = "public"
    members: dict[str, None] = field(default_factory=dict)


@dataclass(frozen=True)
class Declared:
    types: dict[str, tuple[str, ...]]
    functions: tuple[str, ...]

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self.types) + self.functions


def enumerators(body: str) -> list[str]:
    found = []
    for piece in body.split(","):
        m = WORD.search(piece)
        if m:
            found.append(m.group())
    return found


def declare(head: str, clean: str, frame: Frame, types: dict[str, dict[str, None]], functions: dict[str, None]) -> Frame:
    m = TYPE.match(head)
    if m:
        kind, name = m.group(1), m.group(2)
        members: dict[str, None] = {}
        if frame.kind == "namespace":
            members = types.setdefault(name, {})
        elif frame.access == "public":
            frame.members[name] = None
        if kind.startswith("enum"):
            members.update(dict.fromkeys(enumerators(clean.partition("{")[2].partition("}")[0])))
            return Frame("enum", name, members=members)
        return Frame("type", name, "private" if kind == "class" else "public", members)
    m = ALIAS.match(head)
    if m:
        if frame.kind == "namespace":
            types.setdefault(m.group(1), {})
        elif frame.access == "public":
            frame.members[m.group(1)] = None
        return Frame("block")
    if NAMESPACE.match(head):
        return Frame("namespace")
    name = declared_function(head.partition("{")[0], in_header=True)
    if name is not None and not name.startswith("operator") and "::" not in name and name != frame.name:
        if frame.kind == "namespace" or FRIEND.search(head):
            functions[name] = None
        elif frame.access == "public":
            frame.members[name] = None
    return Frame("block")


def declarations(text: str) -> Declared:
    types: dict[str, dict[str, None]] = {}
    functions: dict[str, None] = {}
    stack: list[Frame] = []
    for _, line, statement in statements(text):
        if not STATEMENT_END.search(line):
            continue
        clean = QUIET.sub("", statement)
        frame = stack[-1] if stack else Frame("namespace")
        for access in ACCESS.findall(clean):
            frame.access = access
        head = declaration_head(clean).strip()
        opened = Frame("block")
        if frame.kind == "enum":
            frame.members.update(dict.fromkeys(enumerators(clean.partition("}")[0])))
        elif frame.kind in ("namespace", "type") and DECLARATION_START.match(head):
            opened = declare(head, clean, frame, types, functions)
        for brace in re.findall(r"[{}]", clean):
            if brace == "{":
                stack.append(opened)
                opened = Frame("block")
            elif stack:
                stack.pop()
    return Declared({name: tuple(members) for name, members in types.items()}, tuple(functions))


def declared(root: str = PROJECT) -> dict[str, Declared]:
    return {header: declarations(read(os.path.join(root, header))) for header in PUBLIC_HEADERS}


Topic = tuple[str, tuple[str, ...], tuple[str, ...]]


def series(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


TITLE = "# CarpetX's public API"
PREAMBLE = (
    f"`python3 scripts/lint doc --write` writes this file from `{SOURCE}`, the public headers and `{CLIENT}`, and the edit hook refuses an edit to it. "
    f"A topic names every declaration of the headers, and every member of a type where it names any; a refusal a topic names is a parfile of `{ERRORS}` that sets that mode of `{MODES}`, with what CarpetX says in the `.expect` beside it."
)
INTRO = (
    f"A thorn reaches CarpetX through two headers that `{INTERFACE}` exports from `{PUBLIC_DIR}`: `carpetx.hxx` for everything a general thorn needs, and `carpetx_rk.hxx` for the time integrator's contract. "
    f"Every other header under `{PRIVATE_DIR}` is private and the public headers include none of it; `{MAP}` is the map of what is behind them.",
)
TOPICS: tuple[Topic, ...] = (
    (
        "Using the headers",
        ("hierarchy-absent",),
        (
            "In `interface.ccl`, `USES INCLUDE HEADER: carpetx.hxx` (and `carpetx_rk.hxx` for an integrator); in `configuration.ccl`, `REQUIRES CarpetX`, which also provides the AMReX, Loop and Arith headers that `carpetx.hxx` includes; in C++, `#include <carpetx.hxx>` and qualify every name with `CarpetX::`. The headers add no using-directive at any scope. `Loop::GridDescBase` is a Loop type and comes from `<loop.hxx>`, not from CarpetX.",
            "The header is the list of calls; this file states only what a signature cannot. Every call is available between the driver's SetupGH and its Shutdown, and a bad argument is a `CCTK_VERROR`, never garbage.",
        ),
    ),
    (
        "Indices and handles",
        ("level-missing", "timelevel-missing", "window-outside-traversal"),
        (
            "An index is an `index_t`: `patch_t` names a patch, `gi_t` a Cactus group, `tl_t` a time level, `vi_t` a variable by its place in its group, and `var_t` a variable as the flesh numbers it. Each is built explicitly from the flesh's `int`, `patch_t{cctk_patch}`, `gi_t{CCTK_GroupIndex(\"…\")}`, `tl_t{0}` or `var_t{CCTK_VarIndex(\"…\")}`, and read back only through `index()`, where a value leaves for the flesh (`CCTK_FullGroupName(gi.index())`). An index does not convert to an `int` or from one implicitly, nor to another kind of index, and it has no arithmetic; two of a kind compare equal and order, and `index_hash` is their `std::hash`. The one named step is `tl_t::next()`, the following time level, which is refused where it is used if the group has no such level. `variable(var)` splits a flesh variable index into `variable_t{gi, vi}`, its group and its place there; `valid` and `reduce` take that `vi_t`. `TestCarpetXAPI/src/compile_checks.cxx` holds the `static_assert`s that pin this.",
            "A handle is made of indices and resolved on every call, so a stale one names a level that no longer exists and the call reports it instead of touching freed memory. `level_t{patch_t{cctk_patch}, cctk_level}` names a refinement level of a patch; it has no public fields, is read back through `patch()` and `index()`, and has no arithmetic: `level.parent()` is the next coarser level and an error on level 0, and `level.child()` is the next finer one, empty on the finest level of its patch. `group_t{level, gi}` names a grid-function group on a level (`gi` is a `gi_t`). `num_patches()` is the number of patches, and `num_levels(patch)` takes a `patch_t` and is per patch. `window_t` is a half-open range of levels on a half-open range of patches, both for traversal (`loop_serially`, `loop_coarse_to_fine`, and `loop_parallel`, whose kernel receives the component's cGH) and as the \"which levels\" argument of every operation. It has no public fields and is built only by `active_window()`, the window of the current traversal and an error outside one, `all_levels()`, the whole hierarchy, `window_t::one_level(level)`, `window_t::levels(min_level, max_level)` over every patch, and `window.narrow(min_level, max_level)` over the same patches as `window`. Each checks the bounds against the hierarchy when it is called, so a bad window is an error that names the line that built it, not the operation that later uses it. The two ints of `levels` and `narrow` are the bounds of a range, not levels, and `window.level_bounds()` and `window.patch_bounds()` read them back as a `window_t::bounds_t`, a `{min, max}` pair of ints. `window.coarsest()` and `window.finest()` are its lowest and highest level on its first patch, and an error on an empty window.",
        ),
    ),
    (
        "Data, validity and the write scope",
        ("write-twice", "sync-during-write", "write-unfilled", "require-invalid"),
        (
            "`data(group, tl)` is the `const amrex::MultiFab &` of a time level; AMReX's const overloads give `Array4<const T>`. Writing goes through a scope, a `write_t`: `auto w = write(group, tl, leaves_valid, \"what\"); w.mfab()`, and `mfab()` is for an lvalue only, so the scope cannot be a temporary. When it opens, every variable of the group at `tl` gets exactly `leaves_valid` as its validity, with reason `\"what (iteration N)\"`, and the regions it does not promise are poisoned, so the scope must not read them. When it is destroyed, the promised regions are scanned for poison and NaN, as a scheduled function's WRITES are. While it is open, a second `write` on the same group and time level, any communication operation or `invalidate` on the group, and scheduling a function that reads, writes or syncs the group are all errors.",
            "`valid_t` is three bits, interior, outer boundary and ghosts; `make_valid_int()`, `make_valid_outer()`, `make_valid_ghosts()` and `make_valid_all()` build one with one bit set, or every bit, and `~`, `&` and `|` combine them. `valid_all()` and `valid_any()` fold the bits, and `explanation()` is the text an error prints for them. `valid(group, tl, vi)` reads the bits of a variable. `require_valid(group, tl, where, \"why\")` raises an error with CarpetX's explanation if any variable of the group lacks a bit of `where`, and runs the poison and NaN scan. `invalidate(window, groups, tl, \"why\")` clears every bit of every variable at `tl`. No thorn outside CarpetX sets a validity bit: the write scope and the communication operations are the only way a bit is set.",
            "`scratch(group)` is a `scratch_t`, a move-only buffer with the boxes, distribution, components and ghosts of `data(group, tl_t{0})`, drawn from a per-group pool that CarpetX keeps between calls; `mfab()` is its MultiFab. There is no init or free call; the destructor returns the buffer. The pool dies with the level on a regrid, and using a `scratch_t` after that is an error, not a dangling pointer. It has no validity bits.",
        ),
    ),
    (
        "Communication",
        ("prolongate-unwritten-parent", "sync-invalid-coarse"),
        (
            "Each operation takes a window and a list of grid-function groups as a `std::span<const gi_t>`, so a `std::vector<gi_t>` or a `std::array<gi_t, N>` converts in place (a braced list does not; write `std::array{gi}` for one group), and is an error while a write scope is open on any of them. An operation requires what it reads valid, the parent level of a prolongation and, under subcycling, the coarser level a sync fills the refinement boundary from; finding it invalid is an error.",
            "- `sync(window, groups)` is a scheduled SYNC: every time level but the oldest, the subcycling variant when `use_subcycling`, and it sets the ghost and outer bits it fills. Under subcycling the SYNC of an integrated group on a refined level at iteration > 0 copies only the same-level ghosts and requires the ghost bit set already, by `state_updated` or by the driver's refill after a recovery; finding it unset is an error. On level 0 it fills and marks every ghost itself.\n"
            "- `sync_ghosts_only(window, groups)`: the same-level copy only, on every time level but the oldest; `sync_ghosts_only(window, groups, tl)` does that one time level.\n"
            "- `prolongate_ghosts(window, groups)`: coarse-to-fine interpolation only, on every time level but the oldest; marks the ghost and outer bits of the levels it filled. `prolongate_ghosts(window, groups, tl)` does that one time level.\n"
            "- `prolongate_aligned(window, groups)` and `prolongate_aligned(window, groups, tl)`: as `prolongate_ghosts`, but under subcycling only into levels time-aligned with their parent.\n"
            "- `restrict_no_poison(window, groups)`: fine-to-coarse over every level pair in the window.\n"
            "- `synchronize()` waits for the device.",
            "`reduce(gi, vi, tl)` is a `reduction<T, D>`, whose fields and norms are what their names say; so are `Timer` and `Interval`, `geometry(level)`, `symmetries(level)`, a `symmetry_t` per face, `numvars`, `indextype` and `nghostzones`. `rat64`, the type of `iteration(level)` and `delta_iteration(level)`, is a rational. The BoxArray and DistributionMapping of a level are those of any group's MultiFab on it, and the refinement ratio is 2 between every pair of levels. Driver parameters are read through the flesh (`SHARES: Driver` in `param.ccl`, or `CCTK_ParameterGet`), not through the headers.",
        ),
    ),
    (
        "The integrator's contract: `carpetx_rk.hxx`",
        (
            "register-twice",
            "register-at-startup",
            "register-at-paramcheck",
            "register-after-use",
            "dense-output-stage-mismatch",
            "weights-sum-mismatch",
            "reflux-without-weights",
            "subcycling-without-dense-output",
            "integrated-not-checkpointed",
            "unregistered-rk-call",
            "rhs-repeated",
            "step-closed-early",
        ),
        (
            "Exactly one thorn, the time integrator, publishes what it integrates and reports the events of its RK step; CarpetX derives everything else.",
            "Registration is once, `AT WRAGH`: `register_integrator(rk_config_t{dense_output, b, integrated})`. The driver freezes the record when its `WRAGH` traversal ends, so a registration before `SetupGH` or after `WRAGH` is refused, and so is an RK operation before the freeze. `dense_output` is a `rk_dense_output_t` and names the method whose dense output fills the refinement boundary under subcycling (`ssprk3`, `rk4` or `none`); it is named, not inferred from the stage count, so a 3-stage method other than SSPRK3 registers `none` and cannot run under subcycling, and a method that names one has the stage count `rk_dense_output_stages(dense_output)` gives. `b` holds the final-update weight of every stage, so `nstages()` is `b.size()`, and the weights sum to 1 unless all are zero, for a method that never updates the state; an implicit method registers an empty `b`, so `feeds_flux_registers()` is false and it never feeds a flux register, which is refused under reflux for a group that carries fluxes. `integrated` holds `rk_pair_t{state, rhs}` pairs of `gi_t`, and `is_integrated(gi)` says whether `gi` is the state of one; every state is checkpointed and evolved, named once, and never a right-hand side. CarpetX checks the record against its own parameters and refuses a bad one. A driver without a registration still runs; only `open_rk_step` needs one.",
            "`run_schedule_group_at(cctkGH, \"G\", t, delta_time)` sets `cctk_time` and `cctk_delta_time` on the given cGH and on every driver cGH, traverses the group, and restores both. `delta_time` is the raw `cctk_delta_time` the group sees, so pass `cctkGH->cctk_delta_time` unless the step itself changes, as an IMEX half step does.",
            "The step is an `rk_step_t`, `auto step = open_rk_step(window)`, opened once the state holds u(t_n) and outside any OpenMP parallel region. It stores the old state of every registered pair with storage on each level of the window and requires their interior valid. Then `rhs_computed()` and `state_updated()` alternate, `b.size()` times each, and the destructor checks that every stage ran. `rhs_computed` requires the interior of every registered right-hand side valid on every level of the window, stores the stage band, and feeds the flux registers with the stage's weight times that level's `CCTK_DELTA_TIME`. `state_updated` says the state now holds the next stage value: under subcycling CarpetX copies the same-level ghosts of every refined level in the window, fills its refinement boundary from its own clocks, scans the ghosts, and marks the integrated groups' ghosts valid, and their outer boundary where every face has a symmetry or boundary; without subcycling the last `state_updated` applies the flux registers. A call out of order, a call after the last stage, and a step destroyed early are errors naming the window and the stage; no stage index is passed, so stages cannot run out of order. After a recovery the driver refills the integrated groups' refinement boundary itself before it traverses `CCTK_POST_RECOVER_VARIABLES`; the integrator schedules nothing for it.",
        ),
    ),
)
DECLARES_HEADING = "## What the headers declare"
REFUSED = "Refused: "


def words(paragraph: str) -> set[str]:
    return {word for span in SPAN.findall(paragraph) for word in WORD.findall(span)}


def paragraphs() -> list[tuple[str, str]]:
    return [(title, paragraph) for title, _, texts in TOPICS for paragraph in texts]


def coverage(found: dict[str, Declared]) -> list[str]:
    named = [(title, words(paragraph)) for title, paragraph in paragraphs()]
    everywhere = set().union(*(names for _, names in named))
    problems = []
    types = {name: members for declared in found.values() for name, members in declared.types.items()}
    functions = {name for declared in found.values() for name in declared.functions}
    for header, declared in found.items():
        for name in declared.names:
            if name not in everywhere:
                problems.append(f"{name}, declared in {header}, is named in no topic of {SOURCE}")
        for name, members in declared.types.items():
            enumerated = {member for member in members if any(name in names and member in names for _, names in named)}
            if enumerated and enumerated != set(members):
                problems.append(f"a topic of {SOURCE} names {', '.join(f'{name}::{member}' for member in members if member in enumerated)} but not {', '.join(f'{name}::{member}' for member in members if member not in enumerated)}: a topic that names a member of {name} names them all")
    members = {member for found_members in types.values() for member in found_members}
    for title, paragraph in paragraphs():
        for span in SPAN.findall(paragraph):
            for word in WORD.findall(span):
                if word.endswith("_t") and word not in types and word not in members:
                    problems.append(f"`{word}` in the topic {title!r} of {SOURCE} is no public name")
            for scope, member in QUALIFIED.findall(span):
                if scope in types and member not in types[scope]:
                    problems.append(f"`{scope}::{member}` in the topic {title!r} of {SOURCE} is no public member of {scope}")
                if scope == NAMESPACE_NAME and member not in types and member not in functions:
                    problems.append(f"`{scope}::{member}` in the topic {title!r} of {SOURCE} is no public name")
    return problems


def paths(root: str) -> list[str]:
    problems = []
    texts = (PREAMBLE,) + INTRO + tuple(paragraph for _, paragraph in paragraphs())
    for text in texts:
        for span in SPAN.findall(text):
            if "/" in span and " " not in span and not os.path.exists(os.path.join(root, span)):
                problems.append(f"`{span}` in {SOURCE} does not exist")
    return problems


def private_basenames(root: str) -> set[str]:
    found = set()
    for folder, _, files in os.walk(os.path.join(root, PRIVATE_DIR)):
        if os.path.abspath(folder) != os.path.abspath(os.path.join(root, PUBLIC_DIR.rstrip("/"))):
            found.update(files)
    return found


def boundary(root: str) -> list[str]:
    problems = []
    private = private_basenames(root)
    public = {os.path.basename(header) for header in PUBLIC_HEADERS}
    for header in PUBLIC_HEADERS:
        for included in INCLUDE.findall(read(os.path.join(root, header))):
            if os.path.basename(included) in private and os.path.basename(included) not in public:
                problems.append(f"{header} includes {included}, a private file of {PRIVATE_DIR}")
    interface = read(os.path.join(root, INTERFACE))
    for name in public:
        for line in (f"USES INCLUDE HEADER: {name}", f"INCLUDES HEADER: {name} IN {name}"):
            if line not in interface:
                problems.append(f"{INTERFACE} lacks the line {line!r}")
    client = os.path.join(root, CLIENT, "src")
    sources = [read(os.path.join(client, name)) for name in sorted(os.listdir(client)) if name.endswith(".cxx")]
    for name in public:
        if not any(name in INCLUDE.findall(text) for text in sources):
            problems.append(f"no source of {CLIENT}/src/ includes <{name}>")
    return problems


def modes(root: str) -> tuple[str, dict[str, str]]:
    m = MODE_BLOCK.search(read(os.path.join(root, MODES)))
    if m is None:
        raise ValueError(f"{MODES} has no KEYWORD mode")
    return m.group(2), dict(MODE_ROW.findall(m.group(1)))


def parfile(mode: str) -> str:
    return mode.replace("-", "_") + ".par"


def mode_of(text: str, default: str) -> str:
    m = MODE_SET.search(text)
    return m.group(1) if m else default


def refusals(root: str) -> list[str]:
    default, found = modes(root)
    problems = []
    folder = os.path.join(root, ERRORS)
    parfiles = sorted(name for name in os.listdir(folder) if name.endswith(".par"))
    for mode in found:
        if mode == default:
            continue
        name = parfile(mode)
        path = os.path.join(folder, name)
        if not os.path.isfile(path):
            problems.append(f"the mode {mode} of {MODES} has no {ERRORS}{name}")
            continue
        if mode_of(read(path), default) != mode:
            problems.append(f"{ERRORS}{name} does not set {CLIENT}::mode = \"{mode}\"")
        expect = path[: -len(".par")] + ".expect"
        if not os.path.isfile(expect) or not read(expect).strip():
            problems.append(f"{ERRORS}{name} has no {os.path.basename(expect)} beside it, or it is empty")
    for name in parfiles:
        mode = mode_of(read(os.path.join(folder, name)), default)
        if mode != default and name != parfile(mode):
            problems.append(f"{ERRORS}{name} sets {CLIENT}::mode = \"{mode}\" but is not {parfile(mode)}")
    positive = os.path.join(root, TESTS, parfile(default))
    if not os.path.isfile(positive) or mode_of(read(positive), default) != default:
        problems.append(f"{TESTS}{parfile(default)} does not run the default mode {default}")
    claimed: dict[str, str] = {}
    for title, topic_modes, _ in TOPICS:
        for mode in topic_modes:
            if mode not in found or mode == default:
                problems.append(f"the topic {title!r} of {SOURCE} names the mode {mode}, which is not a refusal of {MODES}")
            elif mode in claimed:
                problems.append(f"the mode {mode} is a refusal of both {claimed[mode]!r} and {title!r} in {SOURCE}")
            claimed[mode] = title
    for mode in found:
        if mode != default and mode not in claimed:
            problems.append(f"the mode {mode} of {MODES} is a refusal of no topic of {SOURCE}")
    return problems


def declares(found: dict[str, Declared]) -> str:
    lines = [DECLARES_HEADING, ""]
    for header, declared in found.items():
        types = ", ".join(f"`{name}`" + (f" ({', '.join(f'`{member}`' for member in members)})" if members else "") for name, members in declared.types.items())
        functions = ", ".join(f"`{name}`" for name in declared.functions)
        lines.append(f"- `{os.path.basename(header)}`: the types {types}, and the functions {functions}")
    return "\n".join(lines) + "\n"


def render(root: str = PROJECT) -> str:
    found = declared(root)
    problems = coverage(found) + paths(root) + boundary(root) + refusals(root)
    if problems:
        raise ValueError("\n".join(problems))
    parts = [TITLE, "", PREAMBLE, "", *INTRO, "", declares(found)]
    for title, topic_modes, texts in TOPICS:
        parts.extend([f"## {title}", "", *(text + "\n" for text in texts)])
        if topic_modes:
            parts.append(REFUSED + series([f"`{parfile(mode)}`" for mode in topic_modes]) + ".\n")
    return "\n".join(parts)


GENERATED = {"internals": (MAP, render_map), "public-api": (API, render)}
