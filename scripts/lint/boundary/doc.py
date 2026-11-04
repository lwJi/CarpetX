from __future__ import annotations

import os
import re
from typing import Callable

from .rules import AMREX_CALLBACKS, EXTERN_C, HIERARCHY_LAYERS, INCLUDE_LINE, LAYER_INCLUDE, LAYERS, OWNS, PRIVATE_DIR, RULES, UNDER_HIERARCHY, below, layer_path
from files import PROJECT

MAP = "agent_docs/carpetx-internals.md"
LINK_TIME = (
    ("hierarchy", "regrid", ("CactusAmrCore",)),
    ("hierarchy", "comm", ("apply_boundary_conditions", "get_interpolator")),
)
BASE_LAYERS = below("hierarchy")
DEFINITION = r"(?m)^[^;{{}}()\n]*\b(?:[\w:]+::)?{name}(?:::~?\w+)?\s*\([^;{{}}]*\)[^;{{}}]*\{{"
OUTSIDERS = (("cactus", "Cactus flesh"), ("amrex", "AMReX"), ("thorn", "a thorn's routine"))
LIFECYCLE = ("Initialise", "Evolve", "Shutdown")
STEP_CALL = re.compile(r"\b(CCTK_Traverse|CCTK_OutputGH)\s*\(")
STEP_ENTRIES = ("ScheduleTraverseGH", "OutputGH")
CALL_FUNCTION = re.compile(r"\b(CCTK_CallFunction)\s*\(")
PUBLIC_INCLUDE = re.compile(r"^\s*#\s*include\s*<(carpetx\w*\.hxx)>")
REGRID_ASK = "hierarchy_writer::regrid"
REGRID_ASK_CALL = re.compile(r"\b(" + REGRID_ASK + r")\s*\(")
STARTUP = layer_path("flesh", "startup.cxx")
NOT_A_CALL = (layer_path("flesh", "compile_checks.cxx"),)
REGISTRATION = re.compile(r"\bCCTK_(?:Overload\w+|RegisterGHExtension\w+|InterpRegisterOpLocalUniform)\s*\(\s*(?:ghext_handle\s*,\s*)?(\w+)")
TRAVERSAL = re.compile(r"\bCCTK_ScheduleTraverse\s*\(\s*\w+\s*,\s*\w+\s*,\s*(\w+)\s*\)")
AMREX_REENTRIES = ("regrid",)
AMREX_REENTRY = re.compile(r"\bamrcore->(" + "|".join(AMREX_REENTRIES) + r")\s*\(")
STACK_LEGEND = (
    "a layer sits above every layer it may include, which the layer rule holds, and a solid edge is an include it has today, read from its #include lines, with the headers on it",
    "where no solid edge joins a layer to the one below it, the order is the rule's, not a coupling of the code",
    "the two boxes are the door rule: driver() is called in the upper box, never in the lower",
    "a dashed edge is a declaration whose body is in the higher layer, the one way a call reaches up, at link time",
    "a node names its layer, and no header where nothing can include it",
)
BASE_DIRS = f"{', '.join(f'{layer}/' for layer in BASE_LAYERS[:-1])} or {BASE_LAYERS[-1]}/"
LOOP_LEGEND = (
    "the numbered edges are one pass of the schedule: the flesh starts the driver (1), the driver names a bin or asks for output (2), the flesh hands it to the driver's entry (3), which asks the flesh to walk the bin (4) through CallFunction (5), which runs each routine (6), which may call the public headers (7), whose bodies are the .cxx files that include them first",
    "the unnumbered edges are a regrid: the driver asks the hierarchy, the hierarchy asks AMReX, and AMReX calls back into the layer that defines its callbacks",
    "a layer reaches a higher one only through an outsider, the Cactus flesh or AMReX, or at link time through a dashed edge of the stack",
    "the table below lists every entry an outsider calls",
)


def legend(lines: list[str], texts: tuple[str, ...]) -> str:
    lines.extend(f"%% {text}" for text in texts)
    return "\n".join(lines) + "\n"


def rules_of(layer: str) -> list[str]:
    return [rule.name for rule in RULES if layer in rule.home]


def sources(root: str, layer: str) -> list[str]:
    folder = os.path.join(root, "CarpetX", "src", layer)
    return [os.path.join(folder, f) for f in sorted(os.listdir(folder)) if f.endswith((".cxx", ".hxx"))]


def read(path: str) -> str:
    with open(path) as f:
        return f.read()


def headerless(root: str) -> tuple[str, ...]:
    return tuple(layer for layer in LAYERS if not any(path.endswith(".hxx") for path in sources(root, layer)))


def node(layer: str, no_header: tuple[str, ...], indent: str = "  ") -> str:
    label = f"{layer}/"
    if layer in no_header:
        label += " no header"
    return f'{indent}{layer}["{label}"]'


def layer_includes(root: str, layer: str) -> set[tuple[str, str]]:
    found: set[tuple[str, str]] = set()
    for path in sources(root, layer):
        with open(path) as f:
            for line in f:
                m = LAYER_INCLUDE.match(line)
                if not m:
                    continue
                target, _, header = (m.group(1) or m.group(2)).partition("/")
                if header and target in LAYERS and target != layer:
                    found.add((target, header))
    return found


def includes(root: str = PROJECT) -> dict[tuple[str, str], tuple[str, ...]]:
    edges: dict[tuple[str, str], set[str]] = {}
    for layer in LAYERS:
        for target, header in layer_includes(root, layer):
            if target not in BASE_LAYERS:
                edges.setdefault((layer, target), set()).add(header)
    order = {layer: -i for i, layer in enumerate(LAYERS)}
    return {edge: tuple(sorted(edges[edge])) for edge in sorted(edges, key=lambda e: (order[e[0]], order[e[1]]))}


def shared_base(root: str = PROJECT) -> tuple[str, ...]:
    shared: set[tuple[str, str]] | None = None
    for layer in HIERARCHY_LAYERS[1:]:
        base = {edge for edge in layer_includes(root, layer) if edge[0] in BASE_LAYERS}
        shared = base if shared is None else shared & base
    return tuple(f"{target}/{header}" for target, header in sorted(shared or ()))


def stack_legend(root: str = PROJECT) -> tuple[str, ...]:
    undrawn = f"an edge into {BASE_DIRS} is not drawn"
    shared = shared_base(root)
    if shared:
        undrawn += f": every layer above hierarchy/ reads {', '.join(shared)}"
    return STACK_LEGEND + (undrawn,)


def stack(root: str = PROJECT) -> str:
    no_header = headerless(root)
    lines = ["flowchart TB", '  subgraph door["driver() allowed"]']
    lines.extend(node(layer, no_header, "    ") for layer in reversed(HIERARCHY_LAYERS))
    lines.extend(["  end", '  subgraph nodoor["no driver()"]'])
    lines.extend(node(layer, no_header, "    ") for layer in reversed(UNDER_HIERARCHY))
    lines.extend(["  end", "  " + " ~~~ ".join(reversed(LAYERS))])
    for (source, target), headers in includes(root).items():
        lines.append(f'  {source} -- "{", ".join(headers)}" --> {target}')
    for low, high, names in LINK_TIME:
        assert LAYERS.index(low) < LAYERS.index(high), (low, high)
        lines.append(f'  {low} -. "{", ".join(names)}" .-> {high}')
    return legend(lines, stack_legend(root))


def defining_layer(root: str, name: str) -> str:
    shape = re.compile(DEFINITION.format(name=name))
    found = sorted({layer for layer in LAYERS for path in sources(root, layer) if path.endswith(".cxx") and shape.search(read(path))})
    if len(found) != 1:
        raise ValueError(f"{name} is defined in {', '.join(found) or 'no layer'}, not in one layer")
    return found[0]


def registrations(root: str = PROJECT) -> list[str]:
    return list(dict.fromkeys(REGISTRATION.findall(read(os.path.join(root, STARTUP)))))


def extern_c_files(root: str = PROJECT) -> list[str]:
    return [os.path.basename(path) for path in sources(root, "flesh") if path.endswith(".cxx") and EXTERN_C.search(read(path))]


def call_sites(root: str, pattern: re.Pattern) -> dict[str, list[str]]:
    sites: dict[str, list[str]] = {}
    for layer in LAYERS:
        for path in sources(root, layer):
            if path.endswith(".cxx") and os.path.relpath(path, root) not in NOT_A_CALL:
                for name in pattern.findall(read(path)):
                    sites.setdefault(layer, []).append(name)
    return {layer: list(dict.fromkeys(names)) for layer, names in sites.items()}


def landing(root: str, names: tuple[str, ...]) -> dict[str, list[str]]:
    lands: dict[str, list[str]] = {}
    for name in names:
        lands.setdefault(defining_layer(root, name), []).append(name)
    return lands


def top_down(found: dict[str, list[str]] | dict[str, tuple[str, ...]]) -> list[tuple[str, list[str] | tuple[str, ...]]]:
    return sorted(found.items(), key=lambda item: -LAYERS.index(item[0]))


def public_bodies(root: str = PROJECT) -> dict[str, tuple[str, ...]]:
    bodies: dict[str, set[str]] = {}
    for layer in LAYERS:
        for path in sources(root, layer):
            if not path.endswith(".cxx"):
                continue
            for line in read(path).splitlines():
                if INCLUDE_LINE.match(line):
                    m = PUBLIC_INCLUDE.match(line)
                    if m:
                        bodies.setdefault(layer, set()).add(m.group(1))
                    break
    return {layer: tuple(sorted(headers)) for layer, headers in bodies.items()}


def the_loop(root: str = PROJECT) -> str:
    registered = registrations(root)
    missing = [name for name in LIFECYCLE + STEP_ENTRIES if name not in registered]
    if missing:
        raise ValueError(f"{', '.join(missing)} not registered in {STARTUP}")
    traversals = top_down(call_sites(root, TRAVERSAL))
    handlers = tuple(dict.fromkeys(name for _, names in traversals for name in names))
    asked = landing(root, (REGRID_ASK,))
    steps: dict[str, list[tuple[str, str, str]]] = {
        "1": [("cactus", f"1 {', '.join(names)}", layer) for layer, names in top_down(landing(root, LIFECYCLE))],
        "2": [(layer, f"2 {', '.join(names)}", "cactus") for layer, names in top_down(call_sites(root, STEP_CALL))],
        "3": [("cactus", f"3 {', '.join(names)}", layer) for layer, names in top_down(landing(root, STEP_ENTRIES))],
        "4": [(layer, "4 CCTK_ScheduleTraverse", "cactus") for layer, _ in traversals],
        "5": [("cactus", f"5 {', '.join(names)}", layer) for layer, names in top_down(landing(root, handlers))],
        "6": [(layer, f"6 {', '.join(names)}", "thorn") for layer, names in top_down(call_sites(root, CALL_FUNCTION))],
        "7": [("thorn", f"7 {', '.join(headers)}", layer) for layer, headers in top_down(public_bodies(root))],
        "the regrid ask": [(layer, REGRID_ASK, target) for target in asked for layer, _ in top_down(call_sites(root, REGRID_ASK_CALL)) if layer != target],
        "the AMReX reentry": [(layer, ", ".join(f"amrex::AmrCore::{name}" for name in names), "amrex") for layer, names in top_down(call_sites(root, AMREX_REENTRY))],
        "the AMReX callbacks": [("amrex", ", ".join(names), layer) for layer, names in top_down(landing(root, AMREX_CALLBACKS))],
    }
    empty = [step for step, found in steps.items() if not found]
    if empty:
        raise ValueError(f"the loop has no edge for {', '.join(empty)}: no call site or body in {os.path.join(root, PRIVATE_DIR)}")
    edges = [edge for found in steps.values() for edge in found]
    drawn = {end for source, _, target in edges for end in (source, target)}
    lines = ["flowchart LR"]
    lines.extend(f'  {name}["{label}"]' for name, label in OUTSIDERS if name in drawn)
    lines.extend(node(layer, ()) for layer in reversed(LAYERS) if layer in drawn)
    lines.extend(f'  {source} -- "{label}" --> {target}' for source, label, target in edges)
    return legend(lines, LOOP_LEGEND)


ENTRIES_HEADING = "## Who the outsiders call"
ENTRIES_INTRO = "An outsider decides when a layer works: it calls one of these entries. Each row is a layer, who calls into it, and what they call."


def entries(root: str = PROJECT) -> str:
    rows: list[tuple[str, str, str]] = [("flesh", "Cactus flesh", f'every extern "C" in {", ".join(extern_c_files(root))}')]
    rows += [(layer, "Cactus flesh", ", ".join(names)) for layer, names in top_down(landing(root, tuple(registrations(root))))]
    rows += [(layer, "a thorn", f"the functions of {', '.join(headers)}") for layer, headers in top_down(public_bodies(root))]
    rows += [(layer, "AMReX", ", ".join(names)) for layer, names in top_down(landing(root, AMREX_CALLBACKS))]
    rows.sort(key=lambda row: -LAYERS.index(row[0]))
    lines = [ENTRIES_HEADING, "", ENTRIES_INTRO, "", "| Layer | Caller | Entries |", "| --- | --- | --- |"]
    lines.extend(f"| `{layer}/` | {caller} | {what} |" for layer, caller, what in rows)
    return "\n".join(lines) + "\n"


OWNS_HEADING = "## What each layer owns"
OWNS_INTRO = "New code goes in the layer that owns what it works on, and a lower layer never decides for a higher one. Each row says what its layer owns and what it never decides, then names the rules whose home is in it."


def owns() -> str:
    assert tuple(OWNS) == LAYERS, tuple(OWNS)
    rows = [f"- `{layer}/`: {OWNS[layer]}" + (f" Rules at home here: {', '.join(rules_of(layer))}." if rules_of(layer) else "") for layer in reversed(LAYERS)]
    return "\n".join([OWNS_HEADING, "", OWNS_INTRO, ""] + rows) + "\n"


TITLE = "# CarpetX's internals"
PREAMBLE = (
    "`python3 scripts/lint doc --write` writes this file from `scripts/lint/boundary/rules.py`, `scripts/lint/boundary/doc.py`, "
    f"the `#include` lines and call sites of `CarpetX/src/` and the registrations in `{STARTUP}`; `scripts/lint/boundary/test_rules.py` fails when "
    "the file differs from what it writes, and the edit hook refuses an edit to it. A client thorn sees none of this: it reaches CarpetX "
    "only through the headers `agent_docs/carpetx-public-api.md` describes. `python3 scripts/lint check --help` prints what every rule "
    "named below holds."
)
SECTIONS: tuple[tuple[str, Callable[[str], str]], ...] = (("The stack", stack), ("The loop", the_loop))


def render(root: str = PROJECT) -> str:
    parts = [TITLE, "", PREAMBLE, ""]
    for title, draw in SECTIONS:
        parts.extend([f"## {title}", "", "```mermaid", draw(root).rstrip("\n"), "```", ""])
    parts.append(entries(root))
    parts.append(owns())
    return "\n".join(parts)


def declares_and_defines(root: str, low: str, high: str, name: str) -> bool:
    def found(layer: str, suffix: str, shape: re.Pattern) -> bool:
        for path in sources(root, layer):
            if path.endswith(suffix):
                with open(path) as f:
                    if shape.search(f.read()):
                        return True
        return False

    return found(low, ".hxx", re.compile(rf"\b{name}\b")) and found(high, ".cxx", re.compile(DEFINITION.format(name=name)))
