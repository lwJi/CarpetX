from __future__ import annotations

import os
import re
from typing import Callable

from .rules import AMREX_CALLBACKS, EXTERN_C, HIERARCHY_LAYERS, LAYER_INCLUDE, LAYERS, OWNS, RULES, UNDER_HIERARCHY, below, layer_path
from files import PROJECT

MAP = "agent_docs/carpetx-internals.md"
LINK_TIME = (
    ("hierarchy", "regrid", ("CactusAmrCore",)),
    ("hierarchy", "comm", ("apply_boundary_conditions", "get_interpolator")),
)
BASE_LAYERS = below("hierarchy")
DEFINITION = r"\b(?:[\w:]+::)?{name}(?:::~?\w+)?\s*\([^;{{}}]*\)[^;{{}}]*\{{"
OUTSIDERS = (("cactus", "Cactus flesh"), ("amrex", "AMReX"))
STARTUP = layer_path("flesh", "startup.cxx")
NOT_A_CALL = (layer_path("flesh", "compile_checks.cxx"),)
REGISTRATION = re.compile(r"\bCCTK_(?:Overload\w+|RegisterGHExtension\w+|InterpRegisterOpLocalUniform)\s*\(\s*(?:ghext_handle\s*,\s*)?(\w+)")
TRAVERSAL = re.compile(r"\bCCTK_ScheduleTraverse\s*\(\s*\w+\s*,\s*\w+\s*,\s*(\w+)\s*\)")
AMREX_REENTRIES = ("regrid",)
AMREX_REENTRY = re.compile(r"\bamrcore->(" + "|".join(AMREX_REENTRIES) + r")\s*\(")
STACK_LEGEND = (
    "a solid edge is may include: a layer includes the layer below it and, through it, every layer further down, which the layer rule holds",
    "the two boxes are the door rule: driver() is called in the upper box, never in the lower",
    "a dashed edge is a declaration whose body is in the higher layer, the one way a call reaches up, at link time",
    "a node names its layer, then the rules whose home is in it, and no header where nothing can include it",
)
INCLUDES_LEGEND = (
    "what each layer includes today, read from its #include lines, with the headers on the edges",
    f"{', '.join(f'{layer}/' for layer in BASE_LAYERS[:-1])} and {BASE_LAYERS[-1]}/ are not drawn: every layer reads them",
    "an edge missing here is a coupling CarpetX does not have",
)
DECIDES_LEGEND = (
    f"an edge from an outsider is a function the layer defines and the outsider calls: the registrations of {STARTUP}, the extern C entries, and the AMReX callbacks of CactusAmrCore",
    "an edge to an outsider is a call that comes back through an edge from it, so the outsider decides when the lower layer works",
    "a call from a layer to a higher one goes through an outsider or, at link time, through a dashed edge of the stack",
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


def node(layer: str, no_header: tuple[str, ...], indent: str = "  ", rules: bool = False) -> str:
    label = f"{layer}/"
    if layer in no_header:
        label += " no header"
    if rules and rules_of(layer):
        label += " " + ", ".join(rules_of(layer))
    return f'{indent}{layer}["{label}"]'


def stack(root: str = PROJECT) -> str:
    no_header = headerless(root)
    lines = ["flowchart TB", '  subgraph door["driver() allowed"]']
    lines.extend(node(layer, no_header, "    ", rules=True) for layer in reversed(HIERARCHY_LAYERS))
    lines.extend(["  end", '  subgraph nodoor["no driver()"]'])
    lines.extend(node(layer, no_header, "    ", rules=True) for layer in reversed(UNDER_HIERARCHY))
    lines.extend(["  end", "  " + " --> ".join(reversed(LAYERS))])
    for low, high, names in LINK_TIME:
        assert LAYERS.index(low) < LAYERS.index(high), (low, high)
        lines.append(f'  {low} -. "{", ".join(names)}" .-> {high}')
    return legend(lines, STACK_LEGEND)


def includes(root: str = PROJECT) -> dict[tuple[str, str], tuple[str, ...]]:
    edges: dict[tuple[str, str], set[str]] = {}
    for layer in LAYERS:
        for path in sources(root, layer):
            with open(path) as f:
                for line in f:
                    m = LAYER_INCLUDE.match(line)
                    if not m:
                        continue
                    target, _, header = (m.group(1) or m.group(2)).partition("/")
                    if header and target in LAYERS and target != layer and target not in BASE_LAYERS:
                        edges.setdefault((layer, target), set()).add(header)
    order = {layer: -i for i, layer in enumerate(LAYERS)}
    return {edge: tuple(sorted(edges[edge])) for edge in sorted(edges, key=lambda e: (order[e[0]], order[e[1]]))}


def include_graph(root: str = PROJECT) -> str:
    no_header = headerless(root)
    lines = ["flowchart TB"]
    lines.extend(node(layer, no_header) for layer in reversed(LAYERS) if layer not in BASE_LAYERS)
    for (source, target), headers in includes(root).items():
        lines.append(f'  {source} -- "{", ".join(headers)}" --> {target}')
    return legend(lines, INCLUDES_LEGEND)


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


def who_decides(root: str = PROJECT) -> str:
    from_cactus = landing(root, tuple(registrations(root)))
    from_amrex = landing(root, AMREX_CALLBACKS)
    to_cactus = call_sites(root, TRAVERSAL)
    to_amrex = call_sites(root, AMREX_REENTRY)
    drawn = set(from_cactus) | set(from_amrex) | set(to_cactus) | set(to_amrex) | {"flesh"}
    lines = ["flowchart LR"]
    lines.extend(f'  {name}["{label}"]' for name, label in OUTSIDERS)
    lines.extend(node(layer, ()) for layer in reversed(LAYERS) if layer in drawn)
    lines.append(f'  cactus -- "extern C in {", ".join(extern_c_files(root))}" --> flesh')
    for layer in reversed(LAYERS):
        if layer in from_cactus:
            lines.append(f'  cactus -- "{", ".join(from_cactus[layer])}" --> {layer}')
    for layer in reversed(LAYERS):
        if layer in to_cactus:
            lines.append(f'  {layer} -- "CCTK_ScheduleTraverse with {", ".join(to_cactus[layer])}" --> cactus')
    for layer in reversed(LAYERS):
        if layer in to_amrex:
            lines.append(f'  {layer} -- "{", ".join(to_amrex[layer])}" --> amrex')
    for layer in reversed(LAYERS):
        if layer in from_amrex:
            lines.append(f'  amrex -- "{", ".join(from_amrex[layer])}" --> {layer}')
    return legend(lines, DECIDES_LEGEND)


OWNS_HEADING = "## What each layer owns"
OWNS_INTRO = "New code goes in the layer that owns what it works on, and a lower layer never decides for a higher one. Each row says what its layer owns and what it never decides."


def owns() -> str:
    assert tuple(OWNS) == LAYERS, tuple(OWNS)
    rows = [f"- `{layer}/`: {OWNS[layer]}" for layer in reversed(LAYERS)]
    return "\n".join([OWNS_HEADING, "", OWNS_INTRO, ""] + rows) + "\n"


TITLE = "# CarpetX's internals"
PREAMBLE = (
    "`python3 scripts/lint doc --write` writes this file from `scripts/lint/boundary/rules.py`, `scripts/lint/boundary/doc.py`, "
    f"the `#include` lines of `CarpetX/src/` and the registrations in `{STARTUP}`; `scripts/lint/boundary/test_rules.py` fails when "
    "the file differs from what it writes, and the edit hook refuses an edit to it. A client thorn sees none of this: it reaches CarpetX "
    "only through the headers `agent_docs/carpetx-public-api.md` describes. `python3 scripts/lint check --help` prints what every rule "
    "named below holds."
)
SECTIONS: tuple[tuple[str, Callable[[str], str]], ...] = (("The stack", stack), ("The includes", include_graph), ("Who decides", who_decides))


def render(root: str = PROJECT) -> str:
    parts = [TITLE, "", PREAMBLE, ""]
    for title, draw in SECTIONS:
        parts.extend([f"## {title}", "", "```mermaid", draw(root).rstrip("\n"), "```", ""])
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
