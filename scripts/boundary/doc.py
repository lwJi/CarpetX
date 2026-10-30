from __future__ import annotations

import os
from typing import Callable, Optional

from rules import (
    CODEC_FILES,
    HIERARCHY_LAYERS,
    INDEX_SOURCE_LAYERS,
    LAYER_INCLUDE,
    LAYERS,
    RECORD_VAR_FILE,
    RECORDS_LAYERS,
    SYNC_FILES,
    TRAVERSAL_FILES,
    UNDER_HIERARCHY,
    VALIDITY_FILES,
    below,
    layer_of,
)
from files import PROJECT  # noqa: E402

MAP = "agent_docs/carpetx-internals.md"
WRITES = "python3 scripts/boundary doc --write rewrites this"
NO_HEADER_LAYERS = ("regrid", "flesh")
LINK_TIME = (
    ("hierarchy", "regrid", ("CactusAmrCore",)),
    ("hierarchy", "comm", ("apply_boundary_conditions", "get_interpolator")),
)
BASE_LAYERS = below("hierarchy")
HOMES = {
    "traversal": TRAVERSAL_FILES,
    "validity": VALIDITY_FILES,
    "varindex": (RECORD_VAR_FILE,),
    "directions": SYNC_FILES,
    "codec": CODEC_FILES,
}
BINDS = {"index": INDEX_SOURCE_LAYERS, "records": RECORDS_LAYERS}


def mark(name: str) -> str:
    return f"%% {WRITES} block: {name}"


def rules_of(layer: str) -> list[str]:
    named = ["layer", "door"]
    named.extend(name for name, files in HOMES.items() if any(layer_of(f) == layer for f in files))
    named.extend(name for name, layers in BINDS.items() if layer in layers)
    return named


def node(layer: str, indent: str = "  ", rules: bool = False) -> str:
    label = f"{layer}/"
    if layer in NO_HEADER_LAYERS:
        label += " no header"
    if rules and rules_of(layer)[2:]:
        label += " " + ", ".join(rules_of(layer)[2:])
    return f'{indent}{layer}["{label}"]'


def stack() -> str:
    lines = [mark("the stack"), "flowchart TB", '  subgraph door["driver() allowed"]']
    lines.extend(node(layer, "    ", rules=True) for layer in reversed(HIERARCHY_LAYERS))
    lines.extend(["  end", '  subgraph nodoor["no driver()"]'])
    lines.extend(node(layer, "    ", rules=True) for layer in reversed(UNDER_HIERARCHY))
    lines.extend(["  end", "  " + " --> ".join(reversed(LAYERS))])
    for low, high, names in LINK_TIME:
        assert LAYERS.index(low) < LAYERS.index(high), (low, high)
        lines.append(f'  {low} -. "{", ".join(names)}" .-> {high}')
    return "\n".join(lines) + "\n"


def sources(root: str, layer: str) -> list[str]:
    folder = os.path.join(root, "CarpetX", "src", layer)
    return [os.path.join(folder, f) for f in sorted(os.listdir(folder)) if f.endswith((".cxx", ".hxx"))]


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
    lines = [mark("the includes"), "flowchart TB"]
    lines.extend(node(layer) for layer in reversed(LAYERS) if layer not in BASE_LAYERS)
    for (source, target), headers in includes(root).items():
        lines.append(f'  {source} -- "{", ".join(headers)}" --> {target}')
    return "\n".join(lines) + "\n"


def opens(name: str, lines: list[str], i: int) -> Optional[int]:
    fenced = lines[i].strip() == "```mermaid" and i + 1 < len(lines) and lines[i + 1] == mark(name)
    return i + 1 if fenced else None


def fence(line: str) -> bool:
    return line.strip() == "```"


REGIONS: tuple[tuple[str, Callable[[], str]], ...] = (("the stack", stack), ("the includes", include_graph))


def regions(text: str) -> dict[str, list[tuple[int, int]]]:
    lines = text.split("\n")
    found: dict[str, list[tuple[int, int]]] = {name: [] for name, _ in REGIONS}
    for name, _ in REGIONS:
        for i in range(len(lines)):
            start = opens(name, lines, i)
            if start is None:
                continue
            end = start
            while end < len(lines) and not fence(lines[end]):
                end += 1
            found[name].append((start, end))
    return found


def render() -> str:
    return "\n".join(draw() for _, draw in REGIONS)


def rewrite(text: str) -> str:
    found = regions(text)
    for name, places in found.items():
        if len(places) != 1:
            raise ValueError(f"{MAP} holds {len(places)} mermaid blocks opening with {mark(name)!r}, not one")
    lines = text.split("\n")
    for name, draw in sorted(REGIONS, key=lambda region: found[region[0]][0][0], reverse=True):
        start, end = found[name][0]
        lines[start:end] = draw().split("\n")[:-1]
    return "\n".join(lines)


def headerless(root: str = PROJECT) -> tuple[str, ...]:
    return tuple(layer for layer in LAYERS if not any(path.endswith(".hxx") for path in sources(root, layer)))


def declares_and_defines(root: str, low: str, high: str, name: str) -> bool:
    def names(layer: str, suffix: str) -> bool:
        for path in sources(root, layer):
            if path.endswith(suffix):
                with open(path) as f:
                    if name in f.read():
                        return True
        return False

    return names(low, ".hxx") and names(high, ".cxx")
