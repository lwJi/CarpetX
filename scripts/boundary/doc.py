from __future__ import annotations

import os
import re
from typing import Callable, Optional

from rules import HIERARCHY_LAYERS, LAYER_INCLUDE, LAYERS, RULES, UNDER_HIERARCHY, below
from files import PROJECT  # noqa: E402

MAP = "agent_docs/carpetx-internals.md"
WRITES = "python3 scripts/boundary doc --write rewrites this"
LINK_TIME = (
    ("hierarchy", "regrid", ("CactusAmrCore",)),
    ("hierarchy", "comm", ("apply_boundary_conditions", "get_interpolator")),
)
BASE_LAYERS = below("hierarchy")
DEFINITION = r"\b(?:[\w:]+::)?{name}(?:::~?\w+)?\s*\([^;{{}}]*\)[^;{{}}]*\{{"


def mark(name: str) -> str:
    return f"%% {WRITES} block: {name}"


def rules_of(layer: str) -> list[str]:
    return [rule.name for rule in RULES if layer in rule.home]


def sources(root: str, layer: str) -> list[str]:
    folder = os.path.join(root, "CarpetX", "src", layer)
    return [os.path.join(folder, f) for f in sorted(os.listdir(folder)) if f.endswith((".cxx", ".hxx"))]


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
    lines = [mark("the stack"), "flowchart TB", '  subgraph door["driver() allowed"]']
    lines.extend(node(layer, no_header, "    ", rules=True) for layer in reversed(HIERARCHY_LAYERS))
    lines.extend(["  end", '  subgraph nodoor["no driver()"]'])
    lines.extend(node(layer, no_header, "    ", rules=True) for layer in reversed(UNDER_HIERARCHY))
    lines.extend(["  end", "  " + " --> ".join(reversed(LAYERS))])
    for low, high, names in LINK_TIME:
        assert LAYERS.index(low) < LAYERS.index(high), (low, high)
        lines.append(f'  {low} -. "{", ".join(names)}" .-> {high}')
    return "\n".join(lines) + "\n"


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
    lines = [mark("the includes"), "flowchart TB"]
    lines.extend(node(layer, no_header) for layer in reversed(LAYERS) if layer not in BASE_LAYERS)
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


def declares_and_defines(root: str, low: str, high: str, name: str) -> bool:
    def found(layer: str, suffix: str, shape: re.Pattern) -> bool:
        for path in sources(root, layer):
            if path.endswith(suffix):
                with open(path) as f:
                    if shape.search(f.read()):
                        return True
        return False

    return found(low, ".hxx", re.compile(rf"\b{name}\b")) and found(high, ".cxx", re.compile(DEFINITION.format(name=name)))
