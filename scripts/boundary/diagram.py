from __future__ import annotations

import os

from rules import HIERARCHY_LAYERS, LAYERS, UNDER_HIERARCHY, layer_of

MAP = "agent_docs/carpetx-internals.md"
MARK = "%% python3 scripts/boundary diagram --write rewrites this block"
NO_HEADER_LAYERS = ("regrid", "flesh")
LINK_TIME = (
    ("hierarchy", "regrid", ("CactusAmrCore",)),
    ("hierarchy", "comm", ("apply_boundary_conditions", "get_interpolator")),
)


def node(layer: str) -> str:
    label = f"{layer}/" + (" no header" if layer in NO_HEADER_LAYERS else "")
    return f'    {layer}["{label}"]'


def diagram() -> str:
    lines = [MARK, "flowchart TB", '  subgraph door["driver() allowed"]']
    lines.extend(node(layer) for layer in reversed(HIERARCHY_LAYERS))
    lines.extend(["  end", '  subgraph nodoor["no driver()"]'])
    lines.extend(node(layer) for layer in reversed(UNDER_HIERARCHY))
    lines.extend(["  end", "  " + " --> ".join(reversed(LAYERS))])
    for low, high, names in LINK_TIME:
        assert LAYERS.index(low) < LAYERS.index(high), (low, high)
        lines.append(f'  {low} -. "{", ".join(names)}" .-> {high}')
    return "\n".join(lines) + "\n"


def blocks(text: str) -> list[tuple[int, int]]:
    lines = text.split("\n")
    found = []
    start = None
    for i, line in enumerate(lines):
        if start is None and line.strip() == "```mermaid":
            start = i + 1
        elif start is not None and line.strip() == "```":
            if lines[start] == MARK:
                found.append((start, i))
            start = None
    return found


def rewrite(text: str) -> str:
    found = blocks(text)
    if len(found) != 1:
        raise ValueError(f"{MAP} holds {len(found)} mermaid blocks that open with {MARK!r}, not one")
    start, end = found[0]
    lines = text.split("\n")
    return "\n".join(lines[:start] + diagram().split("\n")[:-1] + lines[end:])


def headerless(root: str) -> tuple[str, ...]:
    src = os.path.join(root, "CarpetX", "src")
    return tuple(layer for layer in LAYERS if not any(f.endswith(".hxx") for f in os.listdir(os.path.join(src, layer))))


def declares_and_defines(root: str, low: str, high: str, name: str) -> bool:
    src = os.path.join(root, "CarpetX", "src")

    def names(layer: str, suffix: str) -> bool:
        for f in os.listdir(os.path.join(src, layer)):
            if f.endswith(suffix):
                with open(os.path.join(src, layer, f)) as h:
                    if name in h.read():
                        return True
        return False

    assert layer_of(f"CarpetX/src/{low}/x.hxx") == low and layer_of(f"CarpetX/src/{high}/x.cxx") == high
    return names(low, ".hxx") and names(high, ".cxx")
