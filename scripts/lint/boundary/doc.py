from __future__ import annotations

import os
import re
from typing import Callable

from .rules import AMREX_CALLBACKS, DECLARATION_START, EXTERN_C, FLESH_SYNC_OVERLOAD, HIERARCHY_LAYERS, LAYER_INCLUDE, LAYERS, OWNS, PROTOTYPE_END, PUBLIC_DIR, RULES, UNDER_HIERARCHY, below, declaration_head, declared_function, file_local, layer_of, layer_path, statements
from .symbols import ACCEPTED
from files import PROJECT

MAP = "agent_docs/carpetx-internals.md"
SYMBOL_NAMESPACE = "CarpetX::"


def link_name(prefix: str) -> str:
    assert prefix.startswith(SYMBOL_NAMESPACE), prefix
    return prefix[len(SYMBOL_NAMESPACE) :].rstrip("(:")


def link_time() -> tuple[tuple[str, str, tuple[str, ...]], ...]:
    edges: dict[tuple[str, str], dict[str, None]] = {}
    for low, high, prefix in ACCEPTED:
        assert LAYERS.index(low) < LAYERS.index(high), (low, high)
        edges.setdefault((low, high), {})[link_name(prefix)] = None
    order = sorted(edges, key=lambda edge: (LAYERS.index(edge[0]), LAYERS.index(edge[1])))
    return tuple((low, high, tuple(sorted(edges[(low, high)], key=str.lower))) for low, high in order)


LINK_TIME = link_time()
BASE_LAYERS = below("hierarchy")
DEFINITION = r"(?m)^[^;{{}}()\n]*\b(?:[\w:]+::)?{name}(?:::~?\w+)?\s*\([^;{{}}]*\)[^;{{}}]*\{{"
OUTSIDERS = (("cactus", "Cactus flesh"), ("amrex", "AMReX"), ("thorn", "a thorn's routine"))
STARTUP = layer_path("flesh", "startup.cxx")
REGISTRATION = re.compile(r"\bCCTK_(?:Overload\w+|RegisterGHExtension\w+|InterpRegisterOpLocalUniform)\s*\(\s*(?:ghext_handle\s*,\s*)?(\w+)")
REGISTERED = (
    (STARTUP, ("SetupGH", "InitGH", "ScheduleTraverseGH", "MyProc", "nProcs", "Exit", "Abort", "Barrier", "CarpetX_InterpGridArrays", "ArrayGroupSizeB", "GroupDynamicData")),
    (layer_path("schedule", "evolve.cxx"), ("Initialise", "Evolve", "Shutdown")),
    (layer_path("schedule", "storage.cxx"), ("EnableGroupStorage", "DisableGroupStorage", "GroupStorageIncrease", "GroupStorageDecrease", "QueryGroupStorageB")),
    (layer_path("io", "io.cxx"), ("OutputGH",)),
    (layer_path("ops", "interpolate.cxx"), ("InterpLocalUniform",)),
    (layer_path("comm", "sync_restrict.cxx"), (FLESH_SYNC_OVERLOAD,)),
)
CALLBACKS = (layer_path("regrid", "regrid.cxx"), AMREX_CALLBACKS)
PUBLIC_HEADERS = (PUBLIC_DIR + "carpetx.hxx", PUBLIC_DIR + "carpetx_rk.hxx")
PUBLIC_LAYER = "api"
PUBLIC_BODIES = (
    (layer_path("hierarchy", "loops.cxx"), ("synchronize",)),
    (layer_path("hierarchy", "valid.cxx"), ("valid_t",)),
    (layer_path("hierarchy", "group_tags.cxx"), ("operator<<",)),
    (layer_path("ops", "reduction.cxx"), ("reduce",)),
    (layer_path("util", "timer.cxx"), ("Timer",)),
)
REGRID = "regrid"
STEPS = {
    "1": "the flesh starts the driver",
    "2": "the driver names a bin or asks for output",
    "3": "the flesh hands it to the driver's entry",
    "4": "the entry asks the flesh to walk the bin with CallFunction",
    "5": "the flesh calls CallFunction for each routine",
    "6": "CallFunction runs the routine",
    "7": f"the routine may call the public headers, whose bodies are in {PUBLIC_LAYER}/ but for those the table names",
}
REGRID_TEXT = "a regrid: the driver asks the hierarchy, the hierarchy asks AMReX, and AMReX calls back into the layer that defines its callbacks"
LOOP = (
    ("1", "cactus", "schedule", ("Initialise", "Evolve", "Shutdown"), layer_path("schedule", "evolve.cxx")),
    ("2", "schedule", "cactus", ("CCTK_Traverse", "CCTK_OutputGH"), layer_path("schedule", "evolve.cxx")),
    ("3", "cactus", "flesh", ("ScheduleTraverseGH",), STARTUP),
    ("3", "cactus", "io", ("OutputGH",), layer_path("io", "io.cxx")),
    ("4", "flesh", "cactus", ("CCTK_ScheduleTraverse",), STARTUP),
    ("4", "api", "cactus", ("CCTK_ScheduleTraverse",), layer_path("api", "api_rk.cxx")),
    ("5", "cactus", "schedule", ("CallFunction",), layer_path("schedule", "callfunction.cxx")),
    ("6", "schedule", "thorn", ("CCTK_CallFunction",), layer_path("schedule", "callfunction.cxx")),
    (REGRID, "schedule", "hierarchy", ("hierarchy_writer::regrid",), layer_path("hierarchy", "hierarchy.cxx")),
    (REGRID, "hierarchy", "amrex", ("amrcore->regrid",), layer_path("hierarchy", "hierarchy.cxx")),
    (REGRID, "amrex", layer_of(CALLBACKS[0]), CALLBACKS[1], CALLBACKS[0]),
)
assert {row[0] for row in LOOP} | {"7"} == set(STEPS) | {REGRID}, LOOP
STACK_LEGEND = (
    "a layer sits above every layer it may include, which the layer rule holds, and a solid edge is an include it has today, read from its #include lines, with the headers on it",
    "where no solid edge joins a layer to the one below it, the order is the rule's, not a coupling of the code",
    "the two boxes are the door rule: driver() is called in the upper box, never in the lower",
    "a dashed edge is an import the build makes upward, a row of ACCEPTED in scripts/lint/boundary/symbols.py that the symbols check holds to the objects: a declaration whose body is in the higher layer, the one way a call reaches up, at link time, and a class stands for its members",
    "a node names its layer, and no header where nothing can include it",
)
BASE_DIRS = f"{', '.join(f'{layer}/' for layer in BASE_LAYERS[:-1])} or {BASE_LAYERS[-1]}/"
LOOP_LEGEND = (
    "the numbered edges are one pass of the schedule: " + ", ".join(f"{what} ({step})" for step, what in STEPS.items()),
    "the unnumbered edges are " + REGRID_TEXT,
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


def defined(text: str, name: str) -> bool:
    return re.search(DEFINITION.format(name=re.escape(name)), text) is not None


def called(text: str, name: str) -> bool:
    return re.search(r"(?<!\w)" + re.escape(name) + r"\s*\(", text) is not None


def layer_calls(root: str, layer: str, name: str) -> bool:
    return any(called(read(path), name) for path in sources(root, layer) if path.endswith(".cxx"))


def registrations(root: str = PROJECT) -> list[str]:
    return list(dict.fromkeys(REGISTRATION.findall(read(os.path.join(root, STARTUP)))))


def extern_c_files(root: str = PROJECT) -> list[str]:
    return [os.path.basename(path) for path in sources(root, "flesh") if path.endswith(".cxx") and EXTERN_C.search(read(path))]


def top_down(found: dict[str, tuple[str, ...]]) -> list[tuple[str, tuple[str, ...]]]:
    return sorted(found.items(), key=lambda item: -LAYERS.index(item[0]))


def by_layer(rows: tuple[tuple[str, tuple[str, ...]], ...]) -> dict[str, tuple[str, ...]]:
    found: dict[str, tuple[str, ...]] = {}
    for file, names in rows:
        layer = layer_of(file)
        assert layer is not None, file
        found[layer] = found.get(layer, ()) + names
    return found


def registered(root: str = PROJECT) -> dict[str, tuple[str, ...]]:
    found = registrations(root)
    claimed = [name for _, names in REGISTERED for name in names]
    unlisted = [name for name in found if name not in claimed]
    if unlisted:
        raise ValueError(f"{', '.join(unlisted)} registered in {STARTUP} but in no row of REGISTERED in scripts/lint/boundary/doc.py")
    stale = [name for name in claimed if name not in found]
    if stale:
        raise ValueError(f"{', '.join(stale)} in a row of REGISTERED in scripts/lint/boundary/doc.py but not registered in {STARTUP}")
    for file, names in REGISTERED + (CALLBACKS,):
        text = read(os.path.join(root, file))
        missing = [name for name in names if not defined(text, name)]
        if missing:
            raise ValueError(f"{', '.join(missing)} not defined in {file}, where its row in scripts/lint/boundary/doc.py says it is")
    return by_layer(REGISTERED)


def prototypes(root: str = PROJECT) -> dict[str, tuple[str, ...]]:
    found: dict[str, tuple[str, ...]] = {}
    for header in PUBLIC_HEADERS:
        names: dict[str, None] = {}
        for number, line, statement in statements(read(os.path.join(root, header))):
            head = declaration_head(statement)
            if PROTOTYPE_END.search(line) and DECLARATION_START.match(head) and not file_local(head):
                name = declared_function(head, in_header=True)
                if name is None:
                    raise ValueError(f"{header}:{number} declares something the map cannot name: {line.strip()}")
                names[name] = None
        found[header] = tuple(names)
    return found


def public_bodies(root: str = PROJECT) -> dict[str, tuple[str, ...]]:
    declared = "\n".join(read(os.path.join(root, header)) for header in PUBLIC_HEADERS)
    prototyped = {name for names in prototypes(root).values() for name in names}
    for file, names in PUBLIC_BODIES:
        text = read(os.path.join(root, file))
        for name in names:
            if name not in prototyped and not re.search(rf"\b{re.escape(name)}\b", declared):
                raise ValueError(f"{name} is in a row of PUBLIC_BODIES in scripts/lint/boundary/doc.py but in no public header")
            if not defined(text, name):
                raise ValueError(f"{name} not defined in {file}, where its row of PUBLIC_BODIES in scripts/lint/boundary/doc.py says it is")
    strays = by_layer(PUBLIC_BODIES)
    above = [layer for layer in strays if LAYERS.index(layer) >= LAYERS.index(PUBLIC_LAYER)]
    if above:
        raise ValueError(f"a body of a public header in {', '.join(f'{layer}/' for layer in above)}, at or above {PUBLIC_LAYER}/")
    named = {name for names in strays.values() for name in names}
    bodies = [read(path) for path in sources(root, PUBLIC_LAYER) if path.endswith(".cxx")]
    rest = tuple(name for names in prototypes(root).values() for name in names if name not in named)
    missing = [name for name in rest if not any(defined(text, name) for text in bodies)]
    if missing:
        raise ValueError(f"{', '.join(missing)} declared in a public header but defined in no {PUBLIC_LAYER}/ file and in no row of PUBLIC_BODIES in scripts/lint/boundary/doc.py")
    return {PUBLIC_LAYER: rest, **strays}


def public_headers(root: str, names: tuple[str, ...]) -> list[str]:
    return [os.path.basename(header) for header, declared in prototypes(root).items() if any(name in declared for name in names)]


def hold(root: str, row: tuple[str, str, str, tuple[str, ...], str]) -> None:
    step, source, target, names, file = row
    text = read(os.path.join(root, file))
    if target in LAYERS:
        if layer_of(file) != target:
            raise ValueError(f"{file} is not in {target}/, where the loop's step {step} lands")
        missing = [name for name in names if not defined(text, name)]
        if missing:
            raise ValueError(f"{', '.join(missing)} not defined in {file}, where the loop's step {step} lands")
        if source in LAYERS:
            missing = [name for name in names if not layer_calls(root, source, name)]
            if missing:
                raise ValueError(f"{', '.join(missing)} not called in {source}/, where the loop's step {step} starts")
    else:
        if layer_of(file) != source:
            raise ValueError(f"{file} is not in {source}/, where the loop's step {step} starts")
        missing = [name for name in names if not called(text, name)]
        if missing:
            raise ValueError(f"{', '.join(missing)} not called in {file}, where the loop's step {step} starts")


def the_loop(root: str = PROJECT) -> str:
    for row in LOOP:
        hold(root, row)
    edges: dict[tuple[str, str, str], tuple[str, ...]] = {}
    for step, source, target, names, _ in LOOP:
        if step != REGRID:
            edges[(step, source, target)] = edges.get((step, source, target), ()) + names
    for layer, names in top_down(public_bodies(root)):
        edges[("7", "thorn", layer)] = tuple(public_headers(root, names)) if layer == PUBLIC_LAYER else names
    for step, source, target, names, _ in LOOP:
        if step == REGRID:
            edges[(step, source, target)] = edges.get((step, source, target), ()) + names
    drawn = {end for _, source, target in edges for end in (source, target)}
    lines = ["flowchart LR"]
    lines.extend(f'  {name}["{label}"]' for name, label in OUTSIDERS if name in drawn)
    lines.extend(node(layer, ()) for layer in reversed(LAYERS) if layer in drawn)
    lines.extend(f'  {source} -- "{step + " " if step != REGRID else ""}{", ".join(names)}" --> {target}' for (step, source, target), names in edges.items())
    return legend(lines, LOOP_LEGEND)


ENTRIES_HEADING = "## Who the outsiders call"
ENTRIES_INTRO = "An outsider decides when a layer works: it calls one of these entries. Each row is a layer, who calls into it, and what they call."


def entries(root: str = PROJECT) -> str:
    rows: list[tuple[str, str, str]] = [("flesh", "Cactus flesh", f'every extern "C" in {", ".join(extern_c_files(root))}')]
    rows += [(layer, "Cactus flesh", ", ".join(names)) for layer, names in registered(root).items()]
    bodies = public_bodies(root)
    rows.append((PUBLIC_LAYER, "a thorn", f"every function of {', '.join(public_headers(root, bodies[PUBLIC_LAYER]))} not named in a row below"))
    rows += [(layer, "a thorn", ", ".join(names)) for layer, names in bodies.items() if layer != PUBLIC_LAYER]
    rows.append((layer_of(CALLBACKS[0]), "AMReX", ", ".join(CALLBACKS[1])))
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
    "`python3 scripts/lint doc --write` writes this file from `scripts/lint/boundary/rules.py`, `scripts/lint/boundary/doc.py`, the `ACCEPTED` rows of `scripts/lint/boundary/symbols.py`, "
    f"the `#include` lines of `CarpetX/src/` and the rows of doc.py it holds to the code, each a registration in `{STARTUP}`, a call or a body; `scripts/lint/boundary/test_rules.py` fails when "
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
    def found(paths: list[str], shape: re.Pattern) -> bool:
        for path in paths:
            if os.path.exists(path):
                with open(path) as f:
                    if shape.search(f.read()):
                        return True
        return False

    headers = [path for layer in below(low) for path in sources(root, layer) if path.endswith(".hxx")]
    headers += [os.path.join(root, header) for header in PUBLIC_HEADERS]
    bodies = [path for path in sources(root, high) if path.endswith(".cxx")]
    return found(headers, re.compile(rf"\b{name}\b")) and found(bodies, re.compile(DEFINITION.format(name=name)))
