from __future__ import annotations

import dataclasses
import os
import re
from typing import Callable, Optional

from .rules import AMREX_CALLBACKS, DECLARATION_START, EXTERN_C, FLESH_SYNC_OVERLOAD, HIERARCHY_LAYERS, LAYER_INCLUDE, LAYERS, OWNS, PROTOTYPE_END, PUBLIC_DIR, RULES, TRAVERSAL_FILES, TRAVERSAL_WORD, UNDER_HIERARCHY, below, declaration_head, declared_function, file_local, layer_of, layer_path, statements
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
HUB = HIERARCHY_LAYERS[0]
ABOVE = HIERARCHY_LAYERS[1:]
PUBLIC = "include"
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
PUBLIC_INCLUDE = re.compile(r'^\s*#\s*include\s*[<"](' + "|".join(re.escape(os.path.basename(header)) for header in PUBLIC_HEADERS) + r')[>"]')
PUBLIC_LAYER = "api"
PUBLIC_BODIES = (
    (layer_path("hierarchy", "loops.cxx"), ("synchronize",)),
    (layer_path("hierarchy", "valid.cxx"), ("valid_t",)),
    (layer_path("hierarchy", "group_tags.cxx"), ("operator<<",)),
    (layer_path("ops", "reduction.cxx"), ("reduce",)),
    (layer_path("util", "timer.cxx"), ("Timer",)),
)
BODIES = "8"
HANDED = "10"
STEPS = {
    "1": "the flesh starts the driver",
    "2": "the driver names a bin or asks for output",
    "3": "the flesh hands a bin to the driver's entry",
    "4": "the flesh hands output to io/",
    "5": "the entry asks the flesh to walk the bin with CallFunction",
    "6": "the flesh calls CallFunction for each routine",
    "7": "CallFunction runs the routine",
    BODIES: f"the routine may call the public headers, whose bodies are in {PUBLIC_LAYER}/ but for those the table names",
    "9": f"{PUBLIC_LAYER}/ may ask the flesh to walk a schedule group, back to 6",
    HANDED: "what a routine asks of the flesh, the flesh hands to the layer that owns the work",
}
PASS = (
    ("1", "cactus", "schedule", ("Initialise", "Evolve", "Shutdown"), layer_path("schedule", "evolve.cxx")),
    ("2", "schedule", "cactus", ("CCTK_Traverse", "CCTK_OutputGH"), layer_path("schedule", "evolve.cxx")),
    ("3", "cactus", "flesh", ("ScheduleTraverseGH",), STARTUP),
    ("4", "cactus", "io", ("OutputGH",), layer_path("io", "io.cxx")),
    ("5", "flesh", "cactus", ("CCTK_ScheduleTraverse",), STARTUP),
    ("6", "cactus", "schedule", ("CallFunction",), layer_path("schedule", "callfunction.cxx")),
    ("7", "schedule", "thorn", ("CCTK_CallFunction",), layer_path("schedule", "callfunction.cxx")),
    ("9", "api", "cactus", ("CCTK_ScheduleTraverse",), layer_path("api", "api_rk.cxx")),
)
assert {row[0] for row in PASS} | {BODIES, HANDED} == set(STEPS), PASS
REGRID_STEPS = {
    "1": "the driver asks the hierarchy",
    "2": "the hierarchy asks AMReX",
    "3": "AMReX calls back into the layer that defines its callbacks",
}
REGRID = (
    ("1", "schedule", "hierarchy", ("hierarchy_writer::regrid",), layer_path("hierarchy", "hierarchy.cxx")),
    ("2", "hierarchy", "amrex", ("amrcore->regrid",), layer_path("hierarchy", "hierarchy.cxx")),
    ("3", "amrex", layer_of(CALLBACKS[0]), CALLBACKS[1], CALLBACKS[0]),
)
assert {row[0] for row in REGRID} == set(REGRID_STEPS), REGRID

PHASE_FILE, PHASE_HEADER = TRAVERSAL_FILES
PHASE_LAYER = layer_of(PHASE_FILE)
assert PHASE_LAYER is not None, PHASE_FILE
CLASS = re.compile(r"^class (\w+) \{$", re.MULTILINE)
REFUSAL = re.compile(r"\brefuse\s*\(")
DEFINITION_OF = r"(?m)^[^;{{}}()\n]*\b(?:[\w:]+::)?(?<!~){name}\s*\([^;{{}}]*\)[^;{{}}]*\{{"
DESTRUCTOR_OF = r"::{name}\s*\(\s*\)[^;{{}}]*\{{"
CONSTRUCTED = r"(?<!\w){name}\s+\w+\s*[;({{]"
ASSIGNED = r"\b{name}\b\s*(?:=(?!=)|\.reset\s*\()"


@dataclasses.dataclass(frozen=True)
class Slot:
    region: str
    names: tuple[str, ...]
    states: tuple[str, ...]
    arrows: tuple[tuple[str, str, str], ...]
    reader: str

    def writers(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(function for _, _, function in self.arrows if not function.startswith("~")))

    def functions(self) -> tuple[str, ...]:
        return self.writers() + (self.reader,)


PHASE = (
    Slot("window", ("active_levels",), ("closed", "open on a window"), (("closed", "open on a window", "traversal_scope_t"), ("open on a window", "closed", "~traversal_scope_t")), "scheduled_window"),
    Slot("bin", ("tail_bin",), ("none", "a tail bin"), (("none", "a tail bin", "tail_bin_scope_t"), ("a tail bin", "none", "~tail_bin_scope_t")), "scheduled_tail_bin"),
    Slot("recovery", ("recovery_traversal",), ("off", "on"), (("off", "on", "recovery_scope_t"), ("on", "off", "~recovery_scope_t")), "in_recovery_traversal"),
    Slot("routine", ("last_routine",), ("none", "the last routine run"), (("none", "the last routine run", "routine_scope_t"), ("the last routine run", "the last routine run", "~routine_scope_t")), "scheduled_routine"),
    Slot("iteration", ("recover_iteration",), ("unknown", "recorded"), (("unknown", "recorded", "record_recovered_iteration"),), "recovered_iteration"),
    Slot(
        "integrator",
        ("integrator_config", "integrator_frozen"),
        ("none", "registered", "frozen with none", "frozen"),
        (("none", "registered", "record_integrator"), ("none", "frozen with none", "freeze_integrator"), ("registered", "frozen", "freeze_integrator")),
        "integrator",
    ),
)
REFUSED = {
    "traversal_scope_t": "inside an open traversal",
    "tail_bin_scope_t": "inside a tail bin",
    "recovery_scope_t": "inside a recovery traversal",
    "record_recovered_iteration": "a second time",
    "record_integrator": "a second time, or after the freeze",
    "freeze_integrator": "a second time",
    "scheduled_window": "when closed",
    "integrator": "until frozen",
}
SLOTS = tuple(name for slot in PHASE for name in slot.names)
assert sorted(SLOTS) == sorted(TRAVERSAL_WORD.pattern[3:-3].split("|")), SLOTS
PHASE_FUNCTIONS = tuple(dict.fromkeys(function for slot in PHASE for function in slot.functions()))
assert set(REFUSED) <= set(PHASE_FUNCTIONS), REFUSED
for slot in PHASE:
    assert len(set(slot.states)) == len(slot.states), slot
    for source, target, _ in slot.arrows:
        assert source in slot.states and target in slot.states, (slot.region, source, target)
STAYS = tuple(f"{function} keeps {' and '.join(slot.names)}" for slot in PHASE for source, target, function in slot.arrows if source == target)
BOXES = (("above", f"above {HUB}/"), ("door", "driver() allowed"), ("nodoor", "no driver()"))
STACK_LEGEND = (
    f"the layer rule holds the order {', '.join(f'{layer}/' for layer in LAYERS)} from the bottom: a file includes only its own layer or a lower one",
    "a solid arrow is an include a layer has today, read from its #include lines, with the headers on it, and a layer is drawn above what it includes",
    f"an arrow from a box stands for the layers in it: its label is every header one of them reads from the target, and which layer reads which is not drawn; the inner box, every layer above {HUB}/, is there for the arrow into {HUB}/",
    "the outer boxes are the door rule: driver() is called in the upper box, never in the lower",
    f"{PUBLIC}/ is the public headers, {', '.join(os.path.basename(header) for header in PUBLIC_HEADERS)}, not a layer: a thorn reads nothing else of CarpetX, and agent_docs/carpetx-public-api.md describes them",
    "a dashed line is an import the build makes upward, a row of ACCEPTED in scripts/lint/boundary/symbols.py that the symbols check holds to the objects: the lower end declares, the upper end holds the body, the one way a call reaches up, at link time, and a class stands for its members",
    "a node names its layer, and no header where nothing can include it",
)
PASS_LEGEND = (
    "the numbered arrows are one pass of the schedule: " + ", ".join(f"{what} ({step})" for step, what in STEPS.items()),
    "a layer reaches a higher one only through an outsider, the Cactus flesh or AMReX, or at link time through a dashed line of the stack",
)
REGRID_LEGEND = (
    "the numbered arrows are one regrid: " + ", ".join(f"{what} ({step})" for step, what in REGRID_STEPS.items()),
    "the table below lists every entry an outsider calls",
)
PHASE_LEGEND = (
    f"each row is a slot of {PHASE_LAYER}/, the run's phase, and moves on its own: its left state is the value at startup, and an arrow is the one function of {os.path.basename(PHASE_HEADER)} that moves it, read from the assignment in its body in {os.path.basename(PHASE_FILE)}",
    "a row's title names the layers whose calls move it, read from their sources; an arrow named with ~ is a scope's destructor, which runs when the scope ends",
    f"a step that stays in one state is not drawn: {', '.join(STAYS)}",
    f"the table below lists every function of {os.path.basename(PHASE_HEADER)} a layer calls, who calls it, and where a refuse in its body stops it",
)


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


def public_includes(root: str, layer: str) -> set[str]:
    found: set[str] = set()
    for path in sources(root, layer):
        with open(path) as f:
            for line in f:
                m = PUBLIC_INCLUDE.match(line)
                if m:
                    found.add(m.group(1))
    return found


def group_of(layer: str) -> str | None:
    return "above" if layer in ABOVE else "nodoor" if layer in UNDER_HIERARCHY else None


def box_of(reader: str, target: str) -> str:
    if reader in UNDER_HIERARCHY:
        return "nodoor"
    return "above" if target == HUB else "door"


def reads(root: str) -> dict[tuple[str, str], set[str]]:
    found: dict[tuple[str, str], set[str]] = {}
    for layer in LAYERS:
        for target, header in layer_includes(root, layer):
            assert LAYERS.index(target) < LAYERS.index(layer), (layer, target)
            found.setdefault((layer, target), set()).add(header)
        for header in public_includes(root, layer):
            found.setdefault((layer, PUBLIC), set()).add(header)
    return found


def is_pair(reader: str, target: str) -> bool:
    return target != PUBLIC and group_of(reader) is not None and group_of(reader) == group_of(target)


def top_down_order(layer: str) -> int:
    return -LAYERS.index(layer)


def includes(root: str = PROJECT) -> dict[tuple[str, str], tuple[str, ...]]:
    edges = {edge: headers for edge, headers in reads(root).items() if is_pair(*edge)}
    return {edge: tuple(sorted(edges[edge])) for edge in sorted(edges, key=lambda e: (top_down_order(e[0]), top_down_order(e[1])))}


def floor(root: str = PROJECT) -> dict[tuple[str, str], tuple[str, ...]]:
    edges: dict[tuple[str, str], set[str]] = {}
    for (reader, target), headers in reads(root).items():
        if not is_pair(reader, target):
            edges.setdefault((box_of(reader, target), target), set()).update(headers)
    boxes = [box for box, _ in BOXES]
    targets = list(reversed(LAYERS)) + [PUBLIC]
    return {edge: tuple(sorted(edges[edge])) for edge in sorted(edges, key=lambda e: (boxes.index(e[0]), targets.index(e[1])))}


def stack(root: str = PROJECT) -> str:
    no_header = headerless(root)
    titles = dict(BOXES)
    lines = ["flowchart TB", f'  subgraph door["{titles["door"]}"]', f'    subgraph above["{titles["above"]}"]']
    lines.extend(node(layer, no_header, "      ") for layer in reversed(ABOVE))
    lines.extend(["    end", node(HUB, no_header, "    "), "  end", f'  subgraph nodoor["{titles["nodoor"]}"]'])
    lines.extend(node(layer, no_header, "    ") for layer in reversed(UNDER_HIERARCHY))
    lines.extend(["  end", f'  {PUBLIC}["{PUBLIC}/"]'])
    for (source, target), headers in includes(root).items():
        lines.append(f'  {source} -- "{", ".join(headers)}" --> {target}')
    for (box, target), headers in floor(root).items():
        lines.append(f'  {box} -- "{", ".join(headers)}" --> {target}')
    for low, high, names in LINK_TIME:
        assert LAYERS.index(low) < LAYERS.index(high), (low, high)
        lines.append(f'  {high} -. "{", ".join(names)}" .- {low}')
    return "\n".join(lines) + "\n"


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
    return sorted(found.items(), key=lambda item: top_down_order(item[0]))


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
            raise ValueError(f"{file} is not in {target}/, where step {step} lands")
        missing = [name for name in names if not defined(text, name)]
        if missing:
            raise ValueError(f"{', '.join(missing)} not defined in {file}, where step {step} lands")
        if source in LAYERS:
            missing = [name for name in names if not layer_calls(root, source, name)]
            if missing:
                raise ValueError(f"{', '.join(missing)} not called in {source}/, where step {step} starts")
    else:
        if layer_of(file) != source:
            raise ValueError(f"{file} is not in {source}/, where step {step} starts")
        missing = [name for name in names if not called(text, name)]
        if missing:
            raise ValueError(f"{', '.join(missing)} not called in {file}, where step {step} starts")


def handed(root: str = PROJECT) -> list[tuple[str, tuple[str, ...]]]:
    drawn = {name for _, _, _, names, _ in PASS for name in names}
    found: dict[str, tuple[str, ...]] = {}
    for file, names in REGISTERED:
        layer = layer_of(file)
        assert layer is not None, file
        rest = tuple(name for name in names if name not in drawn)
        if layer != "flesh" and rest:
            found[layer] = found.get(layer, ()) + rest
    return top_down(found)


def draw(edges: dict[tuple[str, str, str], tuple[str, ...]]) -> str:
    drawn = {end for _, source, target in edges for end in (source, target)}
    lines = ["flowchart LR"]
    lines.extend(f'  {name}["{label}"]' for name, label in OUTSIDERS if name in drawn)
    lines.extend(node(layer, ()) for layer in reversed(LAYERS) if layer in drawn)
    lines.extend(f'  {source} -- "{step} {", ".join(names)}" --> {target}' for (step, source, target), names in edges.items())
    return "\n".join(lines) + "\n"


def the_pass(root: str = PROJECT) -> str:
    for row in PASS:
        hold(root, row)
    edges: dict[tuple[str, str, str], tuple[str, ...]] = {}
    for layer, names in top_down(public_bodies(root)):
        edges[(BODIES, "thorn", layer)] = tuple(public_headers(root, names)) if layer == PUBLIC_LAYER else names
    for layer, names in handed(root):
        edges[(HANDED, "cactus", layer)] = names
    for step, source, target, names, _ in PASS:
        edges[(step, source, target)] = edges.get((step, source, target), ()) + names
    return draw(dict(sorted(edges.items(), key=lambda item: int(item[0][0]))))


def the_regrid(root: str = PROJECT) -> str:
    for row in REGRID:
        hold(root, row)
    return draw({(step, source, target): names for step, source, target, names, _ in REGRID})


def both(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def body(text: str, name: str, file: str) -> str:
    shape = DESTRUCTOR_OF.format(name=re.escape(name)) if name.startswith("~") else DEFINITION_OF.format(name=re.escape(name))
    m = re.search(shape, text)
    if not m:
        raise ValueError(f"{name} not defined in {file}, where its row of PHASE in scripts/lint/boundary/doc.py says it is")
    depth = 0
    for i in range(m.end() - 1, len(text)):
        depth += {"{": 1, "}": -1}.get(text[i], 0)
        if depth == 0:
            return text[m.end() : i]
    raise ValueError(f"{name} opens a body in {file} that never closes")


def uses(text: str, name: str) -> bool:
    return called(text, name) or re.search(CONSTRUCTED.format(name=re.escape(name)), text) is not None


def callers(root: str, name: str) -> tuple[str, ...]:
    return tuple(layer for layer in reversed(LAYERS) if layer != PHASE_LAYER and any(uses(read(path), name) for path in sources(root, layer) if path.endswith(".cxx")))


def declared_in_phase_header(root: str) -> list[str]:
    text = read(os.path.join(root, PHASE_HEADER))
    names: dict[str, None] = dict.fromkeys(CLASS.findall(text))
    for number, line, statement in statements(text):
        head = declaration_head(statement)
        if PROTOTYPE_END.search(line) and DECLARATION_START.match(head) and not file_local(head):
            name = declared_function(head, in_header=True)
            if name is None:
                raise ValueError(f"{PHASE_HEADER}:{number} declares something the map cannot name: {line.strip()}")
            names[name] = None
    return list(names)


def phase(root: str = PROJECT) -> dict[str, tuple[str, ...]]:
    declared = declared_in_phase_header(root)
    unlisted = [name for name in declared if name not in PHASE_FUNCTIONS]
    if unlisted:
        raise ValueError(f"{', '.join(unlisted)} declared in {PHASE_HEADER} but in no row of PHASE in scripts/lint/boundary/doc.py")
    stale = [name for name in PHASE_FUNCTIONS if name not in declared]
    if stale:
        raise ValueError(f"{', '.join(stale)} in a row of PHASE in scripts/lint/boundary/doc.py but not declared in {PHASE_HEADER}")
    text = read(os.path.join(root, PHASE_FILE))
    found: dict[str, tuple[str, ...]] = {}
    for slot in PHASE:
        for _, _, function in slot.arrows:
            if not any(re.search(ASSIGNED.format(name=name), body(text, function, PHASE_FILE)) for name in slot.names):
                raise ValueError(f"{function} assigns none of {', '.join(slot.names)} in {PHASE_FILE}, where its arrow of PHASE in scripts/lint/boundary/doc.py says it moves the slot")
        for function in slot.functions():
            refuses = REFUSAL.search(body(text, function, PHASE_FILE)) is not None
            if function in REFUSED and not refuses:
                raise ValueError(f"{function} refuses nothing in {PHASE_FILE}, but its row of PHASE in scripts/lint/boundary/doc.py says it is refused {REFUSED[function]}")
            if refuses and function not in REFUSED:
                raise ValueError(f"{function} refuses in {PHASE_FILE}, but no row of REFUSED in scripts/lint/boundary/doc.py says when")
            if function not in found:
                found[function] = callers(root, function)
                if not found[function]:
                    raise ValueError(f"{function} is called in no layer above {PHASE_LAYER}/, so no row can say who calls it")
    return found


def the_phase(root: str = PROJECT) -> str:
    who = phase(root)
    lines = ["flowchart LR"]
    for slot in PHASE:
        writers = sorted({layer for function in slot.writers() for layer in who[function]}, key=top_down_order)
        lines.append(f'  subgraph {slot.region}["{" and ".join(slot.names)}, written by {both([f"{layer}/" for layer in writers])}"]')
        ids = {state: f"{slot.region}{i}" for i, state in enumerate(slot.states)}
        lines.extend(f'    {ids[state]}["{state}"]' for state in slot.states)
        lines.extend(f'    {ids[source]} -- "{function}" --> {ids[target]}' for source, target, function in slot.arrows if source != target)
        lines.append("  end")
    return "\n".join(lines) + "\n"


PHASE_HEADING = "## Who moves and reads the phase"
PHASE_INTRO = f"A layer above {PHASE_LAYER}/ moves the phase by opening a scope or recording a fact, and reads it through one function per slot. Each row is a slot, a function of {os.path.basename(PHASE_HEADER)}, the layers that call it, and when {os.path.basename(PHASE_FILE)} refuses the call."


def phase_table(root: str = PROJECT) -> str:
    who = phase(root)
    lines = [PHASE_HEADING, "", PHASE_INTRO, "", "| Slot | Function | Called from | Refused |", "| --- | --- | --- | --- |"]
    for slot in PHASE:
        lines.extend(f"| `{', '.join(slot.names)}` | {function} | {', '.join(f'{layer}/' for layer in who[function])} | {REFUSED.get(function, 'never')} |" for function in slot.functions())
    return "\n".join(lines) + "\n"


ENTRIES_HEADING = "## Who the outsiders call"
ENTRIES_INTRO = "An outsider decides when a layer works: it calls one of these entries. Each row is a layer, who calls into it, and what they call."


def entries(root: str = PROJECT) -> str:
    rows: list[tuple[str, str, str]] = [("flesh", "Cactus flesh", f'every extern "C" in {", ".join(extern_c_files(root))}')]
    rows += [(layer, "Cactus flesh", ", ".join(names)) for layer, names in registered(root).items()]
    bodies = public_bodies(root)
    rows.append((PUBLIC_LAYER, "a thorn", f"every function of {', '.join(public_headers(root, bodies[PUBLIC_LAYER]))} not named in a row below"))
    rows += [(layer, "a thorn", ", ".join(names)) for layer, names in bodies.items() if layer != PUBLIC_LAYER]
    rows.append((layer_of(CALLBACKS[0]), "AMReX", ", ".join(CALLBACKS[1])))
    rows.sort(key=lambda row: top_down_order(row[0]))
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
    f"the `#include` lines of `CarpetX/src/` and the rows of doc.py it holds to the code, each a registration in `{STARTUP}`, a call, a body or a refusal; `scripts/lint/boundary/test_rules.py` fails when "
    "the file differs from what it writes, and the edit hook refuses an edit to it. A client thorn sees none of this: it reaches CarpetX "
    "only through the headers `agent_docs/carpetx-public-api.md` describes. `python3 scripts/lint check --help` prints what every rule "
    "named below holds. The list under each picture is its key."
)
SECTIONS: tuple[tuple[str, Callable[[str], str], tuple[str, ...], Optional[Callable[[str], str]]], ...] = (
    ("The stack", stack, STACK_LEGEND, None),
    ("One pass of the schedule", the_pass, PASS_LEGEND, None),
    ("One regrid", the_regrid, REGRID_LEGEND, None),
    ("The run's phase", the_phase, PHASE_LEGEND, phase_table),
)


def legend(texts: tuple[str, ...]) -> str:
    return "\n".join(f"- {text}" for text in texts) + "\n"


def render(root: str = PROJECT) -> str:
    parts = [TITLE, "", PREAMBLE, ""]
    for title, picture, texts, table in SECTIONS:
        parts.extend([f"## {title}", "", "```mermaid", picture(root).rstrip("\n"), "```", "", legend(texts)])
        if table:
            parts.append(table(root))
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
