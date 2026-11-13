from __future__ import annotations

import os
import re
from typing import Callable, Iterator

from .rules import AMREX_CALLBACKS, DECLARATION_START, EXTERN_C, FLESH_SYNC_OVERLOAD, GUARD_PREFIX, HIERARCHY_LAYERS, LAYER_INCLUDE, LAYERS, OWNS, PRIVATE_DIR, PROTOTYPE_END, PUBLIC_DIR, READS, READS_HOME, READS_TEXT, RULES, STEMS, STEMS_HOME, UNDER_HIERARCHY, declaration_head, declared_function, file_local, layer_of, layer_path, statements
from comments.lex_cxx import lex
from comments.tokens import Kind
from files import PROJECT

MAP = "agent_docs/carpetx-internals.md"
HUB = HIERARCHY_LAYERS[0]
ABOVE = HIERARCHY_LAYERS[1:]
PUBLIC = "include"
DEFINITION = r"(?m)^[^;{{}}()\n]*\b(?:[\w:]+::)?{name}(?:::~?\w+)?\s*\([^;{{}}]*\)[^;{{}}]*\{{"
OUTSIDERS = (("cactus", "Cactus flesh"), ("amrex", "AMReX"), ("thorn", "a thorn's routine"))
STARTUP = layer_path("flesh", "startup.cxx")
REGISTRATION = re.compile(r"\bCCTK_(?:Overload\w+|RegisterGHExtension\w+|InterpRegisterOpLocalUniform)\s*\(\s*(?:ghext_handle\s*,\s*)?(\w+)")
REGISTERED = (
    (STARTUP, ("SetupGH", "InitGH", "ScheduleTraverseGH", "Initialise_entry", "Evolve_entry", "MyProc", "nProcs", "Exit", "Abort", "Barrier", "EnableGroupStorage_entry", "DisableGroupStorage_entry", "GroupStorageIncrease_entry", "GroupStorageDecrease_entry", "CarpetX_InterpGridArrays", "ArrayGroupSizeB", "GroupDynamicData")),
    (layer_path("schedule", "evolve.cxx"), ("Shutdown",)),
    (layer_path("schedule", "storage.cxx"), ("QueryGroupStorageB",)),
    (layer_path("io", "io.cxx"), ("OutputGH",)),
    (layer_path("ops", "interpolate.cxx"), ("InterpLocalUniform",)),
    (layer_path("comm", "sync_restrict.cxx"), (FLESH_SYNC_OVERLOAD,)),
)
CALLBACKS = (layer_path("regrid", "regrid.cxx"), AMREX_CALLBACKS)
PUBLIC_HEADERS = (PUBLIC_DIR + "carpetx.hxx", PUBLIC_DIR + "carpetx_rk.hxx")
PUBLIC_INCLUDE = re.compile(r'^\s*#\s*include\s*[<"](' + "|".join(re.escape(os.path.basename(header)) for header in PUBLIC_HEADERS) + r')[>"]')
PUBLIC_LAYER = "api"
PUBLIC_BODIES = (
    (layer_path("hierarchy", "handles.cxx"), ("variable", "active_window", "all_levels")),
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
    ("1", "cactus", "flesh", ("Initialise_entry", "Evolve_entry"), STARTUP),
    ("1", "flesh", "schedule", ("Initialise", "Evolve"), layer_path("schedule", "evolve.cxx")),
    ("1", "cactus", "schedule", ("Shutdown",), layer_path("schedule", "evolve.cxx")),
    ("2", "schedule", "cactus", ("CCTK_Traverse", "CCTK_OutputGH"), layer_path("schedule", "evolve.cxx")),
    ("3", "cactus", "flesh", ("ScheduleTraverseGH",), STARTUP),
    ("4", "cactus", "io", ("OutputGH",), layer_path("io", "io.cxx")),
    ("5", "flesh", "cactus", ("CCTK_ScheduleTraverse",), STARTUP),
    ("6", "cactus", "schedule", ("CallFunction",), layer_path("schedule", "callfunction.cxx")),
    ("7", "schedule", "thorn", ("CCTK_CallFunction",), layer_path("schedule", "callfunction.cxx")),
    ("9", "api", "cactus", ("CCTK_ScheduleTraverse",), layer_path("api", "api_rk.cxx")),
    (HANDED, "cactus", "flesh", ("EnableGroupStorage_entry", "DisableGroupStorage_entry", "GroupStorageIncrease_entry", "GroupStorageDecrease_entry"), STARTUP),
    (HANDED, "flesh", "schedule", ("EnableGroupStorage", "DisableGroupStorage", "GroupStorageIncrease", "GroupStorageDecrease"), layer_path("schedule", "storage.cxx")),
)
assert {row[0] for row in PASS} | {BODIES, HANDED} == set(STEPS), PASS
REGRID_STEPS = {
    "1": "the driver asks the hierarchy",
    "2": "the hierarchy asks AMReX",
    "3": "AMReX calls back into the layer that defines its callbacks",
}
REGRID = (
    ("1", "schedule", "hierarchy", ("structure_writer_t::regrid",), layer_path("hierarchy", "hierarchy.cxx")),
    ("2", "hierarchy", "amrex", ("amrcore->regrid",), layer_path("hierarchy", "hierarchy.cxx")),
    ("3", "amrex", layer_of(CALLBACKS[0]), CALLBACKS[1], CALLBACKS[0]),
)
assert {row[0] for row in REGRID} == set(REGRID_STEPS), REGRID
CALLED_AS = {"structure_writer_t::regrid": "structure.regrid"}
assert set(CALLED_AS) <= {name for row in PASS + REGRID for name in row[3]}, CALLED_AS
BOXES = (("above", f"above {HUB}/"), ("door", "driver() allowed"), ("nodoor", "no driver()"))
STACK_LEGEND = (
    f"the layer rule holds what each layer reads, the table READS in {READS_HOME}: a file includes only its own layer or one its layer reads, and {READS_TEXT}",
    "a layer reads only layers before it in that list, an include outside the table is refused, and an entry no file of the layer includes fails the tests, so the table holds exactly the layer-to-layer includes the tree has today",
    "a solid arrow is an include a layer has today, read from its #include lines, with the headers on it, and a layer is drawn above what it includes",
    f"an arrow from a box stands for the layers in it: its label is every header one of them reads from the target, and which layer reads which is not drawn; the inner box, every layer above {HUB}/, is there for the arrow into {HUB}/",
    "the outer boxes are the door rule: driver() is called in the upper box, never in the lower",
    f"{PUBLIC}/ is the public headers, {', '.join(os.path.basename(header) for header in PUBLIC_HEADERS)}, not a layer: a thorn reads nothing else of CarpetX, and agent_docs/carpetx-public-api.md describes them",
    "a node names its layer, and no header where nothing can include it",
)
PASS_LEGEND = (
    "the numbered arrows are one pass of the schedule: " + ", ".join(f"{what} ({step})" for step, what in STEPS.items()),
    "a layer reaches a higher one only through an outsider, the Cactus flesh or AMReX",
)
REGRID_LEGEND = (
    "the numbered arrows are one regrid: " + ", ".join(f"{what} ({step})" for step, what in REGRID_STEPS.items()),
    "the table below lists every entry an outsider calls",
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


def layer_includes(root: str, layer: str, suffixes: tuple[str, ...] = (".cxx", ".hxx")) -> set[tuple[str, str]]:
    found: set[tuple[str, str]] = set()
    for path in sources(root, layer):
        if not path.endswith(suffixes):
            continue
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
            assert target in READS[layer], (layer, target)
            found.setdefault((layer, target), set()).add(header)
        for header in public_includes(root, layer):
            found.setdefault((layer, PUBLIC), set()).add(header)
    return found


def header_reads(root: str = PROJECT) -> dict[str, tuple[str, ...]]:
    found = {layer: {target for target, _ in layer_includes(root, layer, (".hxx",))} for layer in LAYERS}
    return {layer: tuple(target for target in LAYERS if target in found[layer]) for layer in LAYERS}


def unread(root: str = PROJECT) -> list[tuple[str, str]]:
    found = reads(root)
    return [(layer, target) for layer in LAYERS for target in READS[layer] if (layer, target) not in found]


NAME = re.compile(r"[A-Za-z_]\w*")
LITERAL = re.compile(r"(?:u8|[uUL])?R?[\"']")
OPERATOR = "operator"
OPERATOR_SIGN = re.compile(r"[^\w\s(\"']")
OPENERS = {"(": ")", "[": "]", "{": "}"}
CLOSERS = frozenset(OPENERS.values())
CLASS_KEYS = ("struct", "class", "union", "enum")
SCOPED_ENUM = ("class", "struct")
CALLED_SPECIFIERS = ("decltype", "alignas", "noexcept", "explicit", "requires", "__attribute__")
NOT_A_NAME = frozenset(
    "auto bool char char8_t char16_t char32_t class const consteval constexpr constinit decltype double enum explicit extern float friend "
    "inline int long mutable namespace noexcept operator requires short signed static struct template thread_local typedef typename union "
    "unsigned using virtual void volatile wchar_t".split()
)


def stems(root: str, layer: str) -> set[str]:
    return {os.path.splitext(os.path.basename(path))[0] for path in sources(root, layer)}


def stem_rows(root: str = PROJECT) -> dict[str, dict[str, str]]:
    for layer, rows in STEMS.items():
        found = stems(root, layer)
        unlisted = sorted(found - set(rows))
        if unlisted:
            raise ValueError(f"{', '.join(unlisted)} in {PRIVATE_DIR}{layer}/ but in no row of STEMS[{layer!r}] in {STEMS_HOME}: add a row that says the one concern the stem holds")
        stale = [stem for stem in rows if stem not in found]
        if stale:
            raise ValueError(f"{', '.join(stale)} in a row of STEMS[{layer!r}] in {STEMS_HOME} but no file of {PRIVATE_DIR}{layer}/ has that stem: remove the row")
    return STEMS


def lone_sources(root: str = PROJECT) -> list[str]:
    public = {file for file, _ in PUBLIC_BODIES}
    found = []
    for layer in STEMS:
        names = {os.path.basename(path) for path in sources(root, layer)}
        found.extend(
            layer_path(layer, name)
            for name in sorted(names)
            if name.endswith(".cxx") and os.path.splitext(name)[0] + ".hxx" not in names and layer_path(layer, name) not in public
        )
    return found


def spelled(text: str) -> list[tuple[str, int]]:
    tokens = [token for token in lex(text) if token.kind is Kind.CODE]
    found = []
    at = 0
    while at < len(tokens):
        word, line = tokens[at].text, tokens[at].line
        at += 1
        if word == OPERATOR:
            if "".join(token.text for token in tokens[at : at + 2]) in ("()", "[]"):
                word += tokens[at].text + tokens[at + 1].text
                at += 2
            while at < len(tokens) and OPERATOR_SIGN.fullmatch(tokens[at].text):
                word += tokens[at].text
                at += 1
        elif word == ":" and at < len(tokens) and tokens[at].text == ":":
            word = "::"
            at += 1
        found.append((word, line))
    return found


def code(text: str) -> str:
    return " ".join(word for word, _ in spelled(text) if not LITERAL.match(word))


def words_in(text: str) -> set[str]:
    return {token.text for token in lex(text) if token.kind in (Kind.CODE, Kind.DIRECTIVE) and NAME.fullmatch(token.text)}


def macros(text: str) -> set[str]:
    directive = [token.text for token in lex(text) if token.kind is Kind.DIRECTIVE]
    found = {name for hash_, define, name in zip(directive, directive[1:], directive[2:]) if hash_ in ("#", "%:") and define == "define"}
    return {name for name in found if not name.startswith(GUARD_PREFIX)}


def closing(words: list[str], at: int) -> int:
    depth = 0
    for end in range(at, len(words)):
        depth += words[end] in OPENERS
        depth -= words[end] in CLOSERS
        if depth == 0:
            return end
    return len(words) - 1


def angles_end(words: list[str], at: int) -> int:
    depth = 0
    end = at
    while end < len(words):
        if words[end] in OPENERS:
            end = closing(words, end)
        elif words[end] == "<":
            depth += 1
        elif words[end] == ">":
            depth -= 1
            if depth == 0:
                return end
        end += 1
    return len(words) - 1


def bare(head: list[str]) -> list[str]:
    words = []
    at = 0
    while at < len(head):
        if head[at : at + 2] == ["[", "["]:
            at = closing(head, at) + 1
        else:
            words.append(head[at])
            at += 1
    while words:
        if words[0] == "extern" and len(words) > 1 and (LITERAL.match(words[1]) or words[1] == "template"):
            words = words[1 + bool(LITERAL.match(words[1])) :]
        elif words[0] == "template":
            words = words[angles_end(words, 1) + 1 :] if words[1:2] == ["<"] else words[1:]
        else:
            break
    return words


def declarator_end(words: list[str]) -> int:
    angles = 0
    at = 0
    while at < len(words):
        word = words[at]
        if word == "<":
            angles += 1
        elif word == ">" and angles:
            angles -= 1
        elif word in OPENERS:
            if angles == 0 and word != "{" and not (word == "(" and at and words[at - 1] in CALLED_SPECIFIERS):
                return at
            at = closing(words, at)
        elif word == "=" and angles == 0:
            return at
        at += 1
    return len(words)


def scope_opening(head: list[str]) -> bool:
    words = [word for word in head if word != "inline"]
    return (words[:1] == ["namespace"] and "=" not in words) or (len(words) == 2 and words[0] == "extern" and bool(LITERAL.match(words[1])))


def function_like(head: list[str]) -> bool:
    words = bare(head)
    if not words or words[0] in CLASS_KEYS + ("namespace", "using", "typedef", "static_assert"):
        return False
    return words[declarator_end(words) :][:1] == ["("]


def declarations(text: str) -> Iterator[tuple[int, list[str], list[str]]]:
    tokens = spelled(text)
    head: list[str] = []
    body: list[str] = []
    nest: list[str] = []
    blocks = 0
    line = 0
    for at, (word, number) in enumerate(tokens):
        if nest:
            in_block = nest[0] == "{"
            if word in OPENERS:
                nest.append(word)
            elif word in CLOSERS:
                nest.pop()
            if not in_block:
                if blocks == 0:
                    head.append(word)
            elif nest:
                if blocks == 0:
                    body.append(word)
            else:
                blocks += 1
                following = tokens[at + 1][0] if at + 1 < len(tokens) else ""
                if following not in (";", ",", "{") and function_like(head):
                    yield line, head, body
                    head, body, blocks = [], [], 0
            continue
        if word == "}":
            head, body, blocks = [], [], 0
        elif word == ";":
            if head:
                yield line, head, body
            head, body, blocks = [], [], 0
        elif word == "{" and blocks == 0 and scope_opening(head):
            head = []
        elif word in CLOSERS:
            continue
        else:
            if word in OPENERS:
                nest.append(word)
            if word != "{" and blocks == 0:
                if not head:
                    line = number
                head.append(word)


def outermost(words: list[str], at: int) -> str:
    first = at
    while first >= 2 and words[first - 1] == "::":
        first -= 2
        if words[first] == ">":
            depth = 0
            while first >= 0:
                depth += words[first] == ">"
                depth -= words[first] == "<"
                first -= 1
                if depth == 0:
                    break
    return words[first] if first >= 0 and NAME.fullmatch(words[first]) and words[first] not in NOT_A_NAME else words[at]


def enumerators(body: list[str]) -> set[str]:
    found = set()
    at = 0
    first = True
    while at < len(body):
        if body[at] in OPENERS:
            at = closing(body, at)
        elif body[at] == ",":
            first = True
        elif first:
            first = False
            if NAME.fullmatch(body[at]):
                found.add(body[at])
        at += 1
    return found


def declared_by(head: list[str], body: list[str]) -> set[str] | None:
    words = bare(head)
    if not words or words[0] == "static_assert":
        return set()
    if words[0] in CLASS_KEYS:
        scoped = words[0] == "enum" and words[1:2] and words[1] in SCOPED_ENUM
        rest = words[1 + bool(scoped) :]
        found = {rest[0]} if rest and NAME.fullmatch(rest[0]) and rest[0] != "final" else set()
        if words[0] == "enum" and not scoped:
            found |= enumerators(body)
        return found or None
    if words[0] == "using":
        if words[1:2] in (["namespace"], ["enum"]):
            return set()
        name = words[1] if "=" in words else words[-1]
        return {name} if NAME.fullmatch(name) or name.startswith(OPERATOR) else None
    if words[0] == "namespace":
        return {words[1]} if words[2:3] == ["="] and NAME.fullmatch(words[1]) else None
    declarator = words[1 : declarator_end(words)] if words[0] == "typedef" else words[: declarator_end(words)]
    for at, word in enumerate(declarator):
        if word.startswith(OPERATOR) and (word == OPERATOR or not NAME.fullmatch(word)):
            name = outermost(declarator, at)
            return {name} if name != OPERATOR else None
    if not declarator or not NAME.fullmatch(declarator[-1]) or declarator[-1] in NOT_A_NAME:
        return None
    return {outermost(declarator, len(declarator) - 1)}


def declared_names(text: str) -> set[str]:
    found = macros(text)
    for line, head, body in declarations(text):
        names = declared_by(head, body)
        if names is None:
            raise ValueError(f"line {line} declares something the include-use check cannot name: {' '.join(head[:12])}")
        found |= names
    return found


def stem_headers(root: str = PROJECT) -> dict[str, str]:
    return {layer_path(layer, os.path.basename(path)): read(path) for layer in STEMS for path in sources(root, layer) if path.endswith(".hxx")}


def unused_includes(root: str = PROJECT) -> list[tuple[str, str]]:
    declared: dict[str, set[str]] = {}
    for header, text in stem_headers(root).items():
        try:
            declared[header] = declared_names(text)
        except ValueError as e:
            raise ValueError(f"{header}: {e}") from None
    found = []
    for layer in LAYERS:
        for path in sources(root, layer):
            includer = layer_path(layer, os.path.basename(path))
            text = read(path)
            included = dict.fromkeys(PRIVATE_DIR + (m.group(1) or m.group(2)) for m in map(LAYER_INCLUDE.match, text.splitlines()) if m)
            headers = [header for header in included if header in declared and header != includer]
            if headers:
                used = words_in(text)
                found.extend((includer, header) for header in headers if declared[header].isdisjoint(used))
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
            missing = [call for call in (CALLED_AS.get(name, name) for name in names) if not layer_calls(root, source, call)]
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
OWNS_INTRO = (
    "New code goes in the layer that owns what it works on, and a lower layer never decides for a higher one. Each row says what its layer owns and what it never decides, then names the rules whose home is in it. "
    f"Under a layer that the table STEMS in {STEMS_HOME} lists is one row per file stem, a header and its source sharing the stem, that says the one concern the stem holds: "
    "a stem with no row or a row with no stem fails the generator, `scripts/lint symbols` refuses a definition there that the header of its stem does not name, "
    "and the tests refuse a source there with no header and an include of one of its headers that uses no name the header declares."
)


def owns(root: str = PROJECT) -> str:
    assert tuple(OWNS) == LAYERS, tuple(OWNS)
    listed = stem_rows(root)
    rows = []
    for layer in reversed(LAYERS):
        rows.append(f"- `{layer}/`: {OWNS[layer]}" + (f" Rules at home here: {', '.join(rules_of(layer))}." if rules_of(layer) else ""))
        rows.extend(f"  - `{stem}`: {holds}" for stem, holds in listed.get(layer, {}).items())
    return "\n".join([OWNS_HEADING, "", OWNS_INTRO, ""] + rows) + "\n"


TITLE = "# CarpetX's internals"
PREAMBLE = (
    "`python3 scripts/lint doc --write` writes this file from `scripts/lint/boundary/rules.py`, `scripts/lint/boundary/doc.py`, "
    f"the `#include` lines of `CarpetX/src/` and the rows of doc.py it holds to the code, each a registration in `{STARTUP}`, a call or a body; `scripts/lint/boundary/test_rules.py` fails when "
    "the file differs from what it writes, and the edit hook refuses an edit to it. A client thorn sees none of this: it reaches CarpetX "
    "only through the headers `agent_docs/carpetx-public-api.md` describes. `python3 scripts/lint check --help` prints what every rule "
    "named below holds. The list under each picture is its key."
)
SECTIONS: tuple[tuple[str, Callable[[str], str], tuple[str, ...]], ...] = (
    ("The stack", stack, STACK_LEGEND),
    ("One pass of the schedule", the_pass, PASS_LEGEND),
    ("One regrid", the_regrid, REGRID_LEGEND),
)


def legend(texts: tuple[str, ...]) -> str:
    return "\n".join(f"- {text}" for text in texts) + "\n"


def render(root: str = PROJECT) -> str:
    parts = [TITLE, "", PREAMBLE, ""]
    for title, picture, texts in SECTIONS:
        parts.extend([f"## {title}", "", "```mermaid", picture(root).rstrip("\n"), "```", "", legend(texts)])
    parts.append(entries(root))
    parts.append(owns(root))
    return "\n".join(parts)
