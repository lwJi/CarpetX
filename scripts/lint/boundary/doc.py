from __future__ import annotations

import os
import re
from typing import Callable, Iterator

from .rules import AMREX_CALLBACKS, DECLARATION_START, GIVEN, GIVEN_HOME, GIVEN_LAYER, GRANT_LAYERS, GRANTS, GRANTS_HOME, GUARD_PREFIX, HIERARCHY_LAYERS, INSIDE_SUFFIX, LAYER_INCLUDE, LAYERS, OWNS, PARAMETERS_HOME, PARAMETERS_TEXT, PRIVATE_DIR, PROTOTYPE_END, PUBLIC_DIR, READS, READS_HOME, RULES, SEES_ALL, STEMS, STEMS_HOME, UNDER_HIERARCHY, declaration_head, declared_function, file_local, friends, layer_of, layer_path, row_of, statements, ungranted
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
    (STARTUP, ("SetupGH", "InitGH", "ScheduleTraverseGH", "Initialise_entry", "Evolve_entry", "MyProc", "nProcs", "Exit", "Abort", "Barrier", "EnableGroupStorage_entry", "DisableGroupStorage_entry", "GroupStorageIncrease_entry", "GroupStorageDecrease_entry", "CarpetX_InterpGridArrays", "ArrayGroupSizeB", "GroupDynamicData", "SyncGroups_entry")),
    (layer_path("schedule", "evolve.cxx"), ("Shutdown",)),
    (layer_path("schedule", "storage.cxx"), ("QueryGroupStorageB",)),
    (layer_path("io", "io.cxx"), ("OutputGH",)),
    (layer_path("ops", "interpolate.cxx"), ("InterpLocalUniform",)),
)
CALLBACKS = (layer_path("regrid", "regrid.cxx"), AMREX_CALLBACKS)
PUBLIC_HEADERS = (PUBLIC_DIR + "carpetx.hxx", PUBLIC_DIR + "carpetx_rk.hxx")
PUBLIC_INCLUDE = re.compile(r'^\s*#\s*include\s*[<"](' + "|".join(re.escape(os.path.basename(header)) for header in PUBLIC_HEADERS) + r')[>"]')
PUBLIC_LAYER = "api"
PUBLIC_BODIES = (
    (layer_path("hierarchy", "handles.cxx"), ("variable", "active_window", "all_levels")),
    (layer_path("hierarchy", "loops.cxx"), ("synchronize",)),
    (layer_path("hierarchy", "valid.cxx"), ("valid_t",)),
    (layer_path("config", "boundaries.cxx"), ("operator<<",)),
    (layer_path("ops", "reduction.cxx"), ("reduce",)),
    (layer_path("util", "timer.cxx"), ("Timer",)),
)
BODIES = "8"
HANDED = "10"
STEPS = tuple(str(step) for step in range(1, 11))
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
    (HANDED, "cactus", "flesh", ("EnableGroupStorage_entry", "DisableGroupStorage_entry", "GroupStorageIncrease_entry", "GroupStorageDecrease_entry", "SyncGroups_entry"), STARTUP),
    (HANDED, "flesh", "schedule", ("EnableGroupStorage", "DisableGroupStorage", "GroupStorageIncrease", "GroupStorageDecrease"), layer_path("schedule", "storage.cxx")),
    (HANDED, "flesh", "schedule", ("SyncScheduled",), layer_path("schedule", "clocks.cxx")),
)
assert {row[0] for row in PASS} | {BODIES, HANDED} == set(STEPS), PASS
REGRID_STEPS = ("1", "2", "3")
REGRID = (
    ("1", "schedule", "hierarchy", ("structure_writer_t::regrid",), layer_path("hierarchy", "writers.cxx")),
    ("2", "hierarchy", "amrex", ("amrcore_->regrid",), layer_path("hierarchy", "writers.cxx")),
    ("3", "amrex", layer_of(CALLBACKS[0]), CALLBACKS[1], CALLBACKS[0]),
)
assert {row[0] for row in REGRID} == set(REGRID_STEPS), REGRID
CALLED_AS = {"structure_writer_t::regrid": "structure.regrid"}
assert set(CALLED_AS) <= {name for row in PASS + REGRID for name in row[3]}, CALLED_AS
BOXES = (("above", f"above {HUB}/"), ("door", "driver() allowed"), ("nodoor", "no driver()"))
STACK_LEGEND = (
    f"an arrow not into {GIVEN_LAYER}/ is an include a layer has today; the layer rule refuses one outside the table READS in {READS_HOME}",
    f"an arrow into {GIVEN_LAYER}/ is the interface that layer is given, the table GIVEN in {GIVEN_HOME}; the interface rule refuses an include outside it, and {SEES_ALL}, which checks the inside of {GIVEN_LAYER}/, adds nothing to an arrow",
    f"{PUBLIC}/ is the public headers, not a layer",
    f"{PARAMETERS_TEXT}; the parameters rule refuses a read outside the tables PARAMETER_READERS and DECODES in {PARAMETERS_HOME}",
)
PASS_LEGEND = ("a layer reaches a higher one only through an outsider, the Cactus flesh or AMReX",)
REGRID_LEGEND: tuple[str, ...] = ()


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
            if target != GIVEN_LAYER:
                found.setdefault((layer, target), set()).add(header)
        for header in public_includes(root, layer):
            found.setdefault((layer, PUBLIC), set()).add(header)
    for layer, stems in GIVEN.items():
        found[(layer, GIVEN_LAYER)] = {stem + ".hxx" for stem in stems}
    return found


def header_reads(root: str = PROJECT) -> dict[str, tuple[str, ...]]:
    found = {layer: {target for target, _ in layer_includes(root, layer, (".hxx",))} for layer in LAYERS}
    return {layer: tuple(target for target in LAYERS if target in found[layer]) for layer in LAYERS}


def unread(root: str = PROJECT) -> list[tuple[str, str]]:
    found = reads(root)
    return [(layer, target) for layer in LAYERS for target in READS[layer] if (layer, target) not in found]


def given_includes(root: str, layer: str) -> set[str]:
    found: set[str] = set()
    for path in sources(root, layer):
        if layer_path(layer, os.path.basename(path)) == SEES_ALL:
            continue
        for m in filter(None, map(LAYER_INCLUDE.match, read(path).splitlines())):
            target, _, header = (m.group(1) or m.group(2)).partition("/")
            if target == GIVEN_LAYER:
                found.add(header)
    return found


def unincluded(root: str = PROJECT) -> list[tuple[str, str]]:
    found = []
    for layer, stems in GIVEN.items():
        included = given_includes(root, layer)
        found.extend((layer, stem) for stem in stems if stem + ".hxx" not in included)
    return found


def unlisted_headers(root: str = PROJECT) -> list[str]:
    given = {stem for stems in GIVEN.values() for stem in stems}
    found = []
    for path in sources(root, GIVEN_LAYER):
        stem, extension = os.path.splitext(os.path.basename(path))
        if extension == ".hxx" and row_of(stem) == stem and stem not in given:
            found.append(layer_path(GIVEN_LAYER, stem + extension))
    return found


def standing(stem: str) -> str:
    assert stem in STEMS[GIVEN_LAYER], stem
    readers = [f"{layer}/" for layer, stems in GIVEN.items() if stem in stems]
    return f"Interface for {', '.join(readers)}." if readers else "Inside."


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
    return {row_of(os.path.splitext(os.path.basename(path))[0]) for path in sources(root, layer)}


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
            if name.endswith(".cxx") and names.isdisjoint(row_of(os.path.splitext(name)[0]) + suffix + ".hxx" for suffix in ("", INSIDE_SUFFIX)) and layer_path(layer, name) not in public
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


def declarations(text: str) -> Iterator[tuple[int, list[str], list[str], bool]]:
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
                    yield line, head, body, True
                    head, body, blocks = [], [], 0
            continue
        if word == "}":
            head, body, blocks = [], [], 0
        elif word == ";":
            if head:
                yield line, head, body, blocks > 0
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


def forward_declaration(head: list[str], has_body: bool) -> bool:
    return not has_body and bare(head)[:1] in ([key] for key in CLASS_KEYS)


def declared_names(text: str, forward: bool = True) -> set[str]:
    found = macros(text)
    for line, head, body, has_body in declarations(text):
        names = declared_by(head, body)
        if names is None:
            raise ValueError(f"line {line} declares something the include-use check cannot name: {' '.join(head[:12])}")
        if forward or not forward_declaration(head, has_body):
            found |= names
    return found


def stem_headers(root: str = PROJECT) -> dict[str, str]:
    found = {}
    for layer in STEMS:
        for path in sources(root, layer):
            stem, extension = os.path.splitext(os.path.basename(path))
            if extension == ".hxx" and stem in (row_of(stem), row_of(stem) + INSIDE_SUFFIX):
                found[layer_path(layer, stem + extension)] = read(path)
    return found


def stem_declarations(root: str = PROJECT, forward: bool = True) -> dict[str, set[str]]:
    declared: dict[str, set[str]] = {}
    for header, text in stem_headers(root).items():
        try:
            declared[header] = declared_names(text, forward)
        except ValueError as e:
            raise ValueError(f"{header}: {e}") from None
    return declared


def unused_includes(root: str = PROJECT) -> list[tuple[str, str]]:
    declared = stem_declarations(root)
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


def needs(head: list[str], body: list[str]) -> set[str]:
    return set(head) if function_like(head) else set(head) | set(body)


def given_words(root: str, layers: list[str]) -> set[str]:
    found: set[str] = set()
    for layer in layers:
        for path in sources(root, layer):
            if layer_path(layer, os.path.basename(path)) != SEES_ALL:
                found |= words_in(read(path))
    return found


def unused_names(root: str = PROJECT) -> list[tuple[str, str]]:
    found = []
    for stem in sorted({stem for stems in GIVEN.values() for stem in stems}):
        header = layer_path(GIVEN_LAYER, stem + ".hxx")
        path = os.path.join(root, header)
        if not os.path.exists(path):
            continue
        text = read(path)
        try:
            declared = declared_names(text, forward=False)
            needed = [(declared_by(head, body) or set(), needs(head, body)) for _, head, body, has_body in declarations(text) if not forward_declaration(head, has_body)]
        except ValueError as e:
            raise ValueError(f"{header}: {e}") from None
        used = declared & given_words(root, [layer for layer, stems in GIVEN.items() if stem in stems])
        grown = True
        while grown:
            more = set().union(*(need for names, need in needed if not names.isdisjoint(used))) & declared
            grown = not more <= used
            used |= more
        found.extend((header, name) for name in sorted(declared - used))
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


def top_down(found: dict[str, tuple[str, ...]]) -> list[tuple[str, tuple[str, ...]]]:
    return sorted(found.items(), key=lambda item: top_down_order(item[0]))


def by_layer(rows: tuple[tuple[str, tuple[str, ...]], ...]) -> dict[str, tuple[str, ...]]:
    found: dict[str, tuple[str, ...]] = {}
    for file, names in rows:
        layer = layer_of(file)
        assert layer is not None, file
        found[layer] = found.get(layer, ()) + names
    return found


def hold_registered(root: str = PROJECT) -> None:
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
    hold_registered(root)
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


OWNS_HEADING = "## What each layer owns"
OWNS_INTRO = "New code goes in the layer that owns what it works on, and a lower layer never decides for a higher one. A row under a layer is a file stem and the one concern it holds."


def owns(root: str = PROJECT) -> str:
    assert tuple(OWNS) == LAYERS, tuple(OWNS)
    listed = stem_rows(root)
    rows = []
    for layer in reversed(LAYERS):
        rows.append(f"- `{layer}/`: {OWNS[layer]}" + (f" Rules at home here: {', '.join(rules_of(layer))}." if rules_of(layer) else ""))
        rows.extend(f"  - `{stem}`: {holds}" + (f". {standing(stem)}" if layer == GIVEN_LAYER else "") for stem, holds in listed.get(layer, {}).items())
    return "\n".join([OWNS_HEADING, "", OWNS_INTRO, ""] + rows) + "\n"


def grant_headers(root: str = PROJECT) -> dict[str, str]:
    return {layer_path(layer, os.path.basename(path)): read(path) for layer in GRANT_LAYERS for path in sources(root, layer) if path.endswith(".hxx")}


def unbacked(root: str = PROJECT) -> list[tuple[str, str]]:
    found = {(owner, friend) for text in grant_headers(root).values() for _, owner, friend in friends(text)}
    return [(owner, friend) for owner, granted in GRANTS.items() for friend in granted if (owner, friend) not in found]


def hold_grants(root: str = PROJECT) -> None:
    stray = [f"{header}:{line}" for header, text in grant_headers(root).items() for line in ungranted(header, text)]
    if stray:
        raise ValueError(f"{', '.join(stray)}: a friend line that no entry of GRANTS in {GRANTS_HOME} lists: add the entry")
    stale = unbacked(root)
    if stale:
        raise ValueError(f"{', '.join(f'{friend} in {owner}' for owner, friend in stale)}: an entry of GRANTS in {GRANTS_HOME} that no friend line backs: remove it")


TITLE = "# CarpetX's internals"
PREAMBLE = "`python3 scripts/lint doc --write` writes this file. `python3 scripts/lint check --help` prints what every rule named below holds."
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
        parts.extend([f"## {title}", "", "```mermaid", picture(root).rstrip("\n"), "```", ""])
        if texts:
            parts.append(legend(texts))
    hold_grants(root)
    parts.append(owns(root))
    return "\n".join(parts)
