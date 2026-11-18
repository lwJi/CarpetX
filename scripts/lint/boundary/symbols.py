from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Iterable, Optional

from checks import plural, wrap
from files import PROJECT

from .doc import MAP, PUBLIC_BODIES, PUBLIC_HEADERS, declared_names, header_reads, read, stem_declarations
from .rules import INSIDE_SUFFIX, LAYERS, PRIVATE_DIR, READS_HOME, STEMS, STEMS_HOME, dirs, linkable, row_of

OBJECT_SUFFIX = ".cxx.o"
HEADER_SUFFIX = ".hxx"
NAMESPACE = "7CarpetX"
STRONG = frozenset("TDBRSG")
WEAK = "W"
UNNAMED_NAMESPACE = "_GLOBAL__N"
MACH_O = sys.platform == "darwin"
MACH_O_WEAK = re.compile(r"\)\s+weak\s.*\s(\S+)$")
MEMBER = re.compile(r"_{1,2}Z(?:T[VTISCWH]|G[VR]|Thn?\d+_|Tvn?\d+_n?\d+_)?Z?N[rVK]*[RO]?" + NAMESPACE + r"(?:(\d+)|L(\d+)|([a-zA-Z]{2}))")
OPERATORS = {
    "ps": "+", "ng": "-", "ad": "&", "de": "*", "co": "~", "pl": "+", "mi": "-", "ml": "*", "dv": "/", "rm": "%", "an": "&", "or": "|", "eo": "^",
    "aS": "=", "pL": "+=", "mI": "-=", "mL": "*=", "dV": "/=", "rM": "%=", "aN": "&=", "oR": "|=", "eO": "^=", "ls": "<<", "rs": ">>", "lS": "<<=", "rS": ">>=",
    "eq": "==", "ne": "!=", "lt": "<", "gt": ">", "le": "<=", "ge": ">=", "ss": "<=>", "nt": "!", "aa": "&&", "oo": "||", "pp": "++", "mm": "--",
    "cm": ",", "pm": "->*", "pt": "->", "cl": "()", "ix": "[]",
}
OPERATOR = "operator"
REACH_RULE = wrap(
    "symbols: an object the build made from a CarpetX .cxx imports a symbol of namespace CarpetX only from an "
    f"object of its own layer, of a layer its own reads by READS in {READS_HOME}, or of a layer the headers of one "
    "it reads include, since a header carries its imports into the object that includes it, whatever declared the "
    "symbol: a header, a prototype, a block-scope declaration or an extern. No reach is excepted: a body a layer "
    "calls moves into a layer it reads."
)
STEM_RULE = wrap(
    f"stems: in a layer that STEMS in {STEMS_HOME} lists, a header and its source share a stem, and the object the "
    f"build made from <layer>/<stem>.cxx strongly defines in namespace CarpetX only what <layer>/<stem>.hxx and "
    f"<layer>/<stem>{INSIDE_SUFFIX}.hxx declare: a "
    "function, an object or the outermost class of a member that one of them declares at namespace scope, or that a public "
    "header declares when PUBLIC_BODIES in scripts/lint/boundary/doc.py lists the source; a header that only uses a "
    "name does not declare it, and one that declares a class with no body leaves the members to the header that "
    "defines the class. A body another header declares moves "
    "to the source of that header's stem, and one no header declares is declared there or becomes file-local. The "
    f"match is by name, never by signature, and {MAP} says what each stem holds."
)
RULE = REACH_RULE + "\n" + STEM_RULE

Demangle = Callable[[Iterable[str]], dict[str, str]]


class SymbolsError(RuntimeError):
    pass


@dataclass(frozen=True)
class Reach:
    object: str
    layer: str
    to: str
    symbol: str


@dataclass(frozen=True)
class Stray:
    object: str
    symbol: str
    header: str


def in_tree(obj: str, root: str = PROJECT) -> bool:
    return os.path.isfile(os.path.join(root, "CarpetX", "src", obj[: -len(".o")]))


def layer_objects(objdir: str, keep: Callable[[str], bool] = in_tree) -> dict[str, str]:
    found = {}
    for layer in LAYERS:
        path = os.path.join(objdir, layer)
        if not os.path.isdir(path):
            continue
        for entry in sorted(os.listdir(path)):
            if entry.endswith(OBJECT_SUFFIX) and keep(f"{layer}/{entry}"):
                found[f"{layer}/{entry}"] = layer
    if not found:
        raise SymbolsError(f"no {OBJECT_SUFFIX} under a layer directory of {objdir}; build first")
    return found


def parse_definitions(text: str) -> set[tuple[str, str]]:
    found = set()
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 2 and NAMESPACE in parts[-1]:
            found.add((parts[-2], parts[-1]))
    return found


def parse_imports(text: str) -> set[str]:
    found = set()
    for line in text.splitlines():
        parts = line.split()
        if parts and NAMESPACE in parts[-1]:
            found.add(parts[-1])
    return found


def parse_mach_o_weak(text: str) -> set[str]:
    return {m.group(1) for m in map(MACH_O_WEAK.search, text.splitlines()) if m and NAMESPACE in m.group(1)}


def weakened(found: set[tuple[str, str]], weak: set[str]) -> set[tuple[str, str]]:
    return {(WEAK if name in weak else kind, name) for kind, name in found}


def nm(path: str, *flags: str) -> str:
    done = subprocess.run(["nm", *flags, path], capture_output=True, text=True)
    if done.returncode != 0:
        raise SymbolsError(f"nm {' '.join(flags)} {path}: {done.stderr.strip()}")
    return done.stdout


def defined_in(path: str, mach_o: bool = MACH_O) -> set[tuple[str, str]]:
    found = parse_definitions(nm(path, "--defined-only"))
    return weakened(found, parse_mach_o_weak(nm(path, "-m", "--defined-only"))) if mach_o else found


def strong_definitions(definitions: dict[str, set[tuple[str, str]]], layers: dict[str, str]) -> dict[str, str]:
    homes: dict[str, set[str]] = defaultdict(set)
    for obj, found in definitions.items():
        for kind, name in found:
            if kind in STRONG:
                homes[name].add(obj)
    return {name: layers[next(iter(objs))] for name, objs in homes.items() if len(objs) == 1}


def reaches(layers: dict[str, str], defined: dict[str, str], imports: dict[str, set[str]], headers: dict[str, tuple[str, ...]]) -> list[Reach]:
    found = []
    for obj in sorted(layers):
        for name in sorted(imports.get(obj, ())):
            to = defined.get(name)
            if to is not None and to not in linkable(layers[obj], headers):
                found.append(Reach(obj, layers[obj], to, name))
    return found


def declared_name(symbol: str) -> Optional[str]:
    m = MEMBER.match(symbol)
    if m is None:
        return None
    length, local, operator = m.groups()
    if operator is not None:
        return OPERATOR + OPERATORS[operator] if operator in OPERATORS else symbol
    name = symbol[m.end() : m.end() + int(length or local)]
    return None if name.startswith(UNNAMED_NAMESPACE) else name


def headers_of(obj: str) -> tuple[str, str]:
    layer, stem = os.path.split(obj[: -len(OBJECT_SUFFIX)])
    base = PRIVATE_DIR + os.path.join(layer, row_of(stem))
    return base + HEADER_SUFFIX, base + INSIDE_SUFFIX + HEADER_SUFFIX


def strays(layers: dict[str, str], definitions: dict[str, set[tuple[str, str]]], headers: dict[str, set[str]], public: dict[str, set[str]]) -> list[Stray]:
    found = []
    for obj in sorted(layers):
        if layers[obj] not in STEMS:
            continue
        held = headers_of(obj)
        header = next((header for header in held if header in headers), held[0])
        declared = set().union(*(headers.get(header, set()) for header in held)) | public.get(obj, set())
        for kind, symbol in sorted(definitions.get(obj, ()), key=lambda definition: definition[1]):
            name = declared_name(symbol) if kind in STRONG else None
            if name is not None and name not in declared:
                found.append(Stray(obj, symbol, header))
    return found


def stem_names(root: str = PROJECT) -> dict[str, set[str]]:
    try:
        return stem_declarations(root, forward=False)
    except ValueError as e:
        raise SymbolsError(str(e)) from None


def public_names(root: str = PROJECT) -> dict[str, set[str]]:
    declared: set[str] = set()
    for header in PUBLIC_HEADERS:
        try:
            declared |= declared_names(read(os.path.join(root, header)), forward=False)
        except ValueError as e:
            raise SymbolsError(f"{header}: {e}") from None
    return {file[len(PRIVATE_DIR) :] + ".o": declared for file, _ in PUBLIC_BODIES}


def demangle(names: Iterable[str]) -> dict[str, str]:
    names = list(dict.fromkeys(names))
    if not names:
        return {}
    if shutil.which("c++filt") is None:
        raise SymbolsError("c++filt is not on PATH")
    done = subprocess.run(["c++filt"], input="\n".join(names) + "\n", capture_output=True, text=True)
    return dict(zip(names, done.stdout.splitlines()))


def summary(unread: list[Reach], stray: list[Stray]) -> str:
    if not unread and not stray:
        held = f", and every definition of {dirs(tuple(STEMS), 'and')} declared in a header of its stem" if STEMS else ""
        return "✓ symbols: every CarpetX import from the importing object's layer or one it reads" + held
    counts = []
    if unread:
        counts.append(f"{plural(len(unread), 'import')} from a layer the importing object's layer does not read")
    if stray:
        counts.append(f"{plural(len(stray), 'definition')} no header of its stem declares")
    return "✗ symbols: " + ", ".join(counts)


def render(unread: list[Reach], stray: list[Stray], readable: dict[str, str]) -> str:
    out = []
    if unread:
        width = max(len(r.object) for r in unread)
        out.extend(f"{r.object.ljust(width)}   -> {r.to.ljust(9)}   {readable[r.symbol]}" for r in unread)
        out.extend(REACH_RULE.split("\n"))
    if stray:
        width = max(len(s.object) for s in stray)
        out.extend(f"{s.object.ljust(width)}   not in {s.header[len(PRIVATE_DIR) :]}   {readable[s.symbol]}" for s in stray)
        out.extend(STEM_RULE.split("\n"))
    out.append(summary(unread, stray))
    return "\n".join(out)


def check(objdir: str, demangler: Optional[Demangle] = None) -> tuple[str, bool]:
    layers = layer_objects(objdir)
    definitions = {obj: defined_in(os.path.join(objdir, obj)) for obj in layers}
    imports = {obj: parse_imports(nm(os.path.join(objdir, obj), "--undefined-only")) for obj in layers}
    unread = reaches(layers, strong_definitions(definitions, layers), imports, header_reads())
    stray = strays(layers, definitions, stem_names(), public_names())
    readable = (demangler or demangle)([r.symbol for r in unread] + [s.symbol for s in stray])
    return render(unread, stray, readable), not unread and not stray
