from __future__ import annotations

import os
import shutil
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Sequence

from checks import plural, wrap
from files import PROJECT

from .rules import LAYERS, LINKS, TABLE, links

OBJECT_SUFFIX = ".cxx.o"
NAMESPACE = "7CarpetX"
STRONG = frozenset("TDBRSG")
NO_IMPORT = "no import behind this row of LINKS"
RULE = wrap(
    "symbols: an object the build made from a CarpetX .cxx imports a symbol of namespace CarpetX only from an "
    f"object of its own layer or of a layer its rows of READS and LINKS in {TABLE} name, whatever declared it: a "
    "header, a prototype, a block-scope declaration or an extern. A row of LINKS is a pair no #include shows, an "
    "import that arrives through another layer's header, and it is deleted with the last import across it. No reach "
    "is excepted: a body a lower layer calls moves down to that layer, and a pair in use has its row."
)

Demangle = Callable[[Iterable[str]], dict[str, str]]


class SymbolsError(RuntimeError):
    pass


@dataclass(frozen=True)
class Reach:
    object: str
    layer: str
    to: str
    symbol: str


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


def nm(path: str, flag: str) -> str:
    done = subprocess.run(["nm", flag, path], capture_output=True, text=True)
    if done.returncode != 0:
        raise SymbolsError(f"nm {flag} {path}: {done.stderr.strip()}")
    return done.stdout


def strong_definitions(definitions: dict[str, set[tuple[str, str]]], layers: dict[str, str]) -> dict[str, str]:
    homes: dict[str, set[str]] = defaultdict(set)
    for obj, found in definitions.items():
        for kind, name in found:
            if kind in STRONG:
                homes[name].add(obj)
    return {name: layers[next(iter(objs))] for name, objs in homes.items() if len(objs) == 1}


def reaches(layers: dict[str, str], defined: dict[str, str], imports: dict[str, set[str]]) -> list[Reach]:
    found = []
    for obj in sorted(layers):
        for name in sorted(imports.get(obj, ())):
            to = defined.get(name)
            if to is not None and to not in links(layers[obj]):
                found.append(Reach(obj, layers[obj], to, name))
    return found


def idle_links(layers: dict[str, str], defined: dict[str, str], imports: dict[str, set[str]]) -> list[tuple[str, str]]:
    used = {(layer, defined[name]) for obj, layer in layers.items() for name in imports.get(obj, ()) if name in defined}
    return [(layer, target) for layer, targets in LINKS.items() for target in targets if (layer, target) not in used]


def demangle(names: Iterable[str]) -> dict[str, str]:
    names = list(dict.fromkeys(names))
    if not names:
        return {}
    if shutil.which("c++filt") is None:
        raise SymbolsError("c++filt is not on PATH")
    done = subprocess.run(["c++filt"], input="\n".join(names) + "\n", capture_output=True, text=True)
    return dict(zip(names, done.stdout.splitlines()))


def summary(found: Sequence[Reach], idle: Sequence[tuple[str, str]] = ()) -> str:
    if not found and not idle:
        return "✓ symbols: every CarpetX import from the importing object's layer or one its rows name, and an import behind every row of LINKS"
    counts = [f"{plural(len(found), 'import')} outside the layer table"] if found else []
    counts += [f"{plural(len(idle), 'row')} of LINKS with no import"] if idle else []
    return "✗ symbols: " + ", ".join(counts)


def render(found: Sequence[Reach], readable: dict[str, str], idle: Sequence[tuple[str, str]] = ()) -> str:
    out = []
    if found or idle:
        width = max(len(place) for place in [r.object for r in found] + [f"{layer}/" for layer, _ in idle])
        out.extend(f"{r.object.ljust(width)}   -> {r.to.ljust(9)}   {readable[r.symbol]}" for r in found)
        out.extend(f"{f'{layer}/'.ljust(width)}   -> {target.ljust(9)}   {NO_IMPORT}" for layer, target in idle)
        out.extend(RULE.split("\n"))
    out.append(summary(found, idle))
    return "\n".join(out)


def check(objdir: str, demangler: Optional[Demangle] = None) -> tuple[str, bool]:
    layers = layer_objects(objdir)
    definitions = {obj: parse_definitions(nm(os.path.join(objdir, obj), "--defined-only")) for obj in layers}
    imports = {obj: parse_imports(nm(os.path.join(objdir, obj), "--undefined-only")) for obj in layers}
    defined = strong_definitions(definitions, layers)
    found = reaches(layers, defined, imports)
    idle = idle_links(layers, defined, imports)
    readable = (demangler or demangle)(r.symbol for r in found)
    return render(found, readable, idle), not found and not idle
