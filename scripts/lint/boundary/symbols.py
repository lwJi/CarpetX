from __future__ import annotations

import os
import shutil
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Iterable, Optional

from checks import plural, wrap
from files import PROJECT

from .rules import LAYERS, READS_HOME, linkable

OBJECT_SUFFIX = ".cxx.o"
NAMESPACE = "7CarpetX"
STRONG = frozenset("TDBRSG")
RULE = wrap(
    "symbols: an object the build made from a CarpetX .cxx imports a symbol of namespace CarpetX only from an "
    f"object of its own layer or of one its layer reads, itself or through a layer it reads, by READS in {READS_HOME}, "
    "whatever declared it: a header, a prototype, a block-scope declaration or an extern. No reach is excepted: a "
    "body a layer calls moves into a layer it reads."
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
            if to is not None and to not in linkable(layers[obj]):
                found.append(Reach(obj, layers[obj], to, name))
    return found


def demangle(names: Iterable[str]) -> dict[str, str]:
    names = list(dict.fromkeys(names))
    if not names:
        return {}
    if shutil.which("c++filt") is None:
        raise SymbolsError("c++filt is not on PATH")
    done = subprocess.run(["c++filt"], input="\n".join(names) + "\n", capture_output=True, text=True)
    return dict(zip(names, done.stdout.splitlines()))


def summary(unread: list[Reach]) -> str:
    if not unread:
        return "✓ symbols: every CarpetX import from the importing object's layer or one it reads"
    return f"✗ symbols: {plural(len(unread), 'import')} from a layer the importing object's layer does not read"


def render(unread: list[Reach], readable: dict[str, str]) -> str:
    out = []
    if unread:
        width = max(len(r.object) for r in unread)
        out.extend(f"{r.object.ljust(width)}   -> {r.to.ljust(9)}   {readable[r.symbol]}" for r in unread)
        out.extend(RULE.split("\n"))
    out.append(summary(unread))
    return "\n".join(out)


def check(objdir: str, demangler: Optional[Demangle] = None) -> tuple[str, bool]:
    layers = layer_objects(objdir)
    definitions = {obj: parse_definitions(nm(os.path.join(objdir, obj), "--defined-only")) for obj in layers}
    imports = {obj: parse_imports(nm(os.path.join(objdir, obj), "--undefined-only")) for obj in layers}
    unread = reaches(layers, strong_definitions(definitions, layers), imports)
    readable = (demangler or demangle)(r.symbol for r in unread)
    return render(unread, readable), not unread
