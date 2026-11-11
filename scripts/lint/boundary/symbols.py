from __future__ import annotations

import os
import shutil
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Iterable, Optional

from checks import plural, wrap
from files import PROJECT

from .rules import LAYERS, below

OBJECT_SUFFIX = ".cxx.o"
NAMESPACE = "7CarpetX"
STRONG = frozenset("TDBRSG")
RULE = wrap(
    "symbols: an object the build made from a CarpetX .cxx imports a symbol of namespace CarpetX only from an "
    "object of its own layer or a lower one, whatever declared it: a header, a prototype, a block-scope declaration "
    "or an extern. No reach is excepted: a body a lower layer calls moves down to that layer."
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
            if to is not None and to not in below(layers[obj]):
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


def summary(upward: list[Reach]) -> str:
    if not upward:
        return "✓ symbols: every CarpetX import from the importing object's layer or a lower one"
    return f"✗ symbols: {plural(len(upward), 'upward import')}"


def render(upward: list[Reach], readable: dict[str, str]) -> str:
    out = []
    if upward:
        width = max(len(r.object) for r in upward)
        out.extend(f"{r.object.ljust(width)}   -> {r.to.ljust(9)}   {readable[r.symbol]}" for r in upward)
        out.extend(RULE.split("\n"))
    out.append(summary(upward))
    return "\n".join(out)


def check(objdir: str, demangler: Optional[Demangle] = None) -> tuple[str, bool]:
    layers = layer_objects(objdir)
    definitions = {obj: parse_definitions(nm(os.path.join(objdir, obj), "--defined-only")) for obj in layers}
    imports = {obj: parse_imports(nm(os.path.join(objdir, obj), "--undefined-only")) for obj in layers}
    upward = reaches(layers, strong_definitions(definitions, layers), imports)
    readable = (demangler or demangle)(r.symbol for r in upward)
    return render(upward, readable), not upward
