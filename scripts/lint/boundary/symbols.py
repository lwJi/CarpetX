from __future__ import annotations

import os
import shutil
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Callable, Iterable, Optional

from checks import plural
from files import PROJECT

from .rules import LAYERS, below

OBJECT_SUFFIX = ".cxx.o"
NAMESPACE = "7CarpetX"
STRONG = frozenset("TDBRSG")
ACCEPTED = (
    ("api", "CarpetX::level_t::"),
    ("api", "CarpetX::window_t::"),
    ("api", "CarpetX::all_levels("),
    ("api", "CarpetX::variable("),
    ("api", "CarpetX::RecoverRKBoundary("),
    ("regrid", "CarpetX::CactusAmrCore::"),
    ("comm", "CarpetX::get_interpolator("),
)
RULE = (
    "symbols: an object the build made from a CarpetX .cxx imports a symbol of namespace CarpetX only from an\n"
    "object of its own layer or a lower one, whatever declared it: a header, a prototype, a block-scope\n"
    "declaration or an extern. A reach listed in ACCEPTED in scripts/lint/boundary/symbols.py is reported as\n"
    "accepted until the bodies move down; the list only shrinks."
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


def accepted(reach: Reach, readable: str) -> bool:
    return any(reach.to == layer and readable.startswith(prefix) for layer, prefix in ACCEPTED)


def demangle(names: Iterable[str]) -> dict[str, str]:
    names = list(dict.fromkeys(names))
    if not names:
        return {}
    if shutil.which("c++filt") is None:
        raise SymbolsError("c++filt is not on PATH")
    done = subprocess.run(["c++filt"], input="\n".join(names) + "\n", capture_output=True, text=True)
    return dict(zip(names, done.stdout.splitlines()))


def split(found: list[Reach], readable: dict[str, str]) -> tuple[list[Reach], list[Reach]]:
    upward = [r for r in found if not accepted(r, readable[r.symbol])]
    waived = [r for r in found if accepted(r, readable[r.symbol])]
    return upward, waived


def summary(upward: list[Reach], waived: list[Reach]) -> str:
    tail = f", {plural(len(waived), 'accepted import')} listed in ACCEPTED" if waived else ""
    if not upward:
        return "✓ symbols: every CarpetX import from the importing object's layer or a lower one" + tail
    return f"✗ symbols: {plural(len(upward), 'upward import')}" + tail


def render(upward: list[Reach], waived: list[Reach], readable: dict[str, str]) -> str:
    out = []
    if upward:
        width = max(len(r.object) for r in upward)
        out.extend(f"{r.object.ljust(width)}   -> {r.to.ljust(9)}   {readable[r.symbol]}" for r in upward)
        out.extend(RULE.split("\n"))
    if waived:
        counts = Counter((r.layer, r.to) for r in waived)
        out.append(
            "accepted: " + ", ".join(f"{a}/ -> {b}/ {n}" for (a, b), n in sorted(counts.items(), key=lambda kv: (LAYERS.index(kv[0][0]), LAYERS.index(kv[0][1]))))
        )
    out.append(summary(upward, waived))
    return "\n".join(out)


def check(objdir: str, demangler: Optional[Demangle] = None) -> tuple[str, bool]:
    layers = layer_objects(objdir)
    definitions = {obj: parse_definitions(nm(os.path.join(objdir, obj), "--defined-only")) for obj in layers}
    imports = {obj: parse_imports(nm(os.path.join(objdir, obj), "--undefined-only")) for obj in layers}
    found = reaches(layers, strong_definitions(definitions, layers), imports)
    readable = (demangler or demangle)(r.symbol for r in found)
    upward, waived = split(found, readable)
    return render(upward, waived, readable), not upward
