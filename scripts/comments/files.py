from __future__ import annotations

import os
from enum import Enum
from pathlib import PurePosixPath
from typing import Optional

TOOL = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(os.path.dirname(TOOL))
CXX_SUFFIXES = {".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx"}


class Lang(Enum):
    CXX = "C/C++"


def is_thorn(root: str, name: str) -> bool:
    return os.path.isfile(os.path.join(root, name, "interface.ccl"))


def covered(name: str, root: str = PROJECT) -> Optional[Lang]:
    parts = PurePosixPath(name).parts
    if len(parts) < 3 or parts[1] != "src" or not is_thorn(root, parts[0]):
        return None
    if PurePosixPath(name).suffix in CXX_SUFFIXES:
        return Lang.CXX
    return None


def repo_name(path: str, root: str = PROJECT) -> Optional[str]:
    rel = os.path.relpath(os.path.realpath(path), os.path.realpath(root))
    if rel == os.curdir or rel == os.pardir or rel.startswith(os.pardir + os.sep):
        return None
    return rel.replace(os.sep, "/")


def list_covered(paths: list[str], root: str = PROJECT) -> list[tuple[str, str, Lang]]:
    found = {}
    for path in paths:
        if os.path.isdir(path):
            candidates = []
            for top, dirs, names in os.walk(path):
                dirs[:] = sorted(d for d in dirs if d != ".git")
                candidates.extend(os.path.join(top, n) for n in names)
        elif os.path.isfile(path):
            candidates = [path]
        else:
            raise FileNotFoundError(f"no such file or directory: {path}")
        for candidate in candidates:
            name = repo_name(candidate, root)
            lang = covered(name, root) if name is not None and os.path.isfile(candidate) else None
            if lang is not None:
                found.setdefault(name, (name, candidate, lang))
    return [found[name] for name in sorted(found)]


def read_text(path: str) -> str:
    with open(path, encoding="utf-8", errors="surrogateescape", newline="") as f:
        return f.read()
