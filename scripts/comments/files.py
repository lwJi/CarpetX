import os
import re
import subprocess
from dataclasses import dataclass
from enum import Enum
from pathlib import PurePosixPath


class Lang(Enum):
    CXX = "C/C++"
    FORTRAN = "Fortran"
    CCL = ".ccl"
    PAR = ".par"


CXX_SUFFIXES = {".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx"}
FORTRAN_SUFFIXES = {".f90", ".F90"}
PAR_NAME = re.compile(r"\.par(-\w+)?$")


def covered(path: str) -> Lang | None:
    p = PurePosixPath(path)
    if "src" in p.parts[:-1] and p.suffix in CXX_SUFFIXES:
        return Lang.CXX
    if "src" in p.parts[:-1] and p.suffix in FORTRAN_SUFFIXES:
        return Lang.FORTRAN
    if p.suffix == ".ccl":
        return Lang.CCL
    if PAR_NAME.search(p.name):
        return Lang.PAR
    return None


@dataclass(frozen=True)
class CoveredFile:
    path: str
    name: str
    lang: Lang


def git_toplevel(directory: str) -> str | None:
    result = subprocess.run(
        ["git", "-C", directory, "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def default_root() -> str:
    return git_toplevel(".") or "."


def _git_names(directory: str, untracked: bool) -> list[str]:
    command = ["git", "-C", directory, "ls-files", "-z", "--full-name", "--cached"]
    if untracked:
        command += ["--others", "--exclude-standard"]
    out = subprocess.run(command, capture_output=True, check=True).stdout
    return sorted({n for n in out.decode("utf-8", "surrogateescape").split("\0") if n})


def _walk_names(directory: str) -> list[str]:
    names = []
    for root, dirs, filenames in os.walk(directory):
        dirs[:] = [d for d in dirs if d != ".git"]
        for f in filenames:
            names.append(os.path.relpath(os.path.join(root, f), directory))
    return sorted(names)


def _regular_file(path: str) -> bool:
    return os.path.isfile(path) and not os.path.islink(path)


def _list_directory(directory: str, untracked: bool) -> list[CoveredFile]:
    top = git_toplevel(directory)
    if top is None:
        base, names = directory, _walk_names(directory)
    else:
        base, names = top, _git_names(directory, untracked)
    found = []
    for name in names:
        lang = covered(name)
        path = os.path.join(base, name)
        if lang is not None and _regular_file(path):
            found.append(CoveredFile(os.path.relpath(path), name, lang))
    return found


def _single_file(path: str) -> list[CoveredFile]:
    real = os.path.realpath(path)
    top = git_toplevel(os.path.dirname(real))
    name = os.path.relpath(real, os.path.realpath(top)) if top else path
    lang = covered(name)
    return [CoveredFile(os.path.relpath(path), name, lang)] if lang else []


def list_covered(paths: list[str], untracked: bool = False) -> list[CoveredFile]:
    found: dict[str, CoveredFile] = {}
    for p in paths:
        if os.path.isdir(p):
            items = _list_directory(p, untracked)
        elif os.path.isfile(p):
            items = _single_file(p)
        else:
            raise FileNotFoundError(f"no such file or directory: {p}")
        for item in items:
            found.setdefault(os.path.realpath(item.path), item)
    return sorted(found.values(), key=lambda f: f.path)


def read_text(path: str) -> str:
    with open(path, encoding="utf-8", errors="surrogateescape", newline="") as f:
        return f.read()


def write_text(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8", errors="surrogateescape", newline="") as f:
        f.write(text)
