from __future__ import annotations

import os
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from pathlib import PurePosixPath
from typing import Callable, Iterable, Optional, TypeVar

TOOL = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(os.path.dirname(TOOL))
CXX_SUFFIXES = {".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx"}
FORTRAN_SUFFIXES = {".f90", ".F90", ".f95", ".F95", ".f03", ".F03", ".f08", ".F08"}
PAR_NAME = re.compile(r"\.par(?:-\w+)?$")
BASE_REFS = ("origin/dev", "dev")
EMPTY_TREE = "4b825dc642cb6eb9a060e54bf8d69288fbee4904"

T = TypeVar("T")
R = TypeVar("R")


class Lang(Enum):
    CXX = "C/C++"
    FORTRAN = "Fortran"
    MAKE = "make.code.defn"
    PAR = ".par"
    INTERFACE = "interface.ccl"
    PARAM = "param.ccl"
    SCHEDULE = "schedule.ccl"
    CONFIGURATION = "configuration.ccl"
    TEST = "test.ccl"


CCL_LANGS = {lang.value: lang for lang in (Lang.INTERFACE, Lang.PARAM, Lang.SCHEDULE, Lang.CONFIGURATION, Lang.TEST)}


def is_thorn(root: str, name: str) -> bool:
    return os.path.isfile(os.path.join(root, name, "interface.ccl"))


def lang_of(name: str) -> Optional[Lang]:
    path = PurePosixPath(name)
    parts = path.parts
    if len(parts) < 2:
        return None
    in_src = len(parts) >= 3 and parts[1] == "src"
    if in_src and path.suffix in CXX_SUFFIXES:
        return Lang.CXX
    if in_src and path.suffix in FORTRAN_SUFFIXES:
        return Lang.FORTRAN
    if in_src and path.name == "make.code.defn":
        return Lang.MAKE
    if path.suffix == ".ccl":
        return CCL_LANGS.get(path.name, Lang.TEST)
    if PAR_NAME.search(path.name):
        return Lang.PAR
    return None


def covered(name: str, root: str = PROJECT) -> Optional[Lang]:
    parts = PurePosixPath(name).parts
    if not parts or not is_thorn(root, parts[0]):
        return None
    return lang_of(name)


def thorns_in(names: Iterable[str]) -> set:
    return {name.split("/")[0] for name in names if name.count("/") == 1 and name.endswith("/interface.ccl")}


def repo_name(path: str, root: str = PROJECT) -> Optional[str]:
    rel = os.path.relpath(os.path.realpath(path), os.path.realpath(root))
    if rel == os.curdir or rel == os.pardir or rel.startswith(os.pardir + os.sep):
        return None
    return rel.replace(os.sep, "/")


def is_regular(path: str) -> bool:
    return os.path.isfile(path) and not os.path.islink(path)


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
            lang = covered(name, root) if name is not None and is_regular(candidate) else None
            if lang is not None:
                found.setdefault(name, (name, candidate, lang))
    return [found[name] for name in sorted(found)]


def list_tree(root: str = PROJECT) -> list[tuple[str, str, Lang]]:
    if is_work_tree(root):
        names = split_z(git(root, "ls-files", "-z", "--cached", "--others", "--exclude-standard"))
    else:
        names = []
        for top, dirs, files in os.walk(root):
            dirs[:] = sorted(d for d in dirs if d != ".git")
            names.extend(os.path.relpath(os.path.join(top, n), root).replace(os.sep, "/") for n in files)
    found = []
    for name in sorted(set(names)):
        path = os.path.join(root, name)
        lang = covered(name, root)
        if lang is not None and is_regular(path):
            found.append((name, path, lang))
    return found


def read_text(path: str) -> str:
    with open(path, encoding="utf-8", errors="surrogateescape", newline="") as f:
        return f.read()


def decode(data: bytes) -> str:
    return data.decode("utf-8", "surrogateescape")


def split_z(data: bytes) -> list[str]:
    return [decode(item) for item in data.split(b"\0") if item]


class GitError(RuntimeError):
    pass


def git(root: str, *args: str, check: bool = True) -> bytes:
    proc = subprocess.run(["git", "-C", root, *args], stdin=subprocess.DEVNULL, capture_output=True)
    if check and proc.returncode != 0:
        why = decode(proc.stderr).strip().split("\n")[0]
        raise GitError(f"git {' '.join(args[:2])} exited with status {proc.returncode}: {why}")
    return proc.stdout


def is_work_tree(root: str) -> bool:
    proc = subprocess.run(
        ["git", "-C", root, "rev-parse", "--is-inside-work-tree"], stdin=subprocess.DEVNULL, capture_output=True
    )
    return proc.returncode == 0 and proc.stdout.strip() == b"true"


def resolve(root: str, rev: str) -> Optional[str]:
    proc = subprocess.run(
        ["git", "-C", root, "rev-parse", "--verify", "--quiet", "--end-of-options", rev + "^{commit}"],
        stdin=subprocess.DEVNULL,
        capture_output=True,
    )
    return decode(proc.stdout).strip() if proc.returncode == 0 else None


def merge_base(root: str, refs: tuple = BASE_REFS) -> str:
    head = resolve(root, "HEAD")
    if head is None:
        return EMPTY_TREE
    present = [ref for ref in refs if resolve(root, ref) is not None]
    if not present:
        return head
    base = decode(git(root, "merge-base", "HEAD", *present, check=False)).strip()
    return base or head


def changed(root: str, base: str) -> list[tuple[Optional[str], str]]:
    out = split_z(git(root, "diff", "-z", "--name-status", "-M", "--no-ext-diff", "--no-color", base))
    found = []
    i = 0
    while i < len(out):
        status = out[i]
        if status[:1] in ("R", "C"):
            old, new = out[i + 1], out[i + 2]
            found.append((old if status[0] == "R" else None, new))
            i += 3
            continue
        path = out[i + 1]
        if status[:1] != "D":
            found.append((None if status[:1] == "A" else path, path))
        i += 2
    return found


def untracked(root: str) -> list[str]:
    return split_z(git(root, "ls-files", "-z", "--others", "--exclude-standard"))


def show(root: str, rev: str, name: Optional[str]) -> str:
    if name is None or rev == EMPTY_TREE:
        return ""
    proc = subprocess.run(
        ["git", "-C", root, "cat-file", "blob", f"{rev}:{name}"], stdin=subprocess.DEVNULL, capture_output=True
    )
    return decode(proc.stdout) if proc.returncode == 0 else ""


def tree_names(root: str, rev: str) -> list[str]:
    return split_z(git(root, "ls-tree", "-r", "-z", "--name-only", rev))


def blobs(root: str, rev: str, names: list[str]) -> dict:
    if not names:
        return {}
    request = "".join(f"{rev}:{name}\n" for name in names).encode("utf-8", "surrogateescape")
    proc = subprocess.run(["git", "-C", root, "cat-file", "--batch"], input=request, capture_output=True)
    if proc.returncode != 0:
        raise GitError(f"git cat-file --batch exited with status {proc.returncode}: {decode(proc.stderr).strip()}")
    data = proc.stdout
    found = {}
    pos = 0
    for name in names:
        header_end = data.index(b"\n", pos)
        fields = data[pos:header_end].split(b" ")
        if len(fields) != 3 or fields[1] != b"blob":
            raise GitError(f"{rev}:{name} is not a file")
        start = header_end + 1
        found[name] = decode(data[start : start + int(fields[2])])
        pos = start + int(fields[2]) + 1
    return found


def parallel(fn: Callable[[T], R], items: Iterable[T]) -> list[R]:
    items = list(items)
    if len(items) < 2:
        return [fn(item) for item in items]
    with ThreadPoolExecutor(max_workers=min(32, os.cpu_count() or 1)) as pool:
        return list(pool.map(fn, items))
