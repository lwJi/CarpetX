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


def simulate_edit(tool_name: str, tool_input: dict, before: Optional[str]) -> Optional[str]:
    if tool_name == "Write":
        return tool_input["content"]
    if tool_name != "Edit":
        raise ValueError(f"unexpected tool {tool_name!r}")
    old, new = tool_input["old_string"], tool_input["new_string"]
    if old == "":
        return new if not before else None
    if before is None:
        return None
    replace_all = bool(tool_input.get("replace_all", False))
    count = before.count(old)
    if count == 0 or (count > 1 and not replace_all):
        return None
    return before.replace(old, new, -1 if replace_all else 1)


def decode(data: bytes) -> str:
    return data.decode("utf-8", "surrogateescape")


def split_z(data: bytes) -> list[str]:
    return [decode(item) for item in data.split(b"\0") if item]


class GitError(RuntimeError):
    pass


def git(root: str, *args: str, check: bool = True, input: Optional[bytes] = None) -> bytes:
    stdin = subprocess.DEVNULL if input is None else None
    proc = subprocess.run(["git", "-C", root, *args], stdin=stdin, input=input, capture_output=True)
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


def common_ancestor(root: str, rev: str) -> Optional[str]:
    return decode(git(root, "merge-base", rev, "HEAD", check=False)).strip() or None


def merged_tips(root: str, base: str) -> list[str]:
    if base == EMPTY_TREE:
        return []
    out = decode(git(root, "rev-list", "--first-parent", "--merges", "--parents", f"{base}..HEAD"))
    return [parent for line in out.splitlines() for parent in line.split()[2:]]


def untracked(root: str) -> list[str]:
    return split_z(git(root, "ls-files", "-z", "--others", "--exclude-standard"))


def name_status(fields: list[str]) -> list[tuple[str, str, str]]:
    found = []
    i = 0
    while i < len(fields):
        status = fields[i][:1]
        if status in ("R", "C"):
            found.append((status, fields[i + 1], fields[i + 2]))
            i += 3
        else:
            found.append((status, fields[i + 1], fields[i + 1]))
            i += 2
    return found


def follow(trail: dict[str, Optional[str]], changes: list[tuple[str, str, str]]) -> dict[str, Optional[str]]:
    born: dict[str, Optional[str]] = {}
    gone = set()
    for status, old, new in changes:
        if status in ("R", "C"):
            born[new] = trail.get(old, old)
        elif status == "A":
            born[new] = None
        if status in ("R", "D"):
            gone.add(old)
    kept = {name: back for name, back in trail.items() if name not in gone}
    kept.update(born)
    return kept


def first_parent_steps(root: str, start: str) -> Optional[list[tuple[str, str]]]:
    lines = decode(git(root, "rev-list", "--first-parent", "--reverse", "--parents", f"{start}..HEAD")).splitlines()
    path = [line.split() for line in lines if line]
    joins = [k for k, ids in enumerate(path) if start in ids[1:]]
    if not joins:
        return None
    k = joins[0]
    return [(path[k][0], start)] + [(ids[0], ids[1]) for ids in path[k + 1 :]]


STEP_CHANGES: dict[tuple[str, str], list[tuple[str, str, str]]] = {}


def step_changes(root: str, steps: list[tuple[str, str]]) -> list[list[tuple[str, str, str]]]:
    missing = [step for step in steps if step not in STEP_CHANGES]
    if missing:
        pairs = "".join(f"{commit} {parent}\n" for commit, parent in missing).encode()
        out = split_z(git(root, "diff-tree", "--stdin", "-r", "-z", "-M", "--name-status", input=pairs))
        parent_of = dict(missing)
        step, block = None, []
        for field in out + [""]:
            if field in parent_of or field == "":
                if step is not None:
                    STEP_CHANGES[step] = name_status(block)
                step = (field, parent_of[field]) if field else None
                block = []
            else:
                block.append(field)
        for step in missing:
            STEP_CHANGES.setdefault(step, [])
    return [STEP_CHANGES[step] for step in steps]


def renames_since(root: str, rev: str) -> Optional[dict[str, Optional[str]]]:
    start = resolve(root, rev)
    head = resolve(root, "HEAD")
    if start is None or head is None:
        return None
    steps = [] if start == head else first_parent_steps(root, start)
    if steps is None:
        return None
    trail: dict[str, Optional[str]] = {}
    for changes in step_changes(root, steps):
        trail = follow(trail, changes)
    worktree = split_z(git(root, "diff", "-z", "--name-status", "-M", "--no-ext-diff", "HEAD"))
    return follow(trail, name_status(worktree))


def changes_since(root: str, rev: str) -> dict[str, Optional[str]]:
    fields = split_z(git(root, "diff", "-z", "--name-status", "-M", "--no-ext-diff", rev))
    found: dict[str, Optional[str]] = {}
    for status, old, new in name_status(fields):
        if status != "D":
            found[new] = None if status == "A" else old
    trail = renames_since(root, rev)
    if trail is not None:
        for name in found:
            back = trail.get(name, name)
            if back is not None:
                found[name] = back
    for name in untracked(root):
        found.setdefault(name, None)
    return found


def show(root: str, rev: str, name: Optional[str]) -> str:
    if name is None or rev == EMPTY_TREE:
        return ""
    proc = subprocess.run(
        ["git", "-C", root, "cat-file", "blob", f"{rev}:{name}"], stdin=subprocess.DEVNULL, capture_output=True
    )
    return decode(proc.stdout) if proc.returncode == 0 else ""


def parallel(fn: Callable[[T], R], items: Iterable[T]) -> list[R]:
    items = list(items)
    if len(items) < 2:
        return [fn(item) for item in items]
    with ThreadPoolExecutor(max_workers=min(32, os.cpu_count() or 1)) as pool:
        return list(pool.map(fn, items))
