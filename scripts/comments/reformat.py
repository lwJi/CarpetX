from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from typing import Optional

from files import Lang, blobs, decode, git, is_thorn, lang_of, parallel, read_text, split_z
from lex_clang import ToolMissing
from verify import changed_files, verify_layout_cxx

CLANG_FORMAT = "clang-format"
PINNED = "22.1.1"
VERSION = re.compile(r"\bclang-format version (\d+\.\d+\.\d+)")
HINT = (f"install it with `python3 -m pip install clang-format=={PINNED}` (the PyPI wheel) "
        "and put its bin directory first on PATH")


class ReformatError(RuntimeError):
    pass


def version_problem(output: str) -> Optional[str]:
    first = output.strip().split("\n")[0] if output.strip() else "no version"
    found = VERSION.search(first)
    if first.startswith("Apple") or found is None or found.group(1) != PINNED:
        return first
    return None


def clang_format() -> str:
    path = shutil.which(CLANG_FORMAT)
    if path is None:
        raise ToolMissing(f"{CLANG_FORMAT} not found on PATH; reformat needs clang-format {PINNED}: {HINT}")
    proc = subprocess.run([path, "--version"], stdin=subprocess.DEVNULL, capture_output=True)
    found = version_problem(decode(proc.stdout))
    if proc.returncode != 0 or found is not None:
        raise ToolMissing(f"reformat needs clang-format {PINNED}, but {path} reports {found!r}: {HINT}")
    return path


def style(root: str) -> str:
    return "--style=file:" + os.path.join(root, ".clang-format")


def formatted(exe: str, root: str, name: str, text: str) -> str:
    proc = subprocess.run(
        [exe, style(root), f"--assume-filename={name}"],
        input=text.encode("utf-8", "surrogateescape"),
        capture_output=True,
    )
    if proc.returncode != 0:
        why = decode(proc.stderr).strip().split("\n")[0]
        raise ReformatError(f"{CLANG_FORMAT} exited with status {proc.returncode} on {name}: {why}")
    return decode(proc.stdout)


def strip_commit(root: str, thorn: str) -> str:
    subject = f"{thorn}: remove comments"
    out = split_z(git(root, "log", "-z", "--format=%H %s", "--fixed-strings", f"--grep={subject}", "HEAD"))
    for entry in out:
        sha, _, found = entry.strip("\n").partition(" ")
        if found == subject:
            return sha
    raise ReformatError(f"no commit with the subject {subject!r} in the history of HEAD")


def thorn_name(root: str, arg: str) -> str:
    name = arg.rstrip("/")
    if ("/" in name or not is_thorn(root, name)) and os.path.exists(arg):
        name = os.path.relpath(os.path.realpath(arg), os.path.realpath(root)).replace(os.sep, "/")
    if "/" in name or name.startswith(".") or not is_thorn(root, name):
        raise ReformatError(f"{arg} is not a thorn (a top-level directory with an interface.ccl)")
    return name


@dataclass(frozen=True)
class Plan:
    thorn: str
    commit: str
    subject: str
    reformat: list
    unformatted: list
    clean: list


def plan(root: str, thorn: str, exe: str) -> Plan:
    sha = strip_commit(root, thorn)
    subject = decode(git(root, "log", "-1", "--format=%s", sha)).strip()
    names = [name for status, name in changed_files(root, sha + "^", sha)
             if status == "M" and name.startswith(thorn + "/") and lang_of(name) is Lang.CXX]
    old = blobs(root, sha + "^", names)

    def judge(name: str) -> tuple[str, str]:
        if formatted(exe, root, name, old[name]) != old[name]:
            return name, "unformatted"
        path = os.path.join(root, name)
        if not os.path.isfile(path):
            raise ReformatError(f"{name} is gone from the working tree")
        text = read_text(path)
        return name, "clean" if formatted(exe, root, name, text) == text else "reformat"

    verdicts = parallel(judge, names)
    by = {kind: [name for name, verdict in verdicts if verdict == kind] for kind in ("reformat", "unformatted", "clean")}
    return Plan(thorn, sha, subject, by["reformat"], by["unformatted"], by["clean"])


def write_text(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8", errors="surrogateescape", newline="") as f:
        f.write(text)


def apply(root: str, exe: str, name: str) -> None:
    path = os.path.join(root, name)
    before = read_text(path)
    proc = subprocess.run([exe, "-i", style(root), "--", path], stdin=subprocess.DEVNULL, capture_output=True)
    if proc.returncode != 0:
        write_text(path, before)
        why = decode(proc.stderr).strip().split("\n")[0]
        raise ReformatError(f"{CLANG_FORMAT} -i exited with status {proc.returncode} on {name}: {why}")
    result = verify_layout_cxx(name, before, read_text(path))
    if not result.ok:
        write_text(path, before)
        raise ReformatError(f"{name}: {CLANG_FORMAT} changed more than layout, so it is left as it was: {result.why}")


def reformat(root: str, args: list[str], out) -> list[Plan]:
    exe = clang_format()
    thorns = [thorn_name(root, arg) for arg in args]
    plans = [plan(root, thorn, exe) for thorn in thorns]
    for p in plans:
        for name in p.reformat:
            apply(root, exe, name)
            out(name)
        files = len(p.reformat) + len(p.unformatted) + len(p.clean)
        out(f"{p.thorn}: reformatted {len(p.reformat)} of the {files} C/C++ files {p.commit[:12]} "
            f"({p.subject}) changed, with clang-format {PINNED}; {len(p.unformatted)} were not "
            f"clang-format-clean before it and stay as they are")
    return plans
