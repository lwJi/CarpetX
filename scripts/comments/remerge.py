import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass

from files import covered, git_toplevel, read_text, write_text
from strip import Refused, strip

STAGES = {1: "base", 2: "ours", 3: "theirs"}
REGULAR_FILE = {"100644", "100755"}
CONFLICT_MARKER = re.compile(r"^<{7,}(?: |$)", re.M)
MAX_CONFLICTS = 127


@dataclass(frozen=True)
class Entry:
    mode: str
    oid: str


@dataclass(frozen=True)
class Outcome:
    path: str
    status: str
    detail: str = ""


def _git(top: str, *args: str) -> bytes:
    return subprocess.run(["git", "-C", top, *args], capture_output=True, check=True).stdout


def unmerged(top: str) -> dict[str, dict[int, Entry]]:
    records = _git(top, "ls-files", "-u", "-z").decode("utf-8", "surrogateescape").split("\0")
    found: dict[str, dict[int, Entry]] = {}
    for record in records:
        if not record:
            continue
        info, _, path = record.partition("\t")
        mode, oid, stage = info.split(" ")
        found.setdefault(path, {})[int(stage)] = Entry(mode, oid)
    return found


def theirs_label(top: str) -> str:
    probe = subprocess.run(["git", "-C", top, "rev-parse", "-q", "--verify", "MERGE_HEAD"], capture_output=True)
    return "MERGE_HEAD" if probe.returncode == 0 else "theirs"


def merge_file(top: str, base: str, ours: str, theirs: str, labels: tuple[str, str, str]) -> tuple[str, int]:
    with tempfile.TemporaryDirectory() as tmp:
        paths = []
        for name, text in (("ours", ours), ("base", base), ("theirs", theirs)):
            path = os.path.join(tmp, name)
            write_text(path, text)
            paths.append(path)
        flags = [arg for label in labels for arg in ("-L", label)]
        result = subprocess.run(["git", "-C", top, "merge-file", "-p", *flags, *paths], capture_output=True)
    if not 0 <= result.returncode <= MAX_CONFLICTS:
        raise RuntimeError(f"git merge-file failed: {result.stderr.decode(errors='replace').strip()}")
    return result.stdout.decode("utf-8", "surrogateescape"), result.returncode


def remerge_path(top: str, path: str, stages: dict[int, Entry], labels: tuple[str, str, str]) -> Outcome:
    lang = covered(path)
    if lang is None:
        return Outcome(path, "skipped", "not a covered file")
    missing = [STAGES[n] for n in STAGES if n not in stages]
    if missing:
        return Outcome(path, "skipped", f"no {' or '.join(missing)} version")
    if any(e.mode not in REGULAR_FILE for e in stages.values()):
        return Outcome(path, "skipped", "not a regular file on every side")
    worktree = os.path.join(top, path)
    if not os.path.isfile(worktree) or not CONFLICT_MARKER.search(read_text(worktree)):
        return Outcome(path, "skipped", "no conflict markers in the working tree file")
    texts = {}
    for n, side in STAGES.items():
        text = _git(top, "cat-file", "blob", stages[n].oid).decode("utf-8", "surrogateescape")
        try:
            texts[side] = strip(text, lang)
        except Refused as e:
            return Outcome(path, "skipped", f"strip refused the {side} version at line {e.line}: {e.reason}")
    merged, conflicts = merge_file(top, texts["base"], texts["ours"], texts["theirs"], labels)
    write_text(worktree, merged)
    if conflicts:
        noun = "conflict" if conflicts == 1 else "conflicts"
        more = "+" if conflicts == MAX_CONFLICTS else ""
        return Outcome(path, "conflict", f"{conflicts}{more} {noun} left")
    _git(top, "add", "--", path)
    return Outcome(path, "merged")


def remerge(out=sys.stdout) -> int:
    top = git_toplevel(".")
    if top is None:
        print("remerge: not inside a git repository", file=sys.stderr)
        return 2
    entries = unmerged(top)
    if not entries:
        print("remerge: no unmerged paths", file=out)
        return 0
    labels = ("HEAD", "merge base, stripped", theirs_label(top))
    outcomes = [remerge_path(top, path, entries[path], labels) for path in sorted(entries)]
    width = max(len(o.path) for o in outcomes)
    for o in outcomes:
        print(f"{o.status:<8}  {o.path:<{width}}  {o.detail}".rstrip(), file=out)
    merged = sum(o.status == "merged" for o in outcomes)
    left = len(outcomes) - merged
    print(f"remerge: {merged} of {len(outcomes)} unmerged paths merged, {left} left to resolve by hand", file=out)
    return 1 if left else 0
