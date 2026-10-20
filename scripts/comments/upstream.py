from __future__ import annotations

import os
import re
import subprocess
import tempfile
from typing import Callable, Optional

from files import decode, git, list_covered, parallel, read_text, resolve, show
from lex_clang import LexError
from strip import StripError, strip
from verify import report, verify_revs

IGNORE_REVS = ".git-blame-ignore-revs"
FULL_SHA = re.compile(r"[0-9a-f]{40}|[0-9a-f]{64}")
REFORMAT_SUBJECT = re.compile(r"[A-Za-z0-9_]+: clang-format after removing comments")
UPSTREAM = "EinsteinToolkit/CarpetX main"
TRAILER = "Comments-Stripped-From"
TRAILER_GREP = f"^{TRAILER}: [0-9a-f]{{40}}"
TRAILER_LINE = re.compile(rf"^{TRAILER}: ([0-9a-f]{{40}}|[0-9a-f]{{64}})[ \t]*$", re.M)


class UpstreamError(RuntimeError):
    pass


def stripped_from(root: str, rev: str) -> Optional[str]:
    found = TRAILER_LINE.findall(decode(git(root, "log", "-1", "--format=%B", rev)))
    return found[-1] if found else None


def newest_strip(root: str, head: str = "HEAD") -> Optional[str]:
    out = git(root, "log", "--date-order", "-1", "--format=%H", "--extended-regexp", f"--grep={TRAILER_GREP}", head,
              check=False)
    return decode(out).strip() or None


def listed(text: str) -> list[str]:
    revs = []
    for line in text.splitlines():
        rev = line.split("#", 1)[0].strip()
        if rev and rev not in revs:
            revs.append(rev)
    return revs


def is_ancestor(root: str, rev: str, head: str) -> bool:
    proc = subprocess.run(
        ["git", "-C", root, "merge-base", "--is-ancestor", rev, head], stdin=subprocess.DEVNULL, capture_output=True
    )
    return proc.returncode == 0


def refusal(root: str, rev: str, head: str) -> Optional[str]:
    if not FULL_SHA.fullmatch(rev):
        return "is not a full commit SHA"
    if resolve(root, rev) != rev:
        return "is not a commit in this repository"
    if not is_ancestor(root, rev, head):
        return "is not in this branch's history"
    first = resolve(root, rev + "^1")
    if first is None:
        return "has no parent to compare against"
    source = stripped_from(root, rev)
    if source is not None and first != source:
        return f"has a first parent {first[:12]} that is not the stripped ET commit its {TRAILER}: trailer names"
    return None


def verify_skips(root: str, base: str, flesh: Optional[str], out: Callable[[str], None]) -> bool:
    base_sha = resolve(root, base)
    if base_sha is None:
        raise ValueError(f"no such commit: {base}")
    head = "HEAD"
    old = set(listed(show(root, base_sha, IGNORE_REVS)))
    new = [rev for rev in listed(show(root, head, IGNORE_REVS)) if rev not in old]
    if not new:
        out(f"✓ verify-skips {base}: {IGNORE_REVS} lists no new commits")
        return True
    failed = 0
    for rev in new:
        why = refusal(root, rev, head)
        subject = decode(git(root, "log", "-1", "--format=%s", rev)).strip() if why is None else ""
        if why is not None:
            out(f"FAIL  {rev} {why}")
            failed += 1
            continue
        out(f"{rev[:12]}  {subject}")
        source = stripped_from(root, rev)
        if source is not None:
            out(f"  first parent {source[:12]} is the stripped ET commit its {TRAILER}: trailer names")
        layout = REFORMAT_SUBJECT.fullmatch(subject) is not None
        what = f"verify{' --layout' if layout else ''} {rev[:12]}^1 {rev[:12]}"
        results = verify_revs(root, rev + "^1", rev, flesh, layout=layout)
        ok = report(results, what, lambda line: out("  " + line), "layout" if layout else "comments")
        failed += not ok
    noun = "commit" if len(new) == 1 else "commits"
    if failed:
        out(f"✗ verify-skips {base}: {failed} of {len(new)} new blame-skip {noun} not proven comment- or layout-only")
    else:
        out(f"✓ verify-skips {base}: {len(new)} new blame-skip {noun}, each comment- or layout-only")
    return not failed


def source_of(root: str, prev: str) -> str:
    source = stripped_from(root, prev)
    if source is None:
        raise UpstreamError(f"{prev[:12]} has a {TRAILER}: line that git log --grep matches but no full SHA in it")
    return source


def strip_worktree(work: str, out: Callable[[str], None]) -> None:
    def stripped(item):
        name, path, lang = item
        try:
            text = read_text(path)
            return path, text, strip(text, lang)
        except (LexError, StripError, UnicodeError, OSError) as e:
            raise UpstreamError(f"{name}: {e}") from e

    results = parallel(stripped, list_covered([work], root=work))
    changed = 0
    for path, text, new in results:
        if new != text:
            with open(path, "w", encoding="utf-8", errors="surrogateescape", newline="") as f:
                f.write(new)
            changed += 1
    out(f"stripped {changed} of {len(results)} covered files")


def stripped_tree(root: str, et: str, out: Callable[[str], None]) -> str:
    with tempfile.TemporaryDirectory(prefix="comments-upstream-") as tmp:
        work = os.path.join(tmp, "worktree")
        git(root, "worktree", "add", "--quiet", "--detach", work, et)
        try:
            strip_worktree(work, out)
            git(work, "add", "--update")
            return decode(git(work, "write-tree")).strip()
        finally:
            git(root, "worktree", "remove", "--force", work, check=False)


def strip_upstream(root: str, ref: str, seed: bool, flesh: Optional[str], out: Callable[[str], None]) -> Optional[str]:
    et = resolve(root, ref)
    if et is None:
        raise UpstreamError(f"no such commit: {ref}")
    prev = None if seed else newest_strip(root)
    if not seed and prev is None:
        raise UpstreamError(f"no commit in the history of HEAD has a {TRAILER}: trailer; "
                            f"seed the lineage with strip-upstream --seed")
    if prev is not None:
        source = source_of(root, prev)
        if is_ancestor(root, et, source):
            out(f"{ref} ({et[:12]}) has nothing new: HEAD already holds {prev[:12]}, which strips {source[:12]}")
            return prev
    short = decode(git(root, "rev-parse", "--short", et)).strip()
    out(f"stripping {UPSTREAM} {short}" + (f" onto {prev[:12]}" if prev else " to seed the lineage"))
    tree = stripped_tree(root, et, out)
    parents = ["-p", et] + (["-p", prev] if prev else [])
    message = ["-m", f"Strip comments from {UPSTREAM} {short}", "-m", f"{TRAILER}: {et}"]
    s = decode(git(root, "commit-tree", tree, *parents, *message)).strip()
    if not report(verify_revs(root, et, s, flesh), f"verify {et[:12]} {s[:12]}", out):
        out(f"✗ strip-upstream {ref}: {s[:12]} changed more than comments, so it is not printed")
        return None
    return s
