from __future__ import annotations

import re
import subprocess
from typing import Callable, Optional

from files import decode, git, resolve, show
from verify import report, verify_revs

IGNORE_REVS = ".git-blame-ignore-revs"
FULL_SHA = re.compile(r"[0-9a-f]{40}|[0-9a-f]{64}")


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
    if resolve(root, rev + "^1") is None:
        return "has no parent to compare against"
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
        ok = report(verify_revs(root, rev + "^1", rev, flesh), f"verify {rev[:12]}^1 {rev[:12]}", lambda line: out("  " + line))
        failed += not ok
    noun = "commit" if len(new) == 1 else "commits"
    if failed:
        out(f"✗ verify-skips {base}: {failed} of {len(new)} new blame-skip {noun} not proven comment-only")
    else:
        out(f"✓ verify-skips {base}: {len(new)} new blame-skip {noun}, each comment-only")
    return not failed
