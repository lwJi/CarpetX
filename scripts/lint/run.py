from __future__ import annotations

import os
from typing import Callable, Iterable, Iterator, Sequence

from checks import Check
from files import (
    PROJECT,
    Lang,
    changes_since,
    covered,
    is_regular,
    list_covered,
    list_tree,
    merge_base,
    merged_tips,
    parallel,
    read_text,
    show,
)

Found = list[tuple[Check, list[tuple[str, object]]]]


class FileFailed(RuntimeError):
    pass


def starts(root: str, base: str) -> list[str]:
    return [base] + merged_tips(root, base)


def gather(checks: Sequence[Check], entries: Iterable[tuple[Check, str, object]]) -> Found:
    entries = list(entries)
    return [(check, [(name, f) for c, name, f in entries if c is check]) for check in checks]


def has_findings(found: Found) -> bool:
    return any(entries for _, entries in found)


def render(found: Found, every: bool = True) -> str:
    return "\n\n".join(check.render(entries) for check, entries in found if entries or every)


def render_change(found: Found, edit: bool = False, every: bool = False) -> str:
    return "\n\n".join(check.render_change(entries, edit) for check, entries in found if entries or every)


def check_tree(paths: list[str], checks: Sequence[Check], root: str = PROJECT) -> Found:
    files = list_covered(paths, root) if paths else list_tree(root)

    def listed(item: tuple[str, str, Lang]) -> list[tuple[Check, str, object]]:
        name, path, lang = item
        try:
            text = read_text(path)
            return [
                (check, name, f) for check in checks if check.covers(name, lang) for f in check.found(name, text, lang)
            ]
        except (RuntimeError, UnicodeError, OSError) as e:
            raise FileFailed(f"{name}: {e}") from e

    return gather(checks, (entry for entries in parallel(listed, files) for entry in entries))


def judged(
    name: str, befores: Callable[[], Iterator[tuple[str, str]]], after: str, lang: Lang, checks: Sequence[Check]
) -> list[tuple[Check, str, object]]:
    live = [check for check in checks if check.covers(name, lang)]
    verdicts: dict = {}
    for old_name, old in befores():
        for check in live:
            verdict = check.judge(name, old_name, old, after, lang)
            verdicts[check] = verdict if check not in verdicts else check.meet(verdicts[check], verdict)
        live = [check for check in live if verdicts[check]]
        if not live:
            break
    return [(check, name, verdicts[check]) for check in checks if verdicts.get(check)]


def check_since(root: str, base: str, checks: Sequence[Check]) -> Found:
    revs = starts(root, base)
    renames = [changes_since(root, rev) for rev in revs]
    names = sorted(set(renames[0]).intersection(*renames[1:]))
    items = [(name, lang) for name in names for lang in [covered(name, root)] if lang is not None]

    def judge(item: tuple[str, Lang]) -> list[tuple[Check, str, object]]:
        name, lang = item
        path = os.path.join(root, name)
        after = read_text(path) if is_regular(path) else ""

        def befores() -> Iterator[tuple[str, str]]:
            for rev, old in zip(revs, renames):
                yield old[name] or name, show(root, rev, old[name])

        return judged(name, befores, after, lang, checks)

    return gather(checks, (entry for entries in parallel(judge, items) for entry in entries))


def check_edit(root: str, name: str, before: str, after: str, lang: Lang, checks: Sequence[Check]) -> Found:
    def befores() -> Iterator[tuple[str, str]]:
        yield name, before
        try:
            olds = [(rev, changes_since(root, rev).get(name, name)) for rev in starts(root, merge_base(root))]
            starting = [(old or name, show(root, rev, old)) for rev, old in olds]
        except Exception:
            return
        yield from starting

    return gather(checks, judged(name, befores, after, lang, checks))
