from __future__ import annotations

from checks import excerpt, plural

from .judge import Verdict
from .rules import Unit

RULE = (
    "Comments in thorn files can only be deleted (see agent_docs/code-comments.md).\n"
    "Put what is still true in code (assert, name, CCTK_VERROR) or in the commit message."
)
CHANGE_RULE = (
    "new: a comment line the change adds; take it out.\n"
    "touched: the change edits the code under this comment or deletes some of its lines; delete the whole comment.\n"
    + RULE
)
REDO = "Redo the edit so that it adds none of the new lines and deletes each touched comment whole."


def summary(count: int) -> str:
    mark = "✗" if count else "✓"
    return f"{mark} no-comments: {plural(count, 'comment line')}"


def render(found: list[tuple[str, Unit]]) -> str:
    if not found:
        return summary(0)
    places = [f"{path}:{u.line}" for path, u in found]
    width = max(len(p) for p in places)
    out = [f"{place.ljust(width)}   {excerpt(u.text)}" for place, (_, u) in zip(places, found)]
    out.extend(RULE.split("\n"))
    out.append(summary(len(found)))
    return "\n".join(out)


def change_summary(found: list[tuple[str, Verdict]]) -> str:
    new = sum(len(v.added) for _, v in found)
    touched = sum(len(v.touched) for _, v in found)
    if not new and not touched:
        return "✓ no-comments: no new or touched comments"
    parts = ([plural(new, "new comment line")] if new else []) + ([plural(touched, "touched comment")] if touched else [])
    return "✗ no-comments: " + ", ".join(parts)


def render_change(found: list[tuple[str, Verdict]], edit: bool = False) -> str:
    entries = []
    for path, verdict in found:
        entries.extend((path, u.line, f"{path}:{u.line}", "new", u) for u in verdict.added)
        entries.extend(
            (path, t.first, f"{path}:{t.first}" + (f"-{t.last}" if t.last > t.first else ""), "touched", t.unit)
            for t in verdict.touched
        )
    if not entries:
        return change_summary(found)
    entries.sort(key=lambda e: (e[0], e[1]))
    width = max(len(e[2]) for e in entries)
    out = [f"{place.ljust(width)}   {kind.ljust(7)}   {excerpt(u.text)}" for _, _, place, kind, u in entries]
    out.extend(CHANGE_RULE.split("\n"))
    if edit:
        out.append(REDO)
    out.append(change_summary(found))
    return "\n".join(out)
