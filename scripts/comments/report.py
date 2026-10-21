from __future__ import annotations

from rules import Unit

RULE = (
    "Don't add comments to thorn files (see \"Code comments\" in CLAUDE.md).\n"
    "Put a true invariant in code (assert, name, CCTK_VERROR) or in the commit message."
)
TRIM_RULE = (
    "Read what is left: without the deleted lines it can say something new. If it does, delete the whole comment\n"
    "and put what is still true in code (assert, name, CCTK_VERROR) or in the commit message.\n"
    "A line that stands alone, such as a TODO, a \\param entry or commented-out code, can go by itself."
)
EXCERPT_WIDTH = 72


def excerpt(u: Unit) -> str:
    return u.text[:EXCERPT_WIDTH] + ("…" if len(u.text) > EXCERPT_WIDTH else "")


def summary(count: int) -> str:
    mark = "✗" if count else "✓"
    noun = "comment line" if count == 1 else "comment lines"
    return f"{mark} no-comments: {count} {noun}"


def render(found: list[tuple[str, Unit]]) -> str:
    if not found:
        return summary(0)
    places = [f"{path}:{u.line}" for path, u in found]
    width = max(len(p) for p in places)
    out = [f"{place.ljust(width)}   {excerpt(u)}" for place, (_, u) in zip(places, found)]
    out.extend(RULE.split("\n"))
    out.append(summary(len(found)))
    return "\n".join(out)


def trim_head(path: str, count: int) -> str:
    what = "a comment" if count == 1 else f"{count} comments"
    return f"This edit deletes some lines of {what} in {path} and keeps the rest (see \"Code comments\" in CLAUDE.md):"


def render_trims(path: str, trims: list[list[Unit]]) -> str:
    out = [trim_head(path, len(trims))]
    for i, gone in enumerate(trims):
        if i:
            out.append("")
        out.extend(f"  {excerpt(u)}" for u in gone)
    out.extend(TRIM_RULE.split("\n"))
    return "\n".join(out)
