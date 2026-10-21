from __future__ import annotations

from rules import Unit

RULE = (
    "Don't add comments to thorn files (see \"Code comments\" in CLAUDE.md).\n"
    "Put a true invariant in code (assert, name, CCTK_VERROR) or in the commit message."
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
