from __future__ import annotations

from rules import Violation
from tokens import DIRECTIVE_PLACEMENTS

RULE = (
    'Thorn files allow no comments (see "Code comments" in CLAUDE.md).\n'
    "Put a true invariant in code (assert, name, CCTK_VERROR) or in the commit message."
)
EXCERPT_WIDTH = 72


def excerpt(v: Violation) -> str:
    if v.region.placement in DIRECTIVE_PLACEMENTS:
        code = v.region.code
        return code[:EXCERPT_WIDTH] + ("…" if len(code) > EXCERPT_WIDTH else "")
    lines = v.token.text.strip().splitlines() or [""]
    first = lines[0].strip()
    more = len(lines) > 1 or len(first) > EXCERPT_WIDTH or v.region.text.strip() != v.token.text.strip()
    return first[:EXCERPT_WIDTH] + ("…" if more else "")


def summary(count: int) -> str:
    mark = "✗" if count else "✓"
    noun = "comment" if count == 1 else "comments"
    return f"{mark} no-comments: {count} {noun}"


def render(found: list[tuple[str, Violation]]) -> str:
    if not found:
        return summary(0)
    places = [f"{path}:{v.line}" for path, v in found]
    width = max(len(p) for p in places)
    out = [f"{place.ljust(width)}   {excerpt(v)}" for place, (_, v) in zip(places, found)]
    out.extend(RULE.split("\n"))
    out.append(summary(len(found)))
    return "\n".join(out)
