from rules import Violation

RULE = (
    "Comments are not allowed in thorn sources, .ccl or .par files. Allowed: namespace\n"
    "closers, formatter/lint markers, paper and URL citations. Put the explanation in\n"
    "names, assertions, error messages or tests, and the rest in the commit message."
)
EXCERPT_WIDTH = 72


def excerpt(v: Violation) -> str:
    lines = [line.strip() for line in v.group.text.split("\n")]
    first = lines[0]
    more = len(lines) > 1 or len(first) > EXCERPT_WIDTH
    return first[:EXCERPT_WIDTH] + ("…" if more else "")


def summary(count: int) -> str:
    mark = "✗" if count else "✓"
    noun = "comment" if count == 1 else "comments"
    return f"{mark} no-comments: {count} {noun} outside the allowed exceptions"


def report(found: list[tuple[str, Violation]], full: bool = False) -> str:
    if not found:
        return summary(0)
    out = []
    for path, v in found:
        if full:
            out.append(f"  {path}:{v.line}")
            out.extend(f"    {line}" for line in v.group.text.split("\n"))
        else:
            out.append(f"  {path}:{v.line}   {excerpt(v)}")
    out.append("")
    out.extend(f"  {line}" for line in RULE.split("\n"))
    out.append("")
    out.append(summary(len(found)))
    return "\n".join(out)
