import bisect
import re
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class Span:
    kind: Literal["line", "block", "hash", "disabled", "warning"]
    start: int
    end: int
    line: int
    text: str


class LineIndex:
    def __init__(self, text: str):
        self.starts = [0] + [m.end() for m in re.finditer("\n", text)]

    def line_of(self, offset: int) -> int:
        return bisect.bisect_right(self.starts, offset)

    def lines_starting_inside(self, start: int, end: int) -> range:
        return range(
            bisect.bisect_right(self.starts, start), bisect.bisect_left(self.starts, end)
        )


@dataclass(frozen=True)
class Scan:
    spans: list[Span]
    refusals: list[tuple[int, str]]
    directives: list[tuple[int, int]]


_NEXT = re.compile(r"""[/"']""")
_RAW_OPEN = re.compile(r'"([^()\\ \t\n]{0,16})\(')
_DOUBLE_QUOTED = re.compile(r'"(?:[^"\\\n]|\\.)*"?', re.S)
_SINGLE_QUOTED = re.compile(r"'(?:[^'\\\n]|\\.)*'?", re.S)
_ENCODING_PREFIXES = ("u8", "u", "U", "L")

IF0 = re.compile(r"^\s*#\s*if\s+\(?\s*(?:0|false)\s*\)?\s*(?:&&.*)?$")
IF1 = re.compile(r"^\s*#\s*if\s+\(?\s*(?:1|true)\s*\)?\s*$")
IF_ANY = re.compile(r"^\s*#\s*if")
ELIF = re.compile(r"^\s*#\s*elif")
ELSE = re.compile(r"^\s*#\s*else\b")
ENDIF = re.compile(r"^\s*#\s*endif\b")
WARNING = re.compile(r"^\s*#\s*warning\b")
TODO_MESSAGE = re.compile(r'^"?\s*(?:TODO|FIXME)\b')


def _identifier_char(c: str) -> bool:
    return c.isalnum() or c == "_"


def _prefix_ends_at(text: str, end: int, prefixes: tuple[str, ...]) -> bool:
    for p in prefixes:
        k = end - len(p)
        if k >= 0 and text[k:end] == p and (k == 0 or not _identifier_char(text[k - 1])):
            return True
    return False


def _raw_string_prefix(text: str, quote: int) -> bool:
    return (
        quote > 0
        and text[quote - 1] == "R"
        and _prefix_ends_at(text, quote - 1, _ENCODING_PREFIXES + ("",))
    )


def _char_prefix(text: str, quote: int) -> bool:
    return _prefix_ends_at(text, quote, _ENCODING_PREFIXES)


def _line_end(text: str, newline: int) -> int:
    if newline < 0:
        return len(text)
    return newline - 1 if newline > 0 and text[newline - 1] == "\r" else newline


def _line_comment_end(text: str, start: int) -> int:
    line_start = start
    while True:
        newline = text.find("\n", line_start)
        if newline >= 0 and text[line_start:newline].rstrip().endswith("\\"):
            line_start = newline + 1
            continue
        return _line_end(text, newline)


def _scan_comments(text: str) -> tuple[list[tuple[str, int, int]], list[tuple[int, int]]]:
    comments: list[tuple[str, int, int]] = []
    literals: list[tuple[int, int]] = []
    n = len(text)
    i = 0
    while (m := _NEXT.search(text, i)) is not None:
        i = m.start()
        c = text[i]
        if c == "/":
            following = text[i + 1 : i + 2]
            if following == "/":
                end = _line_comment_end(text, i)
                comments.append(("line", i, end))
                i = end
            elif following == "*":
                close = text.find("*/", i + 2)
                end = n if close < 0 else close + 2
                comments.append(("block", i, end))
                i = end
            else:
                i += 1
            continue
        if c == '"':
            raw = _RAW_OPEN.match(text, i) if _raw_string_prefix(text, i) else None
            if raw is not None:
                terminator = ")" + raw.group(1) + '"'
                close = text.find(terminator, raw.end())
                end = n if close < 0 else close + len(terminator)
            else:
                end = _DOUBLE_QUOTED.match(text, i).end()
            literals.append((i, end))
            i = end
            continue
        if i > 0 and text[i - 1].isalnum() and not _char_prefix(text, i):
            i += 1
            continue
        end = _SINGLE_QUOTED.match(text, i).end()
        literals.append((i, end))
        i = end
    return comments, literals


def _blank_out(text: str, ranges: list[tuple[str, int, int]]) -> str:
    parts = []
    last = 0
    for _, start, end in ranges:
        parts.append(text[last:start])
        parts.append(re.sub(r"[^\n]", " ", text[start:end]))
        last = end
    parts.append(text[last:])
    return "".join(parts)


@dataclass
class _Guard:
    kind: Literal["if0", "if1", "other"]
    first: int
    parent_live: bool
    dead: bool
    else_line: int | None = None
    elif_line: int | None = None

    @property
    def constant(self) -> bool:
        return self.kind != "other"


def _guard_line_ranges(lines: list[str], skip: set[int]):
    stack: list[_Guard] = []
    ranges: list[tuple[int, int]] = []
    refusals: list[tuple[int, str]] = []
    for number, line in enumerate(lines):
        if number in skip or "#" not in line:
            continue
        if IF_ANY.match(line):
            parent_live = not any(g.dead for g in stack)
            if IF0.match(line):
                stack.append(_Guard("if0", number, parent_live, True))
            elif IF1.match(line):
                stack.append(_Guard("if1", number, parent_live, False))
            else:
                stack.append(_Guard("other", number, parent_live, False))
        elif ELIF.match(line) and stack:
            g = stack[-1]
            if g.constant and g.elif_line is None:
                g.elif_line = number
                g.dead = True
        elif ELSE.match(line) and stack:
            g = stack[-1]
            if g.constant and g.elif_line is None and g.else_line is None:
                g.else_line = number
                g.dead = not g.dead
        elif ENDIF.match(line) and stack:
            g = stack.pop()
            if not (g.constant and g.parent_live):
                continue
            if g.elif_line is not None:
                ranges.append((g.first, number))
                refusals.append(
                    (
                        g.first + 1,
                        f"#elif at line {g.elif_line + 1} under a constant #if: "
                        "resolve this guard by hand",
                    )
                )
            elif g.kind == "if0" and g.else_line is None:
                ranges.append((g.first, number))
            elif g.kind == "if0":
                ranges += [(g.first, g.else_line), (number, number)]
            elif g.else_line is None:
                ranges += [(g.first, g.first), (number, number)]
            else:
                ranges += [(g.first, g.first), (g.else_line, number)]
    return ranges, refusals


def _todo_warning_lines(lines: list[str], skip: set[int]) -> list[tuple[int, int]]:
    found = []
    number = 0
    while number < len(lines):
        first = number
        if number in skip or not WARNING.match(lines[number]):
            number += 1
            continue
        while lines[number].rstrip().endswith("\\") and number + 1 < len(lines):
            number += 1
        directive = "\n".join(lines[first : number + 1])
        message = re.sub(r"\\[ \t]*\r?\n", "", WARNING.sub("", directive, count=1)).strip()
        if TODO_MESSAGE.match(message):
            found.append((first, number))
        number += 1
    return found


def _contains(outer: Span, start: int, end: int) -> bool:
    return outer.start <= start and end <= outer.end


def _crosses(outer: Span, start: int, end: int) -> bool:
    return outer.start < end and start < outer.end and not _contains(outer, start, end)


def _directive_ranges(text: str, blanked: str, opaque: list[tuple[int, int]]) -> list[tuple[int, int]]:
    starts = [start for start, _ in opaque]

    def inside(offset: int) -> bool:
        k = bisect.bisect_right(starts, offset) - 1
        return k >= 0 and offset < opaque[k][1]

    ranges = []
    logical = physical = 0
    for newline in [m.start() for m in re.finditer("\n", text)] + [len(text)]:
        spliced = text[physical:newline].rstrip().endswith("\\")
        physical = newline + 1
        if newline < len(text) and (inside(newline) or spliced):
            continue
        segment = blanked[logical:newline]
        body = segment.lstrip()
        if body.startswith("#"):
            ranges.append((newline - len(body), newline))
        logical = newline + 1
    return ranges


def scan(text: str) -> Scan:
    comments, literals = _scan_comments(text)
    index = LineIndex(text)
    skip: set[int] = set()
    for _, start, end in comments:
        skip.update(index.lines_starting_inside(start, end))
    for start, end in literals:
        skip.update(index.lines_starting_inside(start, end))
    blanked = _blank_out(text, comments)
    opaque = sorted([(start, end) for _, start, end in comments] + literals)
    directives = _directive_ranges(text, blanked, opaque)
    lines = blanked.split("\n")
    line_ranges, refusals = _guard_line_ranges(lines, skip)

    def line_offset(number: int) -> int:
        return index.starts[number] if number < len(index.starts) else len(text)

    disabled = []
    for first, last in line_ranges:
        start, end = line_offset(first), line_offset(last + 1)
        disabled.append(Span("disabled", start, end, first + 1, text[start:end]))

    warnings = []
    for first, last in _todo_warning_lines(lines, skip):
        start = line_offset(first) + lines[first].index("#")
        end = _line_end(text, text.find("\n", line_offset(last)))
        if not any(_contains(d, start, end) for d in disabled):
            warnings.append(Span("warning", start, end, first + 1, text[start:end]))

    spans = disabled + warnings
    for kind, start, end in comments:
        if any(_contains(s, start, end) for s in disabled + warnings):
            continue
        line = index.line_of(start)
        if any(_crosses(d, start, end) for d in disabled):
            refusals.append((line, "comment crosses the edge of a constant #if guard"))
        if any(_crosses(w, start, end) for w in warnings):
            refusals.append((line, "comment crosses the end of a #warning directive"))
        spans.append(Span(kind, start, end, line, text[start:end]))
    spans.sort(key=lambda s: s.start)
    refusals.sort()
    return Scan(spans, refusals, directives)


def lex(text: str) -> list[Span]:
    return scan(text).spans


def guard_shape(span: Span) -> tuple[bool, bool]:
    lines = span.text.rstrip("\n").split("\n")
    closes = bool(ELSE.match(lines[0]) or ENDIF.match(lines[0]))
    opens = not ENDIF.match(lines[-1])
    return closes, opens


def guard_groups(disabled: list[Span]) -> list[tuple[Span, ...]]:
    groups: list[list[Span]] = []
    open_groups: list[list[Span]] = []
    for span in sorted(disabled, key=lambda s: s.start):
        closes, opens = guard_shape(span)
        if closes and open_groups:
            current = open_groups.pop()
            current.append(span)
        else:
            current = [span]
            groups.append(current)
        if opens:
            open_groups.append(current)
    return [tuple(g) for g in groups]
