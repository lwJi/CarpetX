from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Optional

from tokens import Kind, Placement, Region, Token

CODE_KINDS = (Kind.CODE, Kind.DIRECTIVE)
NUMBER = re.compile(r"0[xX][0-9a-fA-F]+|0[0-7]*|[1-9][0-9]*")
NAME = re.compile(r"[A-Za-z_]\w*")
BINARY = (
    ("||",),
    ("&&",),
    ("|",),
    ("^",),
    ("&",),
    ("==", "!="),
    ("<=", ">=", "<", ">"),
    ("<<", ">>"),
    ("+", "-"),
    ("*", "/", "%"),
)
OPENERS = ("if", "ifdef", "ifndef")
ALTERNATIVES = ("elif", "elifdef", "elifndef")
STRING_PREFIX = re.compile(r'(?:u8|u|U|L)?(?:R"([^(]*)\(|")')
PRAGMA_MESSAGE = re.compile(r"\s*(?:message\b|GCC\s+warning\b)")


def placements(tokens: list[Token]) -> dict[int, Placement]:
    found = {}
    line: list[int] = []
    for i, token in enumerate(tokens + [Token(Kind.NEWLINE, "", -1, -1)]):
        if token.kind is not Kind.NEWLINE:
            line.append(i)
            continue
        code = [k for k in line if tokens[k].kind in CODE_KINDS]
        for k in line:
            if tokens[k].kind is not Kind.COMMENT:
                continue
            before = bool(code) and code[0] < k
            after = bool(code) and code[-1] > k
            if after:
                found[k] = Placement.INLINE
            elif before:
                found[k] = Placement.TRAILING
            else:
                found[k] = Placement.WHOLE_LINE
        line = []
    return found


def is_separator(token: Token) -> bool:
    return token.text.rstrip() == "//"


def continues_run(tokens: list[Token], last: int, nxt: int) -> bool:
    gap = tokens[last + 1 : nxt]
    if any(t.kind not in (Kind.SPACE, Kind.NEWLINE) for t in gap):
        return False
    return sum(t.kind is Kind.NEWLINE for t in gap) <= 1


def comment_regions(tokens: list[Token], standalone: Callable[[Token], bool] = lambda t: False) -> list[Region]:
    def region(placement: Placement, first: int, last: int) -> Region:
        return Region(placement, first, last, tuple(tokens[first : last + 1]))

    found = []
    run: list[int] = []
    for k, placement in sorted(placements(tokens).items()):
        if placement is Placement.WHOLE_LINE and is_separator(tokens[k]):
            if run:
                found.append(region(Placement.WHOLE_LINE, run[0], run[-1]))
                run = []
            continue
        joins = placement is Placement.WHOLE_LINE and not standalone(tokens[k])
        if run and joins and continues_run(tokens, run[-1], k):
            run.append(k)
            continue
        if run:
            found.append(region(Placement.WHOLE_LINE, run[0], run[-1]))
            run = []
        if joins:
            run = [k]
        else:
            found.append(region(placement, k, k))
    if run:
        found.append(region(Placement.WHOLE_LINE, run[0], run[-1]))
    return found


class Undecidable(Exception):
    pass


def apply(op: str, a: Optional[int], b: Optional[int]) -> Optional[int]:
    if op == "||":
        if (a is not None and a != 0) or (b is not None and b != 0):
            return 1
        return None if a is None or b is None else 0
    if op == "&&":
        if a == 0 or b == 0:
            return 0
        return None if a is None or b is None else 1
    if op in ("/", "%"):
        if b is None or b == 0:
            raise Undecidable
        if a is None:
            return None
        quotient = abs(a) // abs(b) * (1 if (a < 0) == (b < 0) else -1)
        return quotient if op == "/" else a - b * quotient
    if a is None or b is None:
        return None
    if op in ("<<", ">>"):
        if b < 0 or b > 4096:
            raise Undecidable
        return a << b if op == "<<" else a >> b
    return {
        "|": lambda: a | b,
        "^": lambda: a ^ b,
        "&": lambda: a & b,
        "==": lambda: int(a == b),
        "!=": lambda: int(a != b),
        "<=": lambda: int(a <= b),
        ">=": lambda: int(a >= b),
        "<": lambda: int(a < b),
        ">": lambda: int(a > b),
        "+": lambda: a + b,
        "-": lambda: a - b,
        "*": lambda: a * b,
    }[op]()


def binary(words: list[str], k: int, level: int = 0) -> tuple[Optional[int], int]:
    if level == len(BINARY):
        return unary(words, k)
    left, k = binary(words, k, level + 1)
    while k < len(words) and words[k] in BINARY[level]:
        op = words[k]
        right, k = binary(words, k + 1, level + 1)
        left = apply(op, left, right)
    return left, k


def unary(words: list[str], k: int) -> tuple[Optional[int], int]:
    if k >= len(words):
        raise Undecidable
    word = words[k]
    if word in ("!", "~", "-"):
        value, k = unary(words, k + 1)
        if value is None:
            return None, k
        return (int(not value) if word == "!" else ~value if word == "~" else -value), k
    if word == "(":
        value, k = binary(words, k + 1)
        if words[k : k + 1] != [")"]:
            raise Undecidable
        return value, k + 1
    if NUMBER.fullmatch(word):
        base = 16 if word[:2] in ("0x", "0X") else 8 if word.startswith("0") else 10
        return int(word, base), k + 1
    if word == "defined":
        if k + 1 < len(words) and NAME.fullmatch(words[k + 1]):
            return None, k + 2
        if words[k + 1 : k + 2] == ["("]:
            if k + 3 < len(words) and NAME.fullmatch(words[k + 2]) and words[k + 3] == ")":
                return None, k + 4
            raise Undecidable
        return None, k + 1
    if NAME.fullmatch(word):
        k += 1
        if words[k : k + 1] == ["("]:
            depth = 0
            while k < len(words):
                depth += {"(": 1, ")": -1}.get(words[k], 0)
                k += 1
                if depth == 0:
                    break
            if depth != 0:
                raise Undecidable
        return None, k
    raise Undecidable


def constant_value(words: list[str]) -> Optional[int]:
    try:
        value, k = binary(words, 0)
    except (Undecidable, RecursionError):
        return None
    return value if k == len(words) else None


@dataclass
class Directive:
    first: int
    last: int
    words: list
    spliced: bool

    @property
    def name(self) -> str:
        return self.words[1] if len(self.words) > 1 else ""


def directives(tokens: list[Token]) -> list[Directive]:
    found: list[Directive] = []
    current: Optional[Directive] = None
    for i, token in enumerate(tokens):
        if token.kind is Kind.NEWLINE:
            current = None
        elif token.kind is Kind.DIRECTIVE:
            if current is None:
                current = Directive(i, i, [], False)
                found.append(current)
            current.last = i
            current.words.append(token.text)
        elif token.kind is Kind.SPACE and current is not None and "\\" in token.text:
            current.spliced = True
    return found


@dataclass
class Frame:
    outer_dead: bool
    taken: bool
    dead: bool


def is_message(d: Directive) -> bool:
    words = d.words
    if d.name == "warning":
        return True
    return d.name == "pragma" and (words[2:3] == ["message"] or words[2:4] == ["GCC", "warning"])


def condition(d: Directive) -> Optional[int]:
    if d.words[0] != "#" or d.spliced or d.name not in ("if", "elif"):
        return None
    return constant_value(d.words[2:])


def scan_directives(tokens: list[Token]) -> tuple[list[Region], list[tuple[int, int]]]:
    def region(placement: Placement, first: int, last: int) -> Region:
        return Region(placement, first, last, tuple(tokens[first : last + 1]))

    found = []
    dead_ranges = []
    stack: list[Frame] = []
    dead_since: Optional[int] = None
    for d in directives(tokens):
        was_dead = any(f.dead for f in stack)
        if d.name in OPENERS:
            value = condition(d)
            if value is not None and not was_dead:
                found.append(region(Placement.GUARD, d.first, d.last))
            stack.append(Frame(was_dead, value is not None and value != 0, value == 0))
        elif d.name in ALTERNATIVES and stack:
            frame = stack[-1]
            if frame.taken:
                frame.dead = True
            else:
                value = condition(d)
                if value is not None and not frame.outer_dead:
                    found.append(region(Placement.GUARD, d.first, d.last))
                frame.dead = value == 0
                frame.taken = value is not None and value != 0
        elif d.name == "else" and stack:
            stack[-1].dead = stack[-1].taken
        elif d.name == "endif" and stack:
            stack.pop()
        elif is_message(d) and not was_dead:
            found.append(region(Placement.MESSAGE, d.first, d.last))
        now_dead = any(f.dead for f in stack)
        if now_dead and not was_dead:
            dead_since = d.last + 1
        elif was_dead and not now_dead:
            dead_ranges.append((dead_since, d.first - 1))
    if dead_since is not None and any(f.dead for f in stack):
        dead_ranges.append((dead_since, len(tokens) - 1))
    return found, dead_ranges


def string_body(literal: str) -> Optional[str]:
    m = STRING_PREFIX.match(literal)
    if m is None:
        return None
    if m.group(1) is not None:
        closer = ")" + m.group(1) + '"'
        return literal[m.end() : -len(closer)] if literal.endswith(closer) else None
    return literal[m.end() : -1] if len(literal) > m.end() and literal.endswith('"') else None


def pragma_messages(tokens: list[Token], dead_ranges: list[tuple[int, int]]) -> list[Region]:
    significant = [i for i, t in enumerate(tokens) if t.kind in CODE_KINDS]
    found = []
    for k, i in enumerate(significant):
        if tokens[i].text != "_Pragma" or any(a <= i <= b for a, b in dead_ranges):
            continue
        rest = [tokens[j].text for j in significant[k + 1 : k + 4]]
        if len(rest) < 3 or rest[0] != "(" or rest[2] != ")":
            continue
        body = string_body(rest[1])
        if body is not None and PRAGMA_MESSAGE.match(body):
            last = significant[k + 3]
            found.append(Region(Placement.MESSAGE, i, last, tuple(tokens[i : last + 1])))
    return found


def directive_regions(tokens: list[Token]) -> list[Region]:
    found, dead_ranges = scan_directives(tokens)
    found.extend(pragma_messages(tokens, dead_ranges))
    return sorted(found, key=lambda r: r.first)
