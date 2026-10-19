from __future__ import annotations

from typing import Callable

from tokens import Kind, Placement, Region, Token

CODE_KINDS = (Kind.CODE, Kind.DIRECTIVE)


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
