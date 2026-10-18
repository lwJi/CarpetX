import re
from dataclasses import dataclass

import lex_cxx
import lex_hash
from files import Lang
from lex_cxx import Span

NAMESPACE_CLOSER = re.compile(r"^// namespace( [\w:]+)?$")
LINE_MARKER = re.compile(
    r"^//\s*(?:clang-format (?:off|on)|NOLINT(?:NEXTLINE|BEGIN|END)?(?:\([^()\n]*\))?)$"
)
BLOCK_MARKER = re.compile(
    r"^/\*\s*(?:clang-format (?:off|on)|NOLINT(?:NEXTLINE|BEGIN|END)?(?:\([^()\n]*\))?)\s*\*/$"
)
REF = r"(?:<?https?://\S+?>?|arXiv:\d{4}\.\d{4,5}(?:v\d+)?|doi:\S+)"
CITATION_LINE = re.compile(
    rf"^(?:See|From|Taken from|cf\.)?\s*{REF}(?:\s*\[[\w.-]+\])?"
    r"(?:,?\s*(?:eqs?\.?\s*)?\(?[\d,\s–-]+\)?)?\.?$"
)
_OPENER = {"line": re.compile(r"^(?://+|!+)"), "hash": re.compile(r"^#+"), "block": re.compile(r"^/\*+")}
_BLOCK_CLOSER = re.compile(r"\*+/$")
_BLOCK_LINE_LEADER = re.compile(r"^\s*\*+")


@dataclass(frozen=True)
class Group:
    spans: tuple[Span, ...]

    @property
    def line(self) -> int:
        return self.spans[0].line

    @property
    def text(self) -> str:
        return "\n".join(s.text.rstrip("\n") for s in self.spans)


@dataclass(frozen=True)
class Allowed:
    group: Group
    reason: str


@dataclass(frozen=True)
class Violation:
    group: Group

    @property
    def line(self) -> int:
        return self.group.line


def lex(text: str, lang: Lang) -> list[Span]:
    if lang is Lang.CXX:
        return lex_cxx.lex(text)
    return lex_hash.lex(text, lang)


def _line_bounds(text: str, span: Span) -> tuple[int, int]:
    start = text.rfind("\n", 0, span.start) + 1
    end = text.find("\n", span.end)
    return start, len(text) if end < 0 else end


def code_before(text: str, span: Span) -> str:
    start, _ = _line_bounds(text, span)
    return text[start : span.start]


def code_after(text: str, span: Span) -> str:
    _, end = _line_bounds(text, span)
    return text[span.end : end]


def is_full_line(text: str, span: Span) -> bool:
    return not code_before(text, span).strip() and not code_after(text, span).strip()


def is_trailing(text: str, span: Span) -> bool:
    return bool(code_before(text, span).strip()) and not code_after(text, span).strip()


def is_marker(span: Span) -> bool:
    body = span.text.strip()
    pattern = LINE_MARKER if span.kind == "line" else BLOCK_MARKER
    return span.kind in ("line", "block") and bool(pattern.match(body))


def comment_lines(span: Span) -> list[str]:
    body = _OPENER[span.kind].sub("", span.text.rstrip("\n"), count=1)
    if span.kind != "block":
        return [line.strip() for line in body.split("\n")]
    lines = _BLOCK_CLOSER.sub("", body).split("\n")
    return [lines[0].strip()] + [_BLOCK_LINE_LEADER.sub("", line).strip() for line in lines[1:]]


def is_citation(g: Group) -> bool:
    lines = [line for s in g.spans for line in comment_lines(s) if line]
    return bool(lines) and all(CITATION_LINE.match(line) for line in lines)


def _last_line(text: str, span: Span) -> int:
    return span.line + text.count("\n", span.start, span.end)


def group(spans: list[Span], text: str) -> list[Group]:
    disabled = [s for s in spans if s.kind == "disabled"]
    groups = [Group(g) for g in lex_cxx.guard_groups(disabled)]
    run: list[Span] = []
    for span in spans:
        if span.kind == "disabled":
            continue
        if span.kind == "warning" or is_marker(span) or not is_full_line(text, span):
            groups.append(Group((span,)))
            continue
        if run and span.line == _last_line(text, run[-1]) + 1:
            run.append(span)
            continue
        if run:
            groups.append(Group(tuple(run)))
        run = [span]
    if run:
        groups.append(Group(tuple(run)))
    return sorted(groups, key=lambda g: g.spans[0].start)


def classify(g: Group, text: str) -> Allowed | Violation:
    if any(s.kind in ("disabled", "warning") for s in g.spans):
        return Violation(g)
    if all(is_marker(s) for s in g.spans):
        return Allowed(g, "formatter or lint marker")
    if len(g.spans) == 1 and g.spans[0].kind == "line" and is_trailing(text, g.spans[0]):
        span = g.spans[0]
        body = span.text.rstrip()
        if body == "//":
            return Allowed(g, "line-break guard")
        if NAMESPACE_CLOSER.match(body) and code_before(text, span).rstrip().endswith("}"):
            return Allowed(g, "namespace closer")
    if is_citation(g):
        return Allowed(g, "citation")
    return Violation(g)


def check(text: str, lang: Lang) -> list[Violation]:
    found = []
    for g in group(lex(text, lang), text):
        verdict = classify(g, text)
        if isinstance(verdict, Violation):
            found.append(verdict)
    return found
