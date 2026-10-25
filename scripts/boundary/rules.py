from __future__ import annotations

import os
import re
import sys
from collections import Counter
from dataclasses import dataclass
from typing import Optional

TOOL = os.path.dirname(os.path.abspath(__file__))
COMMENTS = os.path.join(os.path.dirname(TOOL), "comments")
if COMMENTS not in sys.path:
    sys.path.append(COMMENTS)

from files import Lang, changes_since, covered, is_regular, read_text, show  # noqa: E402

MARKER = "CARPETX_INTERNAL"
PRIVATE_DIR = "CarpetX/src/"
PUBLIC_DIR = "CarpetX/src/include/"
MAKE_DEFN = "CarpetX/src/make.code.defn"
CHECKED = (Lang.CXX, Lang.MAKE)
MARKER_WORD = re.compile(r"CARPETX_INTERNAL\b")
GUARD = re.compile(r"^\s*#\s*ifndef\s+CARPETX_INTERNAL\s*$")
INCLUDE = re.compile(r'^\s*#\s*include\s*(?:"([^"]*)"|<([^>]*)>)')
PRIVATE_PATH = "CarpetX/src"
VALIDITY_FILES = ("CarpetX/src/valid.cxx", "CarpetX/src/valid.hxx")
VALIDITY_WORD = re.compile(
    r"(\b(error_if_invalid|warn_if_invalid|poison_invalid_g[fa]|check_valid_g[fa])"
    r"|(\.|->)(set_all|set_int|set_outer|set_ghosts|set_invalid|set_valid))\s*\("
)
EXCERPT_WIDTH = 72

RULE = (
    "A thorn reaches CarpetX only through its public headers (see agent_docs/carpetx-public-api.md).\n"
    "include: an #include whose path reaches into CarpetX/src; include <carpetx.hxx> instead.\n"
    "marker: CARPETX_INTERNAL belongs only in CarpetX/src/make.code.defn and in the\n"
    "#ifndef CARPETX_INTERNAL guard of a private CarpetX header.\n"
    "validity: a validity kernel or a why_valid_t setter is named only in CarpetX/src/valid.cxx\n"
    "and valid.hxx; every other site opens a write scope or calls require or invalidate."
)


@dataclass(frozen=True)
class Finding:
    line: int
    rule: str
    text: str


def is_private(name: str) -> bool:
    return name.startswith(PRIVATE_DIR) and not name.startswith(PUBLIC_DIR)


def marker_allowed(name: str, lang: Lang, line: str) -> bool:
    if name == MAKE_DEFN:
        return True
    return is_private(name) and lang is Lang.CXX and bool(GUARD.match(line))


def findings(name: str, text: str, lang: Optional[Lang]) -> list[Finding]:
    if lang not in CHECKED:
        return []
    found = []
    for number, line in enumerate(text.splitlines(), 1):
        if MARKER_WORD.search(line) and not marker_allowed(name, lang, line):
            found.append(Finding(number, "marker", line.strip()))
        if lang is Lang.CXX and not is_private(name):
            match = INCLUDE.match(line)
            if match and PRIVATE_PATH in (match.group(1) or match.group(2) or ""):
                found.append(Finding(number, "include", line.strip()))
        if lang is Lang.CXX and is_private(name) and name not in VALIDITY_FILES and VALIDITY_WORD.search(line):
            found.append(Finding(number, "validity", line.strip()))
    return found


def new_findings(before: list[Finding], after: list[Finding]) -> list[Finding]:
    old = Counter((f.rule, f.text) for f in before)
    found = []
    for f in after:
        if old[(f.rule, f.text)] > 0:
            old[(f.rule, f.text)] -= 1
        else:
            found.append(f)
    return found


def since_findings(root: str, base: str) -> list[tuple[str, Finding]]:
    found = []
    for name, old in sorted(changes_since(root, base).items()):
        lang = covered(name, root)
        if lang not in CHECKED:
            continue
        path = os.path.join(root, name)
        after = read_text(path) if is_regular(path) else ""
        before = findings(old, show(root, base, old), lang) if old is not None else []
        found.extend((name, f) for f in new_findings(before, findings(name, after, lang)))
    return found


def plural(count: int, noun: str) -> str:
    return f"{count} {noun}" + ("" if count == 1 else "s")


def summary(found: list[tuple[str, Finding]]) -> str:
    includes = sum(1 for _, f in found if f.rule == "include")
    markers = sum(1 for _, f in found if f.rule == "marker")
    validity = sum(1 for _, f in found if f.rule == "validity")
    if not found:
        return (
            "✓ boundary: no private CarpetX includes, no stray CARPETX_INTERNAL, "
            "no validity call outside valid.cxx"
        )
    parts = (
        ([plural(includes, "private include")] if includes else [])
        + ([plural(markers, "stray marker")] if markers else [])
        + ([plural(validity, "validity call") + " outside valid.cxx"] if validity else [])
    )
    return "✗ boundary: " + ", ".join(parts)


def excerpt(text: str) -> str:
    return text[:EXCERPT_WIDTH] + ("…" if len(text) > EXCERPT_WIDTH else "")


def render(found: list[tuple[str, Finding]]) -> str:
    if not found:
        return summary(found)
    found = sorted(found, key=lambda e: (e[0], e[1].line))
    places = [f"{name}:{f.line}" for name, f in found]
    width = max(len(p) for p in places)
    out = [f"{place.ljust(width)}   {f.rule.ljust(8)}   {excerpt(f.text)}" for place, (_, f) in zip(places, found)]
    out.extend(RULE.split("\n"))
    out.append(summary(found))
    return "\n".join(out)
