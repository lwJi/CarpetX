from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Callable, Optional

import lex_clang
from files import Lang, blobs, decode, git, lang_of, parallel, split_z, thorns_in, tree_names
from lex_clang import ToolMissing
from strip import unifdef_k
from tokens import Kind, Token

PIRAHA_DUMP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "piraha_dump.pl")
PEGS = {
    Lang.INTERFACE: "interface.peg",
    Lang.PARAM: "param.peg",
    Lang.SCHEDULE: "schedule.peg",
    Lang.CONFIGURATION: "config.peg",
    Lang.PAR: "par.peg",
}
TIGHT = {"(", ")", "[", "]", "{", "}", ",", ";"}
INQUOT = re.compile(r'inquot\("((?:[^"\\]|\\.)*)"\)')
DUMP_ESCAPES = {"n": "\n", "t": "\t", "r": "\r"}
MARKED = re.compile("\x01[^\x02]*\x02")
HORIZONTAL = " \t\f\v"


@dataclass(frozen=True)
class Result:
    name: str
    ok: bool
    how: str
    why: str = ""


def flesh_dir(path: Optional[str]) -> str:
    if not path:
        raise ToolMissing("verifying .ccl and .par files needs the Cactus flesh: set $CACTUSX or pass --flesh")
    if not (os.path.isfile(os.path.join(path, "lib", "sbin", "Piraha.pm"))
            and os.path.isdir(os.path.join(path, "src", "piraha", "pegs"))):
        raise ToolMissing(f"no Cactus flesh at {path} (expected lib/sbin/Piraha.pm and src/piraha/pegs)")
    return path


@dataclass(frozen=True)
class Shape:
    text: str
    directive: bool
    line_start: bool
    spaced: Optional[bool]
    line: int

    @property
    def key(self) -> tuple:
        return self.text, self.directive, self.line_start, self.spaced

    def describe(self) -> str:
        kind = "directive token" if self.directive else "token"
        where = (
            " at a line start" if self.line_start
            else " after a gap" if self.spaced
            else " with no gap before it" if self.spaced is False
            else ""
        )
        return f"line {self.line} {kind} {self.text!r}{where}"


def shapes(tokens: list[Token]) -> list[Shape]:
    found = []
    line_start, spaced = True, False
    previous: Optional[Token] = None
    for t in tokens:
        if t.kind is Kind.NEWLINE:
            line_start = True
        elif t.kind in (Kind.SPACE, Kind.COMMENT):
            spaced = True
        else:
            directive = t.kind is Kind.DIRECTIVE
            compared = not line_start and (directive or (previous.text not in TIGHT and t.text not in TIGHT))
            found.append(Shape(t.text, directive, line_start, spaced if compared else None, t.line))
            previous, line_start, spaced = t, False, False
    return found


def verify_cxx(name: str, before: str, after: str) -> Result:
    how = "clang tokens equal (unifdef -k before)"
    a = shapes(lex_clang.lex(unifdef_k(before)))
    b = shapes(lex_clang.lex(after))
    for x, y in zip(a, b):
        if x.key != y.key:
            return Result(name, False, how, f"before {x.describe()}, after {y.describe()}")
    if len(a) > len(b):
        return Result(name, False, how, f"before has more tokens, from {a[len(b)].describe()}")
    if len(b) > len(a):
        return Result(name, False, how, f"after has more tokens, from {b[len(a)].describe()}")
    return Result(name, True, how)


@dataclass(frozen=True)
class Placed:
    text: str
    kind: Kind
    opens_line: bool
    spaced: Optional[bool]
    line: int

    @property
    def key(self) -> tuple:
        return self.text, self.kind, self.opens_line, self.spaced

    def describe(self) -> str:
        kind = {Kind.DIRECTIVE: "directive token", Kind.COMMENT: "comment"}.get(self.kind, "token")
        where = (
            " on a line of its own" if self.kind is Kind.COMMENT and self.opens_line
            else " after code on its line" if self.kind is Kind.COMMENT
            else " opening a directive" if self.opens_line
            else " after a gap" if self.spaced
            else " with no gap before it" if self.spaced is False
            else ""
        )
        return f"line {self.line} {kind} {self.text!r}{where}"


def placed(tokens: list[Token]) -> list[Placed]:
    found = []
    line_start, spaced = True, False
    previous: Optional[Token] = None
    for t in tokens:
        if t.kind is Kind.NEWLINE:
            line_start = True
        elif t.kind is Kind.SPACE:
            spaced = True
        elif t.kind is Kind.COMMENT:
            found.append(Placed(t.text, t.kind, line_start, None, t.line))
            line_start, spaced = False, True
        else:
            directive = t.kind is Kind.DIRECTIVE
            opens = directive and line_start
            gap = spaced or line_start
            compared = previous is not None and not opens and (
                directive or (previous.text not in TIGHT and t.text not in TIGHT))
            found.append(Placed(t.text, t.kind, opens, gap if compared else None, t.line))
            previous, line_start, spaced = t, False, False
    return found


def verify_layout_cxx(name: str, before: str, after: str) -> Result:
    how = "clang tokens and comments equal, only line breaks and spacing moved"
    a = placed(lex_clang.lex(before))
    b = placed(lex_clang.lex(after))
    for x, y in zip(a, b):
        if x.key != y.key:
            return Result(name, False, how, f"before {x.describe()}, after {y.describe()}")
    if len(a) > len(b):
        return Result(name, False, how, f"before has more tokens, from {a[len(b)].describe()}")
    if len(b) > len(a):
        return Result(name, False, how, f"after has more tokens, from {b[len(a)].describe()}")
    return Result(name, True, how)


def piraha_dumps(flesh: str, peg: str, texts: list[str]) -> list[tuple[bool, str]]:
    perl = shutil.which("perl")
    if perl is None:
        raise ToolMissing("perl not found on PATH")
    with tempfile.TemporaryDirectory(prefix="comments-") as tmp:
        paths = []
        for k, text in enumerate(texts):
            path = os.path.join(tmp, str(k))
            with open(path, "wb") as f:
                f.write(text.encode("utf-8", "surrogateescape"))
            paths.append(path)
        proc = subprocess.run([perl, PIRAHA_DUMP, flesh, peg, *paths], stdin=subprocess.DEVNULL, capture_output=True)
    chunks = decode(proc.stdout).split("\0")
    if proc.returncode != 0 or len(chunks) != len(texts) + 1:
        why = decode(proc.stderr).strip().split("\n")[0]
        raise RuntimeError(f"piraha_dump.pl {peg} exited with status {proc.returncode}: {why}")
    return [(chunk.startswith("OK\n"), chunk.partition("\n")[2]) for chunk in chunks[:-1]]


def unescape(dumped: str) -> str:
    return re.sub(r"\\(.)", lambda m: DUMP_ESCAPES.get(m.group(1), m.group(1)), dumped, flags=re.S)


def only_string_comments_removed(before: str, after: str) -> bool:
    a = after.split("\n")
    reachable = {0}
    for raw in before.split("\n"):
        value = MARKED.sub("", raw)
        comment = "\x01" in raw
        following = set()
        for j in reachable:
            if j < len(a) and (a[j] == raw or (comment and a[j] in (value, value.rstrip(HORIZONTAL)))):
                following.add(j + 1)
            if comment and not value.strip():
                following.add(j)
        reachable = following
    return len(a) in reachable


def first_line_difference(a: str, b: str) -> str:
    for number, (x, y) in enumerate(zip(a.split("\n"), b.split("\n")), start=1):
        if x != y:
            return f"parse tree line {number}: before {x.strip()!r}, after {y.strip()!r}"
    return "the parse trees differ in length"


def only_comment_lines(text: str) -> bool:
    return all(not line.strip() or line.lstrip().startswith("#") for line in text.split("\n"))


def compare_par_strings(a: str, b: str) -> tuple[Optional[str], int]:
    skeleton_a, skeleton_b = INQUOT.sub("inquot(…)", a), INQUOT.sub("inquot(…)", b)
    if skeleton_a != skeleton_b:
        return first_line_difference(skeleton_a, skeleton_b), 0
    lines = 0
    for x, y in zip(map(unescape, INQUOT.findall(a)), map(unescape, INQUOT.findall(b))):
        if x == y:
            continue
        if not only_string_comments_removed(x, y):
            shown = MARKED.sub("#…", x).strip().split("\n")[0]
            return f"the string starting {shown[:60]!r} changed beyond its # lines", 0
        lines += sum("\x01" in line for line in x.split("\n"))
    return None, lines


def verify_tree(name: str, lang: Lang, before: str, after: str, dump_a, dump_b) -> Result:
    peg = PEGS[lang]
    how = f"Piraha {peg} parse trees equal"
    (ok_a, a), (ok_b, b) = dump_a, dump_b
    if not ok_a:
        return Result(name, False, how, "before does not parse: " + " ".join(a.split())[:200])
    if not ok_b:
        return Result(name, False, how, "after does not parse: " + " ".join(b.split())[:200])
    if a == b:
        return Result(name, True, how)
    if not after.strip() and only_comment_lines(before):
        return Result(name, True, f"only # lines before and nothing after; Piraha {peg} parses both")
    if lang is not Lang.PAR:
        return Result(name, False, how, first_line_difference(a, b))
    why, lines = compare_par_strings(a, b)
    if why is not None:
        return Result(name, False, how, why)
    noun = "line" if lines == 1 else "lines"
    return Result(name, True, f"{how}, but for {lines} string {noun} that par.peg's stringcomment drops")


@dataclass(frozen=True)
class Line:
    comment: Optional[int] = None
    whole: bool = False


def odd_backslashes_at_end(text: str) -> bool:
    return (len(text) - len(text.rstrip("\\"))) % 2 == 1


def make_comment(body: str) -> Optional[int]:
    depth = 0
    i = 0
    while i < len(body):
        c = body[i]
        if c == "\\":
            i += 2
            continue
        if c == "$" and body[i + 1 : i + 2] in ("(", "{"):
            depth += 1
            i += 2
            continue
        if depth and c in "({":
            depth += 1
        elif depth and c in ")}":
            depth -= 1
        elif c == "#" and not depth:
            return i
        i += 1
    return None


def make_lines(lines: list[str]) -> list[Line]:
    found = []
    continued = False
    for line in lines:
        body = line.rstrip("\r")
        if continued:
            found.append(Line(0, True))
            continued = odd_backslashes_at_end(body)
            continue
        start = make_comment(body)
        found.append(Line(start, start is not None and not body[:start].strip()))
        continued = start is not None and odd_backslashes_at_end(body)
    return found


def quoted_comment(body: str, i: int, escapes: bool, quote: Optional[str] = None,
                   doubled: bool = False, marker: str = "#") -> tuple[Optional[int], Optional[str]]:
    while i < len(body):
        c = body[i]
        if escapes and c == "\\":
            i += 2
            continue
        if quote is not None:
            if c == quote:
                if doubled and body[i + 1 : i + 2] == quote:
                    i += 2
                    continue
                quote = None
        elif c in "\"'":
            quote = c
        elif c == marker:
            return i, None
        i += 1
    return None, quote


def test_ccl_lines(lines: list[str]) -> list[Line]:
    found = []
    for line in lines:
        body = line.rstrip("\r")
        start, _ = quoted_comment(body, 0, escapes=True)
        found.append(Line(start, start is not None and not body[:start].strip()))
    return found


def fortran_lines(lines: list[str]) -> list[Line]:
    found = []
    quote: Optional[str] = None
    for line in lines:
        body = line.rstrip("\r")
        first = len(body) - len(body.lstrip(HORIZONTAL))
        if body.startswith("#", first):
            found.append(Line())
            continue
        if body.startswith("!$", first):
            start, _ = quoted_comment(body, first + 2, escapes=False, doubled=True, marker="!")
            found.append(Line(start))
            continue
        if quote is not None and (first == len(body) or body[first] == "!"):
            found.append(Line(first if first < len(body) else None, first < len(body)))
            continue
        begin = first + 1 if quote is not None and body.startswith("&", first) else 0
        start, still_open = quoted_comment(body, begin, escapes=False, quote=quote, doubled=True, marker="!")
        quote = still_open if still_open is not None and body.rstrip().endswith("&") else None
        found.append(Line(start, start is not None and not body[:start].strip()))
    return found


AUDITS: dict = {Lang.MAKE: ("#", make_lines), Lang.FORTRAN: ("!", fortran_lines), Lang.TEST: ("#", test_ccl_lines)}


def physical_lines(text: str) -> list[str]:
    lines = text.split("\n")
    return lines[:-1] if lines[-1] == "" else lines


def lost_trailing_comment(line: str, kept: str, info: Line) -> bool:
    if info.comment is None or info.whole or not kept.strip():
        return False
    return kept.endswith("\r") == line.endswith("\r") and kept.rstrip("\r") == line[: info.comment].rstrip(HORIZONTAL)


def removable_blanks(lines: list[str], infos: list[Line]) -> list[bool]:
    found = [False] * len(lines)
    i = 0
    while i < len(lines):
        j = i
        while j < len(lines) and (infos[j].whole or not lines[j].strip()):
            j += 1
        if any(infos[k].whole for k in range(i, j)):
            for k in range(i, j):
                found[k] = True
        i = max(j, i + 1)
    return found


def line_audit(name: str, lang: Lang, before: str, after: str) -> Result:
    marker, scan = AUDITS[lang]
    how = f"only whole-line and trailing {marker} comments removed (line audit)"
    a, b = physical_lines(before), physical_lines(after)
    infos = scan(a)
    removable = removable_blanks(a, infos)
    j = 0
    for i, line in enumerate(a):
        if j < len(b) and (b[j] == line or lost_trailing_comment(line, b[j], infos[i])):
            j += 1
        elif not (infos[i].whole or (removable[i] and not line.strip())):
            return Result(name, False, how, f"before line {i + 1} {line.strip()!r} is not in after")
    if j < len(b):
        return Result(name, False, how, f"after line {j + 1} {b[j].strip()!r} is not in before")
    return Result(name, True, how)


def verify_files(items: list[tuple[str, Lang, str, str]], flesh: Optional[str]) -> list[Result]:
    results: dict = {}
    cxx = [item for item in items if item[1] is Lang.CXX]
    for result in parallel(lambda item: verify_cxx(item[0], item[2], item[3]), cxx):
        results[result.name] = result
    trees = [item for item in items if item[1] in PEGS]
    if trees:
        flesh = flesh_dir(flesh)
        for peg in sorted({PEGS[lang] for _, lang, _, _ in trees}):
            batch = [item for item in trees if PEGS[item[1]] == peg]
            dumps = piraha_dumps(flesh, peg, [text for _, _, a, b in batch for text in (a, b)])
            for k, (name, lang, a, b) in enumerate(batch):
                results[name] = verify_tree(name, lang, a, b, dumps[2 * k], dumps[2 * k + 1])
    for name, lang, a, b in items:
        if lang in AUDITS:
            results[name] = line_audit(name, lang, a, b)
    return [results[name] for name, _, _, _ in items]


def changed_files(root: str, before: str, after: str) -> list[tuple[str, str]]:
    fields = split_z(git(root, "diff", "-z", "--no-renames", "--name-status", "--no-ext-diff", before, after))
    return [(fields[k], fields[k + 1]) for k in range(0, len(fields) - 1, 2)]


def verify_revs(root: str, before: str, after: str, flesh: Optional[str], layout: bool = False) -> list[Result]:
    thorns = thorns_in(tree_names(root, before)) | thorns_in(tree_names(root, after))
    what = "a reformat" if layout else "a strip"
    refused = []
    items = []
    for status, name in changed_files(root, before, after):
        lang = lang_of(name) if name.split("/")[0] in thorns else None
        if status != "M":
            refused.append(Result(name, False, f"{what} only modifies files", f"git reports status {status}"))
        elif lang is None:
            refused.append(Result(name, False, "not a thorn file", f"{what} touches thorn files only"))
        elif layout and lang is not Lang.CXX:
            refused.append(Result(name, False, "not a C/C++ file", f"{what} touches C/C++ sources only"))
        else:
            items.append([name, lang])
    names = [name for name, _ in items]
    old, new = blobs(root, before, names), blobs(root, after, names)
    if layout:
        checked = parallel(lambda item: verify_layout_cxx(item[0], old[item[0]], new[item[0]]), items)
    else:
        checked = verify_files([(name, lang, old[name], new[name]) for name, lang in items], flesh)
    return sorted(refused + checked, key=lambda r: r.name)


def report(results: list[Result], what: str, out: Callable[[str], None], proof: str = "comments") -> bool:
    width = max((len(r.name) for r in results), default=0)
    for r in results:
        out(f"{'ok  ' if r.ok else 'FAIL'}  {r.name.ljust(width)}  {r.how}")
        if not r.ok:
            out(f"      {r.why}")
    failed = sum(not r.ok for r in results)
    noun = "file" if len(results) == 1 else "files"
    if failed:
        out(f"✗ {what}: {failed} of {len(results)} {noun} changed more than {proof}")
    else:
        out(f"✓ {what}: only {proof} ({len(results)} {noun} changed)")
    return not failed
