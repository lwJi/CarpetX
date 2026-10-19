import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import PurePosixPath

from files import Lang, covered, git_toplevel

HERE = os.path.dirname(os.path.abspath(__file__))
PIRAHA_DUMP = os.path.join(HERE, "piraha_dump.pl")
PEGS = {
    "interface.ccl": "interface.peg",
    "param.ccl": "param.peg",
    "schedule.ccl": "schedule.peg",
    "configuration.ccl": "config.peg",
}


class ToolMissing(Exception):
    pass


@dataclass(frozen=True)
class Result:
    path: str
    ok: bool
    evidence: str
    detail: str = ""


def peg_for(path: str) -> str | None:
    lang = covered(path)
    if lang is Lang.PAR:
        return "par.peg"
    if lang is Lang.CCL:
        return PEGS.get(PurePosixPath(path).name)
    return None


def _require(tool: str) -> str:
    found = shutil.which(tool)
    if found is None:
        raise ToolMissing(f"{tool} is not on PATH")
    return found


def unifdef_k(text: str) -> tuple[str, str]:
    unifdef = _require("unifdef")
    data = text.encode("utf-8", "surrogateescape")
    for flags in (["-k"], ["-t", "-k"]):
        result = subprocess.run([unifdef, *flags], input=data, capture_output=True)
        if result.returncode in (0, 1):
            return result.stdout.decode("utf-8", "surrogateescape"), " ".join(["unifdef", *flags])
    raise RuntimeError(f"unifdef -t -k failed: {result.stderr.decode(errors='replace').strip()}")


@dataclass(frozen=True)
class Token:
    kind: str
    spelling: str
    first_on_line: bool
    line: int
    space_before: bool = False
    in_directive: bool = False


_SPLICE = re.compile(r"\\[ \t]*\r?\n")
_TIGHT = frozenset({"l_paren", "r_paren", "l_square", "r_square", "l_brace", "r_brace", "comma", "semi"})


def clang_tokens(text: str) -> list[Token]:
    clang = _require("clang++")
    with tempfile.TemporaryDirectory() as tmp:
        source = os.path.join(tmp, "source.cxx")
        with open(source, "w", encoding="utf-8", errors="surrogateescape", newline="") as f:
            f.write(text)
        result = subprocess.run(
            [clang, "-x", "c++", "-std=c++20", "-w", "-fsyntax-only",
             "-Xclang", "-dump-raw-tokens", source],
            capture_output=True,
        )
        output = result.stderr.decode("utf-8", "surrogateescape")
        if result.returncode != 0:
            raise RuntimeError(f"clang++ -dump-raw-tokens failed: {output[:500]}")
        record = re.compile(
            r"(\w+) '(.*?)'\t(.*?)\tLoc=<" + re.escape(source) + r":(\d+):(\d+)>\n", re.S
        )
    tokens = []
    position = 0
    first_on_line, space, directive = True, False, False
    for m in record.finditer(output):
        if m.start() != position:
            raise RuntimeError(f"cannot parse clang raw-token output near: {output[position:position + 200]!r}")
        position = m.end()
        kind, spelling = m.group(1), m.group(2)
        if kind == "comment":
            space = True
            continue
        if kind == "unknown" and not spelling.strip():
            blank = _SPLICE.sub("", spelling)
            first_on_line = first_on_line or "\n" in blank
            space = space or bool(blank)
            continue
        if first_on_line:
            directive = kind == "hash"
        tokens.append(Token(kind, spelling, first_on_line, int(m.group(4)), space, directive))
        first_on_line = space = False
    if position != len(output):
        raise RuntimeError(f"cannot parse clang raw-token output near: {output[position:position + 200]!r}")
    return tokens


def _spacing(tokens: list[Token]) -> list[bool | None]:
    found: list[bool | None] = []
    for i, t in enumerate(tokens):
        tight = t.kind in _TIGHT or (i > 0 and tokens[i - 1].kind in _TIGHT)
        compared = not t.first_on_line and (t.in_directive or not tight)
        found.append(t.space_before if compared else None)
    return found


def token_shape(tokens: list[Token]) -> list[tuple[str, str, bool, bool | None]]:
    return [(t.kind, t.spelling, t.first_on_line, s) for t, s in zip(tokens, _spacing(tokens))]


def piraha_dumps(flesh: str, peg: str, texts: list[str]) -> list[tuple[bool, str]]:
    perl = _require("perl")
    if not os.path.isfile(os.path.join(flesh, "lib", "sbin", "Piraha.pm")):
        raise ToolMissing(f"no Cactus flesh at {flesh} (expected lib/sbin/Piraha.pm)")
    with tempfile.TemporaryDirectory() as tmp:
        paths = []
        for number, text in enumerate(texts):
            path = os.path.join(tmp, f"{number}.txt")
            with open(path, "w", encoding="utf-8", errors="surrogateescape", newline="") as f:
                f.write(text)
            paths.append(path)
        result = subprocess.run([perl, PIRAHA_DUMP, flesh, peg, *paths], capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f"piraha_dump.pl failed: {result.stderr.decode(errors='replace').strip()}")
    chunks = result.stdout.decode("utf-8", "surrogateescape").split("\0")[: len(texts)]
    dumps = []
    for chunk in chunks:
        status, _, body = chunk.partition("\n")
        dumps.append((status == "OK", body))
    return dumps


def _describe(t: Token, spaced: bool | None) -> str:
    gap = "" if spaced is None else (" after a space" if spaced else " with no space before it")
    return f"line {t.line} {t.kind} {t.spelling!r}{gap}"


def _first_token_difference(a: list[Token], b: list[Token]) -> str:
    for x, y, sx, sy in zip(a, b, token_shape(a), token_shape(b)):
        if sx != sy:
            return f"before {_describe(x, sx[3])} vs after {_describe(y, sy[3])}"
    if len(a) > len(b):
        x = a[len(b)]
        return f"before has more tokens, from line {x.line} {x.kind} {x.spelling!r}"
    y = b[len(a)]
    return f"after has more tokens, from line {y.line} {y.kind} {y.spelling!r}"


def _first_line_difference(a: str, b: str) -> str:
    al, bl = a.split("\n"), b.split("\n")
    for number, (x, y) in enumerate(zip(al, bl), start=1):
        if x != y:
            return f"parse tree line {number}: before {x.strip()!r} vs after {y.strip()!r}"
    return f"parse trees differ in length ({len(al)} vs {len(bl)} lines)"


_TODO_WARNING = re.compile(r'^"?\s*(?:TODO|FIXME)\b')


def drop_todo_warnings(tokens: list[Token]) -> tuple[list[Token], int]:
    kept: list[Token] = []
    dropped = 0
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if (
            t.kind == "hash"
            and t.first_on_line
            and i + 2 < len(tokens)
            and (tokens[i + 1].kind, tokens[i + 1].spelling) == ("raw_identifier", "warning")
            and not tokens[i + 1].first_on_line
            and not tokens[i + 2].first_on_line
            and _TODO_WARNING.match(tokens[i + 2].spelling)
        ):
            i += 2
            while i < len(tokens) and not tokens[i].first_on_line:
                i += 1
            dropped += 1
            continue
        kept.append(t)
        i += 1
    return kept, dropped


def verify_cxx(path: str, before: str, after: str) -> Result:
    resolved, mode = unifdef_k(before)
    a, dropped = drop_todo_warnings(clang_tokens(resolved))
    b, _ = drop_todo_warnings(clang_tokens(after))
    evidence = f"clang raw tokens equal ({mode} before"
    if dropped:
        evidence += f", #warning TODO directives dropped: {dropped}"
    evidence += ")"
    if token_shape(a) == token_shape(b):
        return Result(path, True, evidence)
    return Result(path, False, evidence, _first_token_difference(a, b))


def only_hash_lines(text: str) -> bool:
    return all(not line.strip() or line.lstrip().startswith("#") for line in text.split("\n"))


@dataclass(frozen=True)
class Dialect:
    marker: str
    comment: re.Pattern
    escapes: bool
    no_comment: re.Pattern | None = None


HASH_DIALECT = Dialect("#", re.compile("#"), escapes=True)
FORTRAN_DIALECT = Dialect("!", re.compile(r"!(?!\$)"), escapes=False, no_comment=re.compile(r"\s*#"))


def _outside_quotes(prefix: str, escapes: bool) -> bool:
    quote = None
    i = 0
    while i < len(prefix):
        c = prefix[i]
        if escapes and c == "\\":
            i += 1
        elif quote is None and c in "\"'":
            quote = c
        elif c == quote:
            quote = None
        i += 1
    return quote is None


def _may_hold_comment(dialect: Dialect, line: str) -> bool:
    return dialect.no_comment is None or not dialect.no_comment.match(line)


def _lost_trailing_comment(dialect: Dialect, before: str, after: str) -> bool:
    if not before.startswith(after) or not after.strip() or not _may_hold_comment(dialect, before):
        return False
    rest = before[len(after) :].lstrip(" \t")
    return bool(dialect.comment.match(rest)) and _outside_quotes(after, dialect.escapes)


def _whole_line_comment(dialect: Dialect, line: str) -> bool:
    return bool(dialect.comment.match(line.lstrip())) and _may_hold_comment(dialect, line)


def line_audit(path: str, dialect: Dialect, before: str, after: str) -> Result:
    evidence = f"only whole-line or trailing '{dialect.marker}…' removed (line audit)"
    a, b = before.split("\n"), after.split("\n")
    j = 0
    for number, line in enumerate(a, start=1):
        if j < len(b) and (b[j] == line or _lost_trailing_comment(dialect, line, b[j])):
            j += 1
        elif not line.strip() or _whole_line_comment(dialect, line):
            continue
        else:
            return Result(path, False, evidence, f"before line {number} {line.strip()!r} is not in after")
    if j < len(b):
        return Result(path, False, evidence, f"after line {j + 1} {b[j].strip()!r} is not in before")
    return Result(path, True, evidence)


_PAR_STRING = re.compile(r'inquot\("((?:[^"\\]|\\.)*)"\)')
_DUMP_ESCAPES = {"n": "\n", "t": "\t", "r": "\r"}
_STRING_COMMENT = re.compile("\x01[^\x02]*\x02")


def _unescape(dumped: str) -> str:
    return re.sub(r"\\(.)", lambda m: _DUMP_ESCAPES.get(m.group(1), m.group(1)), dumped, flags=re.S)


def string_lines(marked: str) -> list[tuple[str, bool]]:
    return [(_STRING_COMMENT.sub("", line), "\x01" in line) for line in marked.split("\n")]


def only_string_comments_removed(before: str, after: str) -> bool:
    b = string_lines(before)
    a = [value for value, _ in string_lines(after)]
    reachable = {0}
    for i, (value, comment) in enumerate(b):
        following = set()
        for j in reachable:
            if j < len(a) and (a[j] == value or (comment and a[j] == value.rstrip(" \t"))):
                following.add(j + 1)
            if comment and not value.strip() and i + 1 < len(b):
                following.add(j)
        reachable = following
    return len(a) in reachable


def _compare_par_dumps(a: str, b: str) -> tuple[str | None, int]:
    skeleton_a, skeleton_b = _PAR_STRING.sub("inquot(…)", a), _PAR_STRING.sub("inquot(…)", b)
    if skeleton_a != skeleton_b:
        return _first_line_difference(skeleton_a, skeleton_b), 0
    strings_a = [_unescape(x) for x in _PAR_STRING.findall(a)]
    strings_b = [_unescape(x) for x in _PAR_STRING.findall(b)]
    commented = 0
    for x, y in zip(strings_a, strings_b):
        if x == y:
            continue
        if not only_string_comments_removed(x, y):
            value = _STRING_COMMENT.sub("#…", x)
            return f"string {value[:80]!r} changed beyond its '#' lines", 0
        commented += sum(comment for _, comment in string_lines(x))
    return None, commented


def verify_piraha(path: str, peg: str, before: str, after: str, flesh: str) -> Result:
    evidence = f"Piraha {peg} parse trees equal"
    (ok_a, dump_a), (ok_b, dump_b) = piraha_dumps(flesh, peg, [before, after])
    if not ok_a:
        return Result(path, False, evidence, f"before does not parse: {dump_a.strip()[:300]}")
    if not ok_b:
        return Result(path, False, evidence, f"after does not parse: {dump_b.strip()[:300]}")
    if dump_a == dump_b:
        return Result(path, True, evidence)
    if not after.strip() and only_hash_lines(before):
        return Result(path, True, f"only '#' lines before, empty after; Piraha {peg} parses both")
    if peg != "par.peg":
        return Result(path, False, evidence, _first_line_difference(dump_a, dump_b))
    difference, commented = _compare_par_dumps(dump_a, dump_b)
    if difference is not None:
        return Result(path, False, evidence, difference)
    noun = "line" if commented == 1 else "lines"
    return Result(path, True, f"{evidence} but for {commented} string {noun} that par.peg's stringcomment drops")


def verify_file(path: str, before: str, after: str, flesh: str | None) -> Result:
    lang = covered(path)
    if lang is None:
        return Result(path, False, "not a covered file", "only comments in covered files may change")
    if lang is Lang.CXX:
        return verify_cxx(path, before, after)
    if lang is Lang.FORTRAN:
        return line_audit(path, FORTRAN_DIALECT, before, after)
    peg = peg_for(path)
    if peg is None and lang is Lang.CCL:
        return line_audit(path, HASH_DIALECT, before, after)
    if peg is None:
        return Result(path, False, "no independent check", f"verify has no checker for {PurePosixPath(path).name} yet")
    if flesh is None:
        raise ToolMissing("set $CACTUSX or pass --flesh to the Cactus flesh checkout")
    return verify_piraha(path, peg, before, after, flesh)


def _git(top: str, *args: str) -> bytes:
    return subprocess.run(["git", "-C", top, *args], capture_output=True, check=True).stdout


def _show(top: str, rev: str, path: str) -> str:
    return _git(top, "show", f"{rev}:{path}").decode("utf-8", "surrogateescape")


def changed_files(top: str, before: str, after: str) -> list[tuple[str, str]]:
    out = _git(top, "diff", "--no-renames", "--name-status", "-z", before, after)
    fields = out.decode("utf-8", "surrogateescape").split("\0")
    return [(fields[i], fields[i + 1]) for i in range(0, len(fields) - 1, 2)]


def verify_revs(before: str, after: str, flesh: str | None, out=sys.stdout) -> int:
    top = git_toplevel(".")
    if top is None:
        print("verify: not inside a git repository", file=sys.stderr)
        return 2
    results = []
    for status, path in changed_files(top, before, after):
        if status != "M":
            results.append(Result(path, False, "file added or deleted", f"git status {status}: a strip only modifies files"))
            continue
        results.append(verify_file(path, _show(top, before, path), _show(top, after, path), flesh))
    width = max((len(r.path) for r in results), default=0)
    for r in results:
        mark = "ok  " if r.ok else "FAIL"
        print(f"{mark} {r.path:<{width}}  {r.evidence}", file=out)
        if not r.ok:
            print(f"     {r.detail}", file=out)
    failed = sum(not r.ok for r in results)
    if failed:
        print(f"verify {before} {after}: {failed} of {len(results)} files changed more than comments", file=out)
        return 1
    print(f"verify {before} {after}: {len(results)} files changed, only comments", file=out)
    return 0
