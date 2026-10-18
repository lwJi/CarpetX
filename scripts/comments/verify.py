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


_SPLICE = re.compile(r"\\[ \t]*\r?\n")


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
    first_on_line = True
    for m in record.finditer(output):
        if m.start() != position:
            raise RuntimeError(f"cannot parse clang raw-token output near: {output[position:position + 200]!r}")
        position = m.end()
        kind, spelling = m.group(1), m.group(2)
        if kind == "comment":
            continue
        if kind == "unknown" and not spelling.strip():
            if "\n" in _SPLICE.sub("", spelling):
                first_on_line = True
            continue
        tokens.append(Token(kind, spelling, first_on_line, int(m.group(4))))
        first_on_line = False
    if position != len(output):
        raise RuntimeError(f"cannot parse clang raw-token output near: {output[position:position + 200]!r}")
    return tokens


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


def _first_token_difference(a: list[Token], b: list[Token]) -> str:
    for x, y in zip(a, b):
        if (x.kind, x.spelling, x.first_on_line) != (y.kind, y.spelling, y.first_on_line):
            return f"before line {x.line} {x.kind} {x.spelling!r} vs after line {y.line} {y.kind} {y.spelling!r}"
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


def verify_cxx(path: str, before: str, after: str) -> Result:
    resolved, mode = unifdef_k(before)
    evidence = f"clang raw tokens equal ({mode} before)"
    a, b = clang_tokens(resolved), clang_tokens(after)
    key = lambda t: (t.kind, t.spelling, t.first_on_line)
    if list(map(key, a)) == list(map(key, b)):
        return Result(path, True, evidence)
    return Result(path, False, evidence, _first_token_difference(a, b))


def verify_piraha(path: str, peg: str, before: str, after: str, flesh: str) -> Result:
    evidence = f"Piraha {peg} parse trees equal"
    (ok_a, dump_a), (ok_b, dump_b) = piraha_dumps(flesh, peg, [before, after])
    if not ok_a:
        return Result(path, False, evidence, f"before does not parse: {dump_a.strip()[:300]}")
    if not ok_b:
        return Result(path, False, evidence, f"after does not parse: {dump_b.strip()[:300]}")
    if dump_a == dump_b:
        return Result(path, True, evidence)
    return Result(path, False, evidence, _first_line_difference(dump_a, dump_b))


def verify_file(path: str, before: str, after: str, flesh: str | None) -> Result:
    lang = covered(path)
    if lang is None:
        return Result(path, False, "not a covered file", "only comments in covered files may change")
    if lang is Lang.CXX:
        return verify_cxx(path, before, after)
    peg = peg_for(path)
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
