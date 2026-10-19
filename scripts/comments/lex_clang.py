from __future__ import annotations

import bisect
import os
import re
import shutil
import subprocess
import tempfile

from tokens import Kind, Token

CLANG = "clang++"
CLANG_ARGS = ["-x", "c++", "-std=c++20", "-w", "-fsyntax-only", "-Xclang", "-dump-raw-tokens"]
HEAD = re.compile(rb"(\w+) '")
LINE_END = re.compile(rb"\r\n|\r|\n")
WHITESPACE = re.compile(rb"(?:[ \t\f\v\r\n]|\\[ \t\f\v]*(?:\r\n|\r|\n))+")
WHITESPACE_PIECE = re.compile(rb"(?P<newline>\r\n|\r|\n)|(?:[ \t\f\v]|\\[ \t\f\v]*(?:\r\n|\r|\n))+")
INCLUDE_NAMES = {b"include", b"include_next", b"import"}
CONDITIONAL_NAMES = {b"if", b"elif"}
HAS_INCLUDE = {b"__has_include", b"__has_include_next"}


class ToolMissing(RuntimeError):
    pass


class LexError(RuntimeError):
    pass


def clang() -> str:
    path = shutil.which(CLANG)
    if path is None:
        raise ToolMissing(f"{CLANG} not found on PATH")
    return path


def raw_records(data: bytes) -> list[tuple[str, int, int]]:
    exe = clang()
    tmp = tempfile.mkdtemp(prefix="comments-")
    try:
        path = os.path.join(tmp, "src.cxx")
        with open(path, "wb") as f:
            f.write(data)
        proc = subprocess.run(
            [exe, *CLANG_ARGS, path], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    dump = proc.stderr
    if proc.returncode != 0:
        first = dump.decode("utf-8", "replace").strip().split("\n")[0]
        raise LexError(f"{CLANG} exited with status {proc.returncode}: {first}")
    tail = re.compile(rb"\tLoc=<" + re.escape(os.fsencode(path)) + rb":(\d+):(\d+)>\n")
    records = []
    pos = 0
    for m in tail.finditer(dump):
        head = HEAD.match(dump, pos)
        if head is None:
            raise LexError(f"unexpected {CLANG} output at byte {pos} of its token dump")
        records.append((head.group(1).decode("ascii"), int(m.group(1)), int(m.group(2))))
        pos = m.end()
    if dump[pos:].strip():
        raise LexError(f"unexpected {CLANG} output after its last token record")
    return records


def line_starts(data: bytes) -> list[int]:
    return [0] + [m.end() for m in LINE_END.finditer(data)]


def spans_of(data: bytes, records: list[tuple[str, int, int]], starts: list[int]) -> list[tuple[str, int, int]]:
    offsets = []
    for kind, line, col in records:
        if line < 1 or line > len(starts):
            raise LexError(f"{CLANG} reported line {line}, past the end of the file")
        offsets.append((kind, starts[line - 1] + col - 1))
    if data and (not offsets or offsets[0][1] > 0):
        offsets.insert(0, ("gap", 0))
    spans = []
    for i, (kind, start) in enumerate(offsets):
        end = offsets[i + 1][1] if i + 1 < len(offsets) else len(data)
        if not start < end <= len(data):
            raise LexError(f"{CLANG} token positions are not increasing at byte {start}")
        if kind == "comment" and data[start : start + 1] != b"/":
            raise LexError(f"{CLANG} comment position does not point at a comment at byte {start}")
        spans.append((kind, start, end))
    return spans


def classify(data: bytes, spans: list[tuple[str, int, int]]) -> list[list]:
    items: list[list] = []

    def add(kind: Kind, start: int, end: int) -> None:
        if kind is Kind.SPACE and items and items[-1][0] is Kind.SPACE:
            items[-1][2] = end
        else:
            items.append([kind, start, end])

    for kind, start, end in spans:
        chunk = data[start:end]
        if kind == "comment" or (kind == "unknown" and chunk.startswith(b"/*")):
            add(Kind.COMMENT, start, end)
        elif kind == "gap":
            add(Kind.SPACE, start, end)
        elif kind == "unknown" and WHITESPACE.fullmatch(chunk):
            for m in WHITESPACE_PIECE.finditer(data, start, end):
                add(Kind.NEWLINE if m.group("newline") else Kind.SPACE, m.start(), m.end())
        else:
            add(Kind.CODE, start, end)
    return items


def mark_directives(data: bytes, items: list[list]) -> list[list[int]]:
    directives = []
    at_line_start = True
    current = None
    for i, item in enumerate(items):
        kind = item[0]
        if kind is Kind.NEWLINE:
            at_line_start = True
            current = None
            continue
        if kind in (Kind.SPACE, Kind.COMMENT):
            continue
        if at_line_start and data[item[1] : item[2]] in (b"#", b"%:"):
            current = []
            directives.append(current)
        at_line_start = False
        if current is not None:
            item[0] = Kind.DIRECTIVE
            current.append(i)
    return directives


def header_names(data: bytes, items: list[list], directive: list[int]) -> list[tuple[int, int]]:
    words = [data[items[i][1] : items[i][2]] for i in directive]
    if len(words) < 2:
        return []
    found = []
    if words[1] in INCLUDE_NAMES and len(words) > 2 and words[2] == b"<":
        found.append(directive[2])
    elif words[1] in CONDITIONAL_NAMES:
        for k in range(2, len(words) - 2):
            if words[k] in HAS_INCLUDE and words[k + 1] == b"(" and words[k + 2] == b"<":
                found.append(directive[k + 2])
    names = []
    for i in found:
        start = items[i][1]
        line_end = LINE_END.search(data, start)
        gt = data.find(b">", start + 1, line_end.start() if line_end else len(data))
        if gt >= 0:
            names.append((start, gt + 1))
    return names


def merge_header_names(data: bytes, items: list[list], directives: list[list[int]]) -> tuple[list[list], list]:
    names = {}
    for directive in directives:
        for start, end in header_names(data, items, directive):
            names[start] = end
    merged = []
    disturbed = []
    i = 0
    while i < len(items):
        item = items[i]
        end = names.get(item[1]) if item[0] is Kind.DIRECTIVE else None
        j = i
        while end is not None and items[j][2] < end:
            j += 1
        if end is None or items[j][2] != end:
            if end is not None:
                disturbed.append((item[1], end))
            merged.append(item)
            i += 1
            continue
        merged.append([Kind.DIRECTIVE, item[1], end])
        i = j + 1
    return merged, disturbed


def mask(data: bytes, names: list[tuple[int, int]]) -> bytes:
    out = bytearray(data)
    for start, end in names:
        out[start + 1 : end - 1] = b"_" * (end - start - 2)
    return bytes(out)


def lex(text: str) -> list[Token]:
    data = text.encode("utf-8", "surrogateescape")
    starts = line_starts(data)
    masked = data
    masked_names: set = set()
    while True:
        items = classify(data, spans_of(data, raw_records(masked), starts))
        items, disturbed = merge_header_names(data, items, mark_directives(data, items))
        if not disturbed:
            break
        fresh = [name for name in disturbed if name not in masked_names]
        if not fresh:
            raise LexError(f"masking a header name did not settle the {CLANG} token stream")
        masked_names.update(fresh)
        masked = mask(masked, fresh)
    tokens = [
        Token(kind, data[start:end].decode("utf-8", "surrogateescape"), start, bisect.bisect_right(starts, start))
        for kind, start, end in items
    ]
    if "".join(t.text for t in tokens) != text:
        raise LexError("token texts do not tile the file")
    return tokens
