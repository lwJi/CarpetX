from __future__ import annotations

import subprocess
import unittest
from collections import Counter

from files import PROJECT, Lang, decode, lang_of, parallel, resolve, split_z, thorns_in
from rules import lex, units
from tokens import Kind, join

ET0 = "6c59144364fb5e65c7355d4a445530b8503460db"
HAVE_ET0 = resolve(PROJECT, ET0) == ET0


def blobs(rev: str, names: list[str]) -> dict:
    request = "".join(f"{rev}:{name}\n" for name in names).encode("utf-8", "surrogateescape")
    data = subprocess.run(
        ["git", "-C", PROJECT, "cat-file", "--batch"], input=request, capture_output=True, check=True
    ).stdout
    found = {}
    pos = 0
    for name in names:
        header_end = data.index(b"\n", pos)
        _, kind, size = data[pos:header_end].split(b" ")
        if kind != b"blob":
            raise RuntimeError(f"{rev}:{name} is a {kind.decode()}, not a blob")
        start = header_end + 1
        found[name] = decode(data[start : start + int(size)])
        pos = start + int(size) + 1
    return found


def corpus(rev: str) -> list[tuple[str, Lang, str]]:
    names = split_z(subprocess.run(
        ["git", "-C", PROJECT, "ls-tree", "-r", "-z", "--name-only", rev], capture_output=True, check=True
    ).stdout)
    thorns = thorns_in(names)
    covered = [(name, lang_of(name)) for name in names if name.split("/")[0] in thorns and lang_of(name)]
    texts = blobs(rev, [name for name, _ in covered])
    return [(name, lang, texts[name]) for name, lang in covered]


@unittest.skipUnless(HAVE_ET0, f"needs ET0 ({ET0[:9]}) in the repository's history")
class Et0(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.files = corpus(ET0)

        def lexed(item):
            name, lang, text = item
            try:
                tokens = lex(text, lang)
                return name, lang, text, tokens, units(text, lang), None
            except Exception as e:
                return name, lang, text, None, None, f"{type(e).__name__}: {e}"

        cls.results = parallel(lexed, cls.files)

    def test_the_corpus_holds_every_covered_kind(self):
        kinds = Counter(lang for _, lang, _ in self.files)
        self.assertEqual(set(kinds), set(Lang))
        self.assertGreater(kinds[Lang.CXX], 150)
        self.assertGreater(kinds[Lang.PAR], 100)

    def test_every_file_lexes_and_checks(self):
        self.assertEqual([(name, why) for name, _, _, _, _, why in self.results if why], [])

    def test_tiling(self):
        broken = [name for name, _, text, tokens, _, _ in self.results if tokens is not None and join(tokens) != text]
        self.assertEqual(broken, [])

    def test_offsets_and_lines(self):
        wrong = []
        for name, _, text, tokens, _, _ in self.results:
            if tokens is None:
                continue
            offset, line = 0, 1
            for t in tokens:
                if (t.start, t.line) != (offset, line):
                    wrong.append(f"{name}:{t.line}")
                    break
                offset += len(t.text.encode("utf-8", "surrogateescape"))
                line += t.text.count("\n")
        self.assertEqual(wrong, [])

    def test_comment_tokens_start_with_an_opener(self):
        openers = {Lang.CXX: ("//", "/*"), Lang.FORTRAN: ("!",), Lang.SCHEDULE: ("#", "/*")}
        wrong = []
        for name, lang, _, tokens, _, _ in self.results:
            for t in tokens or []:
                if t.kind is Kind.COMMENT and not t.text.startswith(openers.get(lang, ("#",))):
                    wrong.append(f"{name}:{t.line}")
        self.assertEqual(wrong, [])


if __name__ == "__main__":
    unittest.main()
