from __future__ import annotations

import os
import shutil
import subprocess
import unittest
from collections import Counter

import lex_clang
from files import PROJECT, Lang, blobs, decode, lang_of, parallel, resolve, thorns_in, tree_names
from regions import directive_regions
from rules import check, lex
from strip import strip, unifdef_k
from tokens import Kind, Placement, join
from verify import PEGS, verify_files

ET0 = "6c59144364fb5e65c7355d4a445530b8503460db"
HAVE_CLANG = shutil.which(lex_clang.CLANG) is not None
HAVE_UNIFDEF = shutil.which("unifdef") is not None
HAVE_CLANG_FORMAT = shutil.which("clang-format") is not None
HAVE_ET0 = resolve(PROJECT, ET0) == ET0
FLESH = os.environ.get("CACTUSX")
HAVE_FLESH = (
    shutil.which("perl") is not None
    and FLESH is not None
    and os.path.isfile(os.path.join(FLESH, "lib", "sbin", "Piraha.pm"))
)


def corpus(rev: str) -> list[tuple[str, Lang, str]]:
    names = tree_names(PROJECT, rev)
    thorns = thorns_in(names)
    covered = [(name, lang_of(name)) for name in names if name.split("/")[0] in thorns and lang_of(name)]
    texts = blobs(PROJECT, rev, [name for name, _ in covered])
    return [(name, lang, texts[name]) for name, lang in covered]


@unittest.skipUnless(HAVE_CLANG, "needs clang++ on PATH")
@unittest.skipUnless(HAVE_ET0, f"needs ET0 ({ET0[:9]}) in the repository's history")
class Et0(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.files = corpus(ET0)

        def lexed(item):
            name, lang, text = item
            try:
                tokens = lex(text, lang)
                return name, lang, text, tokens, check(tokens, lang), None
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


def clang_format(name: str, text: str) -> str:
    proc = subprocess.run(
        ["clang-format", f"--assume-filename={os.path.join(PROJECT, name)}"],
        input=text.encode("utf-8", "surrogateescape"), capture_output=True, check=True,
    )
    return decode(proc.stdout)


def layout(text: str) -> list[tuple]:
    found = []
    newlines, spaced = 0, False
    for t in lex_clang.lex(text):
        if t.kind is Kind.NEWLINE:
            newlines += 1
        elif t.kind is Kind.SPACE:
            spaced = True
        else:
            gap = ("blank", newlines - 1) if newlines > 1 else "break" if newlines else "space" if spaced else "none"
            found.append((" ".join(t.text.split()) if t.kind is Kind.COMMENT else t.text, gap, t.line))
            newlines, spaced = 0, False
    return found


def first_layout_difference(a: list[tuple], b: list[tuple]):
    for x, y in zip(a, b):
        rebroken = "break" in (x[1], y[1]) and not any(isinstance(g, tuple) for g in (x[1], y[1]))
        if x[0] != y[0] or (x[1] != y[1] and not rebroken):
            return x, y
    return None if len(a) == len(b) else (len(a), len(b))


@unittest.skipUnless(HAVE_CLANG and HAVE_UNIFDEF, "needs clang++ and unifdef on PATH")
@unittest.skipUnless(HAVE_ET0, f"needs ET0 ({ET0[:9]}) in the repository's history")
class Et0Strip(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        def stripped(item):
            name, lang, text = item
            try:
                return name, lang, text, strip(text, lang), None
            except Exception as e:
                return name, lang, text, None, f"{type(e).__name__}: {e}"

        cls.results = parallel(stripped, corpus(ET0))
        cls.changed = [(n, lang, a, b) for n, lang, a, b, why in cls.results if b is not None and b != a]

    def test_every_file_strips(self):
        self.assertEqual([(name, why) for name, _, _, _, why in self.results if why], [])
        self.assertGreater(len(self.changed), 300)

    def test_guards_are_found_exactly_where_unifdef_folds(self):
        def disagrees(item):
            name, lang, text, _, _ = item
            tokens = lex(text, lang)
            guarded = any(r.placement is Placement.GUARD for r in directive_regions(tokens))
            return name if guarded != (unifdef_k(text, tokens) != text) else None

        cxx = [item for item in self.results if item[1] is Lang.CXX]
        self.assertEqual([name for name in parallel(disagrees, cxx) if name], [])

    def test_strip_leaves_only_compile_time_messages_and_is_idempotent(self):
        def leftover(item):
            name, lang, _, after = item
            found = [v.line for v in check(lex(after, lang), lang) if v.region.placement is not Placement.MESSAGE]
            return name, found, strip(after, lang) == after

        results = parallel(leftover, self.changed)
        self.assertEqual([(name, lines) for name, lines, _ in results if lines], [])
        self.assertEqual([name for name, _, idempotent in results if not idempotent], [])

    def test_verify_passes_for_sources_and_line_audits(self):
        items = [item for item in self.changed if item[1] not in PEGS]
        self.assertEqual([(r.name, r.why) for r in verify_files(items, None) if not r.ok], [])

    @unittest.skipUnless(HAVE_FLESH, "needs perl and a Cactus flesh in $CACTUSX")
    def test_verify_passes_for_ccl_and_par_files(self):
        items = [item for item in self.changed if item[1] in PEGS]
        self.assertEqual([(r.name, r.why) for r in verify_files(items, FLESH) if not r.ok], [])

    @unittest.skipUnless(HAVE_CLANG_FORMAT, "needs clang-format on PATH")
    def test_a_clang_format_clean_file_needs_at_most_new_line_breaks_after_strip(self):
        def difference(item):
            name, lang, before, after = item
            if lang is not Lang.CXX or clang_format(name, before) != before:
                return name, False, None
            return name, True, first_layout_difference(layout(after), layout(clang_format(name, after)))

        results = parallel(difference, self.changed)
        self.assertGreater(sum(clean for _, clean, _ in results), 100)
        self.assertEqual([(name, d) for name, _, d in results if d], [])


if __name__ == "__main__":
    unittest.main()
