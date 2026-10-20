from __future__ import annotations

import shutil
import unittest

import lex_clang
from files import Lang
from report import RULE, excerpt, render, summary
from rules import check_text

HAVE_CLANG = shutil.which(lex_clang.CLANG) is not None


def found(path, text):
    return [(path, v) for v in check_text(text, Lang.CXX)]


class Summary(unittest.TestCase):
    def test_counts(self):
        self.assertEqual(summary(0), "✓ no-comments: 0 comments")
        self.assertEqual(summary(1), "✗ no-comments: 1 comment")
        self.assertEqual(summary(7), "✗ no-comments: 7 comments")

    def test_clean(self):
        self.assertEqual(render([]), "✓ no-comments: 0 comments")


@unittest.skipUnless(HAVE_CLANG, "needs clang++ on PATH")
class Render(unittest.TestCase):
    def test_the_report_shape(self):
        text = "\n" * 811 + "// Must run before the restriction.\nint a;\n"
        self.assertEqual(
            render(found("CarpetX/src/driver.cxx", text)),
            "CarpetX/src/driver.cxx:812   // Must run before the restriction.\n"
            "Don't add comments to thorn files (see \"Code comments\" in CLAUDE.md).\n"
            "Put a true invariant in code (assert, name, CCTK_VERROR) or in the commit message.\n"
            "✗ no-comments: 1 comment",
        )

    def test_places_are_aligned_and_the_rule_comes_once(self):
        report = render(found("A/src/a.cxx", "int a; // x\n" + "\n" * 9 + "// y\n") + found("A/src/bb.cxx", "// z\n"))
        lines = report.split("\n")
        self.assertEqual(lines[:3], ["A/src/a.cxx:1    // x", "A/src/a.cxx:11   // y", "A/src/bb.cxx:1   // z"])
        self.assertEqual(lines[3:5], RULE.split("\n"))
        self.assertEqual(lines[-1], "✗ no-comments: 3 comments")

    def test_a_run_shows_its_first_line(self):
        [(_, v)] = found("A/src/a.cxx", "// first line\n// second line\nint b;\n")
        self.assertEqual(excerpt(v), "// first line…")

    def test_long_comments_are_cut(self):
        [(_, v)] = found("A/src/a.cxx", "// " + "x" * 100 + "\n")
        self.assertEqual(excerpt(v), "// " + "x" * 69 + "…")

    def test_a_guard_or_message_shows_its_directive(self):
        text = "\n#if 0\nint a;\n#endif\n#warning                \\\n    \"TODO: Allow different restriction operators, and ensure this is conservative\"\n"
        self.assertEqual(
            render(found("A/src/a.cxx", text)).split("\n")[:2],
            ["A/src/a.cxx:2   #if 0",
             'A/src/a.cxx:5   #warning "TODO: Allow different restriction operators, and ensure this i…'],
        )


if __name__ == "__main__":
    unittest.main()
