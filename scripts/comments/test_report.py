from __future__ import annotations

import unittest

from files import Lang
from judge import Touched, Verdict, judge
from report import CHANGE_RULE, REDO, RULE, change_summary, excerpt, render, render_change, summary
from rules import Unit, units


def found(path, text):
    return [(path, u) for u in units(text, Lang.CXX) if not u.allowed]


def unit(line, text):
    return Unit(line, ("comment", text[3:]), text, False)


class Summary(unittest.TestCase):
    def test_counts(self):
        self.assertEqual(summary(0), "✓ no-comments: 0 comment lines")
        self.assertEqual(summary(1), "✗ no-comments: 1 comment line")
        self.assertEqual(summary(7), "✗ no-comments: 7 comment lines")

    def test_clean(self):
        self.assertEqual(render([]), "✓ no-comments: 0 comment lines")

    def test_change_counts(self):
        self.assertEqual(change_summary([]), "✓ no-comments: no new or touched comments")
        one = Verdict((unit(1, "// a"),), ())
        self.assertEqual(change_summary([("A/src/a.cxx", one)]), "✗ no-comments: 1 new comment line")
        both = Verdict((unit(1, "// a"), unit(2, "// b")), (Touched(5, 6, unit(5, "// c")),))
        self.assertEqual(change_summary([("A/src/a.cxx", both), ("A/src/b.cxx", Verdict((), both.touched))]),
                         "✗ no-comments: 2 new comment lines, 2 touched comments")


class Render(unittest.TestCase):
    def test_the_report_shape(self):
        text = "\n" * 811 + "// Must run before the restriction.\nint a;\n"
        self.assertEqual(
            render(found("CarpetX/src/driver.cxx", text)),
            "CarpetX/src/driver.cxx:812   // Must run before the restriction.\n"
            "Comments in thorn files can only be deleted (see agent_docs/code-comments.md).\n"
            "Put what is still true in code (assert, name, CCTK_VERROR) or in the commit message.\n"
            "✗ no-comments: 1 comment line",
        )

    def test_places_are_aligned_and_the_rule_comes_once(self):
        report = render(found("A/src/a.cxx", "int a; // x\n" + "\n" * 9 + "// y\n") + found("A/src/bb.cxx", "// z\n"))
        lines = report.split("\n")
        self.assertEqual(lines[:3], ["A/src/a.cxx:1    // x", "A/src/a.cxx:11   // y", "A/src/bb.cxx:1   // z"])
        self.assertEqual(lines[3:5], RULE.split("\n"))
        self.assertEqual(lines[-1], "✗ no-comments: 3 comment lines")

    def test_long_comments_are_cut(self):
        [(_, u)] = found("A/src/a.cxx", "// " + "x" * 100 + "\n")
        self.assertEqual(excerpt(u), "// " + "x" * 69 + "…")

    def test_a_guard_or_message_shows_its_directive(self):
        text = "\n#if 0\nint a;\n#endif\n#warning                \\\n    \"TODO: Allow different restriction operators, and ensure this is conservative\"\n"
        self.assertEqual(
            render(found("A/src/a.cxx", text)).split("\n")[:2],
            ["A/src/a.cxx:2   #if 0",
             'A/src/a.cxx:5   #warning "TODO: Allow different restriction operators, and ensure this i…'],
        )


class RenderChange(unittest.TestCase):
    def test_the_report_shape(self):
        before = "// The clock groups,\n// coarse to fine.\nint g = 1;\n"
        after = "// The clock groups,\n// coarse to fine.\nint g = 2;\n" + "\n" * 8 + "int h; // why\n"
        self.assertEqual(
            render_change([("CarpetX/src/schedule.cxx", judge(before, after, Lang.CXX))]).split("\n"),
            ["CarpetX/src/schedule.cxx:1-2   touched   // The clock groups,",
             "CarpetX/src/schedule.cxx:12    new       // why"] + CHANGE_RULE.split("\n")
            + ["✗ no-comments: 1 new comment line, 1 touched comment"],
        )

    def test_entries_come_in_path_and_line_order(self):
        report = render_change([
            ("B/src/b.cxx", Verdict((unit(3, "// b"),), ())),
            ("A/src/a.cxx", Verdict((unit(9, "// y"),), (Touched(2, 2, unit(2, "// x")),))),
        ])
        self.assertEqual(report.split("\n")[:3], ["A/src/a.cxx:2   touched   // x",
                                                  "A/src/a.cxx:9   new       // y",
                                                  "B/src/b.cxx:3   new       // b"])

    def test_an_edit_is_told_to_redo_itself(self):
        lines = render_change([("A/src/a.cxx", Verdict((unit(1, "// a"),), ()))], edit=True).split("\n")
        self.assertEqual(lines[-2:], [REDO, "✗ no-comments: 1 new comment line"])

    def test_clean(self):
        self.assertEqual(render_change([]), "✓ no-comments: no new or touched comments")


if __name__ == "__main__":
    unittest.main()
