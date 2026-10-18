import unittest

from files import Lang
from report import RULE, excerpt, report, summary
from rules import check


class Report(unittest.TestCase):
    def test_clean(self):
        self.assertEqual(report([]), "✓ no-comments: 0 comments outside the allowed exceptions")

    def test_lists_each_violation_then_the_rule_then_the_count(self):
        text = "int a; // why\n// first line\n// second line\nint b;\n"
        found = [("X/src/a.cxx", v) for v in check(text, Lang.CXX)]
        lines = report(found).split("\n")
        self.assertEqual(lines[0], "  X/src/a.cxx:1   // why")
        self.assertEqual(lines[1], "  X/src/a.cxx:2   // first line…")
        self.assertEqual(lines[3:6], [f"  {line}" for line in RULE.split("\n")])
        self.assertEqual(lines[-1], "✗ no-comments: 2 comments outside the allowed exceptions")

    def test_full_prints_whole_groups(self):
        found = [("X/a.par", v) for v in check("# one\n# two\nA::b = 1\n", Lang.PAR)]
        self.assertEqual(report(found, full=True).split("\n")[:3],
                         ["  X/a.par:1", "    # one", "    # two"])

    def test_long_comments_are_cut(self):
        [v] = check("// " + "x" * 100 + "\n", Lang.CXX)
        self.assertTrue(excerpt(v).endswith("…"))
        self.assertLessEqual(len(excerpt(v)), 73)

    def test_summary_grammar(self):
        self.assertEqual(summary(1), "✗ no-comments: 1 comment outside the allowed exceptions")


if __name__ == "__main__":
    unittest.main()
