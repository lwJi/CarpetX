from __future__ import annotations

import unittest

from boundary.check import CHECK as boundary
from boundary.rules import Finding
from comments.check import CHECK as comments
from files import Lang
from run import judged

PRIVATE = '#include "../../CarpetX/src/driver.hxx"\n'


def found(verdict):
    return [("new", u.line, u.text) for u in verdict.added] + [("touched", t.first, t.last) for t in verdict.touched]


def comments_on(befores, after, lang=Lang.CXX):
    entries = judged("T/src/a.cxx", lambda: ((name, text) for name, text in befores), after, lang, [comments])
    return [found(verdict) for _, _, verdict in entries]


class EveryStartingTree(unittest.TestCase):
    def test_a_comment_another_starting_tree_has(self):
        after = "// up\nint x;\n"
        self.assertEqual(comments_on([("a", "int x;\n"), ("a", after)], after), [])
        self.assertEqual(comments_on([("a", "int x;\n"), ("a", "int x;\n")], after), [[("new", 1, "// up")]])

    def test_code_another_starting_tree_changed(self):
        after = "// a\nint x = 2;\n"
        self.assertEqual(comments_on([("a", "// a\nint x = 1;\n"), ("a", after)], after), [])

    def test_the_first_tree_names_what_is_reported(self):
        befores = [("a", "// a\nint x = 1;\n"), ("a", "int x = 1;\n")]
        self.assertEqual(comments_on(befores, "// a\nint x = 2;\n"), [[("touched", 1, 1)]])

    def test_no_starting_tree_is_clean(self):
        self.assertEqual(comments_on([], "// why\nint x;\n"), [])

    def test_a_finding_another_starting_tree_has_is_not_reported(self):
        after = PRIVATE + "int x;\n"
        entries = judged("T/src/a.cxx", lambda: iter([("a", "int x;\n"), ("a", after)]), after, Lang.CXX, [boundary])
        self.assertEqual(entries, [])
        entries = judged("T/src/a.cxx", lambda: iter([("a", "int x;\n"), ("a", "int x;\n")]), after, Lang.CXX, [boundary])
        self.assertEqual(entries, [(boundary, "T/src/a.cxx", [Finding(1, "include", PRIVATE.strip())])])


class EveryCheck(unittest.TestCase):
    def test_each_starting_tree_is_read_once_for_every_check(self):
        reads = []

        def befores():
            for text in ["int x;\n", "int x;\n"]:
                reads.append(text)
                yield "a", text

        after = PRIVATE + "// why\nint x;\n"
        entries = judged("T/src/a.cxx", befores, after, Lang.CXX, [comments, boundary])
        self.assertEqual([check.name for check, _, _ in entries], ["no-comments", "boundary"])
        self.assertEqual(len(reads), 2)

    def test_reading_stops_when_every_check_is_clear(self):
        reads = []

        def befores():
            for text in ["int x;\n", "never read\n"]:
                reads.append(text)
                yield "a", text

        self.assertEqual(judged("T/src/a.cxx", befores, "int x;\nint y;\n", Lang.CXX, [comments, boundary]), [])
        self.assertEqual(reads, ["int x;\n"])

    def test_a_check_that_does_not_cover_the_file_reads_nothing(self):
        reads = []

        def befores():
            reads.append("read")
            yield "a", "x = 1\n"

        [(check, _, verdict)] = judged("T/test/t.par", befores, "x = 1 # why\n", Lang.PAR, [comments, boundary])
        self.assertIs(check, comments)
        self.assertEqual(found(verdict), [("new", 1, "# why")])
        self.assertEqual(reads, ["read"])


if __name__ == "__main__":
    unittest.main()
