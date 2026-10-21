from __future__ import annotations

import unittest

from files import Lang
from spans import layout


def spans(text, lang=Lang.CXX):
    return [(c.first, c.last, c.end) for c in layout(text, lang).comments]


def texts(text, lang=Lang.CXX):
    found = layout(text, lang)
    return [[found.units[i].text for i in c.units] for c in found.comments]


class Grouping(unittest.TestCase):
    def test_adjacent_comment_lines_are_one_comment(self):
        self.assertEqual(spans("// a\n// b\n//\n// c\nint x;\n"), [(1, 4, 5)])

    def test_a_blank_line_or_code_starts_a_new_one(self):
        self.assertEqual(spans("// a\n\n// b\nint x;\n// c\n"), [(1, 1, 1), (3, 3, 4), (5, 5, 5)])

    def test_a_trailing_comment_and_the_aligned_lines_below_it_are_one_comment(self):
        self.assertEqual(spans("int a; // a\nint b; // b\n       // c\nint d;\n"), [(1, 1, 1), (2, 3, 3)])

    def test_a_comment_below_a_trailing_one_that_is_not_aligned_heads_the_code_below(self):
        self.assertEqual(spans("int a; // a\n// b\nint b;\n"), [(1, 1, 1), (2, 2, 3)])

    def test_a_block_comment_is_one_comment(self):
        self.assertEqual(spans("/* a\n   b */\n// c\nint x; /* d\n e */\n"), [(1, 3, 5), (4, 5, 5)])

    def test_comments_on_one_line_are_one_comment(self):
        self.assertEqual(spans("/* a */ int x; // b\n"), [(1, 1, 1)])
        self.assertEqual(texts("/* a */ int x; // b\n"), [["/* a */", "// b"]])

    def test_a_comment_holds_its_lines_with_words(self):
        self.assertEqual(texts("// a\n//\n// b\nint x;\n#if 0\n#endif\n"), [["// a", "// b"]])

    def test_hash_languages(self):
        self.assertEqual(spans("# a\n# b\nSRCS = x.cxx # c\n", Lang.MAKE), [(1, 2, 3), (3, 3, 3)])
        self.assertEqual(spans("! a\n  ! b\nx = 1 ! c\n", Lang.FORTRAN), [(1, 2, 3), (3, 3, 3)])


class CodeUnderAComment(unittest.TestCase):
    def test_runs_to_the_next_blank_line(self):
        self.assertEqual(spans("// a\nx = 1;\ny = 2;\n\nz = 3;\n"), [(1, 1, 3)])

    def test_or_to_the_next_comment(self):
        self.assertEqual(spans("// a\nx = 1;\n// b\ny = 2;\n"), [(1, 1, 2), (3, 3, 4)])

    def test_trailing_comments_in_between_do_not_end_it(self):
        self.assertEqual(spans("// a\nint x; // x\n       // more\nint y;\n"), [(1, 1, 4), (2, 3, 3)])

    def test_a_block_it_opens_runs_to_the_closing_brace(self):
        self.assertEqual(spans("// f\nvoid f() {\n  x = 1;\n\n  // y\n  y = 2;\n}\n\nint z;\n")[0], (1, 1, 7))
        self.assertEqual(spans("// f\nextern \"C\" void f() {\n  x;\n\n  y;\n}\n"), [(1, 1, 6)])

    def test_a_comment_inside_a_block_stops_at_the_block_s_end(self):
        self.assertEqual(spans("if (c) {\n  // a\n  x;\n}\ny;\n"), [(2, 2, 3)])
        self.assertEqual(spans("if (c) {\n  // a\n  x;\n} else {\n  y;\n}\n"), [(2, 2, 3)])

    def test_braces_in_strings_and_directives_do_not_count(self):
        self.assertEqual(spans("// a\nf(\"{\");\n#define X {\ng('{');\n\nh();\n"), [(1, 1, 4)])

    def test_a_namespace_or_extern_block_does_not_extend_it(self):
        self.assertEqual(spans("// a\nnamespace N {\n\nint x;\n} // namespace N\n"), [(1, 1, 2), (5, 5, 5)])
        self.assertEqual(spans("// a\nextern \"C\" {\n\nint x;\n}\n"), [(1, 1, 2)])

    def test_a_trailing_comment_covers_its_own_lines(self):
        self.assertEqual(spans("int a; // a\n       // b\nint c;\n"), [(1, 2, 2)])

    def test_a_comment_followed_by_a_blank_line_covers_no_code(self):
        self.assertEqual(spans("// a\n\nint x;\n"), [(1, 1, 1)])

    def test_hash_languages(self):
        schedule = "# a\nschedule X AT Y\n{\n  LANG: C\n\n} \"x {\"\n\nschedule Z AT Y\n"
        self.assertEqual(spans(schedule, Lang.SCHEDULE), [(1, 1, 6)])
        self.assertEqual(spans("# a\nT::x = 1\nT::y = 2\n\nT::z = 3\n", Lang.PAR), [(1, 1, 3)])


class Rows(unittest.TestCase):
    def rows(self, text, lang=Lang.CXX):
        return list(layout(text, lang).rows)

    def test_spacing_and_openers_do_not_count(self):
        self.assertEqual(self.rows("int  a =1; /// a\n"), self.rows("  int a = 1;   // a\n"))

    def test_a_row_holds_the_code_and_the_comment_words(self):
        self.assertEqual(self.rows("int a = 1; // a\n// b\n\n"), [("int a = 1 ;", (("comment", "a"),)),
                                                                  ("", (("comment", "b"),)), ("", ())])

    def test_a_compile_time_message_or_constant_guard_is_not_code(self):
        rows = self.rows("#warning \\\n  \"x\"\n#if 0\nint y;\n#endif\n")
        self.assertEqual([code for code, _ in rows], ["", "", "", "int y ;", "# endif"])
        self.assertEqual([len(keys) for _, keys in rows], [1, 0, 1, 0, 0])

    def test_an_empty_file(self):
        self.assertEqual(self.rows(""), [])
        self.assertEqual(spans(""), [])


if __name__ == "__main__":
    unittest.main()
