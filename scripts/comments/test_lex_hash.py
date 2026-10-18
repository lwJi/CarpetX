import unittest

from files import Lang
from lex_hash import lex


def spans(text, lang):
    return [(s.kind, s.text) for s in lex(text, lang)]


class Ccl(unittest.TestCase):
    def test_full_line_and_trailing(self):
        text = '# header\nSCHEDULE X AT Y\n{\n  LANG: C # why\n} "desc"\n'
        self.assertEqual(spans(text, Lang.CCL), [("hash", "# header"), ("hash", "# why")])

    def test_hash_inside_double_quotes(self):
        self.assertEqual(spans('} "desc # not a comment"\n', Lang.CCL), [])

    def test_description_spanning_lines(self):
        self.assertEqual(spans('BOOLEAN b "line one\n# still the string"\n{\n} no\n', Lang.CCL), [])

    def test_hash_inside_single_quoted_tags(self):
        text = "CCTK_REAL x TYPE=gf TAGS='a=\"#b\" c=\"#d\"' # e\n"
        self.assertEqual(spans(text, Lang.CCL), [("hash", "# e")])

    def test_block_comments(self):
        self.assertEqual(spans("/* a\n b */ SCHEDULE X AT Y\n", Lang.CCL), [("block", "/* a\n b */")])

    def test_line_numbers(self):
        self.assertEqual([s.line for s in lex("a\n\n# b\nc # d\n", Lang.CCL)], [3, 4])

    def test_crlf_is_not_part_of_the_comment(self):
        self.assertEqual(spans("# a\r\nx\r\n", Lang.CCL), [("hash", "# a")])


class Par(unittest.TestCase):
    def test_full_line_and_trailing(self):
        text = '#CarpetX::prolongation_order = 5\nA::b = 1  # c\n'
        self.assertEqual(spans(text, Lang.PAR),
                         [("hash", "#CarpetX::prolongation_order = 5"), ("hash", "# c")])

    def test_hash_inside_a_string(self):
        self.assertEqual(spans('IO::out_dir = "x # y"  # z\n', Lang.PAR), [("hash", "# z")])

    def test_hash_after_the_first_line_of_a_string_is_a_comment(self):
        self.assertEqual(spans('A::v = "\n  x\n#  y\n"\n# z\n', Lang.PAR), [("hash", "#  y"), ("hash", "# z")])
        text = 'ActiveThorns = "\n    # SystemTopology\n    ADMBase  # why\n"\n'
        self.assertEqual(spans(text, Lang.PAR), [("hash", "# SystemTopology"), ("hash", "# why")])

    def test_hash_on_the_first_line_of_a_string_is_content(self):
        self.assertEqual(spans('A::v = "x # y\n  z # w\n"\n', Lang.PAR), [("hash", "# w")])
        self.assertEqual(spans('ActiveThorns = "# first\n  CarpetX\n"\n', Lang.PAR), [])

    def test_a_string_comment_ends_at_the_closing_quote(self):
        self.assertEqual(spans('A::v = "a\n  # b" # c\n', Lang.PAR), [("hash", "# b"), ("hash", "# c")])
        self.assertEqual(spans('A::v = "a\n  # b \\" c"\n', Lang.PAR), [("hash", '# b \\" c')])

    def test_string_comments_are_par_only(self):
        self.assertEqual(spans('STRING s "a\n  # b"\n{\n} ""\n', Lang.CCL), [])

    def test_escaped_quote(self):
        self.assertEqual(spans('A::v = "a \\" # b" # c\n', Lang.PAR), [("hash", "# c")])

    def test_single_quotes_and_slashes_are_not_special(self):
        self.assertEqual(spans("A::p = /a/b # c\n# it's\nA::q = 1 # d\n", Lang.PAR),
                         [("hash", "# c"), ("hash", "# it's"), ("hash", "# d")])


class Fortran(unittest.TestCase):
    def test_full_line_and_trailing(self):
        text = "  !// test array group\n  x = 1 ! why\n"
        self.assertEqual(spans(text, Lang.FORTRAN), [("line", "!// test array group"), ("line", "! why")])

    def test_bang_inside_quotes(self):
        self.assertEqual(spans('call CCTK_WARN(1, "a ! b") ! c\n', Lang.FORTRAN), [("line", "! c")])
        self.assertEqual(spans("print *, 'it''s ! not' ! yes\n", Lang.FORTRAN), [("line", "! yes")])
        self.assertEqual(spans("print *, '\"!\"', \"'!'\" ! c\n", Lang.FORTRAN), [("line", "! c")])

    def test_an_unterminated_string_ends_at_the_line_end(self):
        self.assertEqual(spans("s = 'a &\n! c\n", Lang.FORTRAN), [("line", "! c")])

    def test_preprocessor_lines_hold_no_comments(self):
        self.assertEqual(spans("#include <cctk.h>\n  #if !defined(X)\n#endif\n", Lang.FORTRAN), [])

    def test_openmp_sentinels_are_directives(self):
        text = "!$omp parallel do\n!$ n = omp_get_num_threads()\n! $omp is prose\n"
        self.assertEqual(spans(text, Lang.FORTRAN), [("line", "! $omp is prose")])

    def test_operators_are_not_comments(self):
        self.assertEqual(spans("if (a /= b) s = 'x' // 'y'\n", Lang.FORTRAN), [])


if __name__ == "__main__":
    unittest.main()
