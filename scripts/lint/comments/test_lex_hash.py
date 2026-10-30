from __future__ import annotations

import unittest

from files import Lang
from .lex_hash import lex, may_hold_comments
from .tokens import Kind, join

CCL_LANGS = (Lang.INTERFACE, Lang.PARAM, Lang.SCHEDULE, Lang.CONFIGURATION, Lang.TEST)
ALL_LANGS = CCL_LANGS + (Lang.PAR, Lang.MAKE, Lang.FORTRAN)


def comments(text, lang):
    return [t.text for t in lex(text, lang) if t.kind is Kind.COMMENT]


def kinds(text, lang):
    return [(t.kind, t.text) for t in lex(text, lang) if t.kind is not Kind.SPACE]


class Tiling(unittest.TestCase):
    def test_the_tokens_join_back_into_the_file(self):
        for text in [
            "",
            "# a",
            "a # b\r\nc\r\n",
            "x\ry # d\n",
            'A "b\n# c" \'d\' # e\n',
            "/* a\n# b */ c # d",
            '"unterminated\n# x',
            "'unterminated # x\n",
            "s = 'abc&\n! note\n   &def ! x' ! z\n",
            "#define A \\\n  1\n!$omp do ! c\n",
            "X = $(a # b\n# c \\\n d\n",
            "é # ü\n\f\v\t# x",
        ]:
            for lang in ALL_LANGS:
                with self.subTest(text=text, lang=lang):
                    self.assertEqual(join(lex(text, lang)), text)

    def test_starts_are_byte_offsets_and_lines_count_newlines(self):
        text = "é # c\nA::b = \"x\n  # d\"\n"
        tokens = lex(text, Lang.PAR)
        data = text.encode("utf-8")
        for t in tokens:
            self.assertEqual(data[t.start : t.start + len(t.text.encode("utf-8"))].decode("utf-8"), t.text)
        self.assertEqual([(t.text, t.line) for t in tokens if t.kind is Kind.COMMENT], [("# c", 1), ("# d", 3)])

    def test_whitespace_splits_into_spaces_and_newlines(self):
        self.assertEqual(
            [(t.kind, t.text) for t in lex("a \t b\r\n\n  # c\n", Lang.PARAM)],
            [(Kind.CODE, "a"), (Kind.SPACE, " \t "), (Kind.CODE, "b"), (Kind.NEWLINE, "\r\n"), (Kind.NEWLINE, "\n"),
             (Kind.SPACE, "  "), (Kind.COMMENT, "# c"), (Kind.NEWLINE, "\n")],
        )


class Ccl(unittest.TestCase):
    def test_whole_line_and_trailing(self):
        text = '# header\nSCHEDULE X AT Y\n{\n  LANG: C # why\n} "desc"\n'
        for lang in CCL_LANGS:
            with self.subTest(lang=lang):
                self.assertEqual(comments(text, lang), ["# header", "# why"])

    def test_a_hash_inside_double_quotes_is_content(self):
        for lang in (Lang.INTERFACE, Lang.PARAM, Lang.SCHEDULE, Lang.TEST):
            with self.subTest(lang=lang):
                self.assertEqual(comments('} "desc # not a comment" # yes\n', lang), ["# yes"])

    def test_a_description_spanning_lines_is_one_code_token(self):
        text = 'BOOLEAN b "line one\n# still the string"\n{\n} no # c\n'
        self.assertEqual(comments(text, Lang.PARAM), ["# c"])
        self.assertIn((Kind.CODE, '"line one\n# still the string"'), kinds(text, Lang.PARAM))

    def test_single_quoted_tags_in_interface_ccl(self):
        text = "CCTK_REAL x TYPE=gf TAGS='a=\"#b\" c=\"#d\"' # e\n"
        self.assertEqual(comments(text, Lang.INTERFACE), ["# e"])
        self.assertIn((Kind.CODE, "TAGS='a=\"#b\" c=\"#d\"'"), kinds(text, Lang.INTERFACE))

    def test_block_comments_only_in_schedule_ccl(self):
        text = "/* a\n b */ SCHEDULE X AT Y\n"
        self.assertEqual(comments(text, Lang.SCHEDULE), ["/* a\n b */"])
        for lang in (Lang.INTERFACE, Lang.PARAM, Lang.CONFIGURATION, Lang.TEST):
            with self.subTest(lang=lang):
                self.assertEqual(comments(text, lang), [])

    def test_a_block_comment_inside_a_string_is_content(self):
        self.assertEqual(comments('} "a /* b */ c"\n', Lang.SCHEDULE), [])

    def test_a_hash_comment_is_not_continued_by_a_backslash(self):
        self.assertEqual(comments("# a \\\nb # c\n", Lang.PARAM), ["# a \\", "# c"])

    def test_configuration_ccl_has_no_strings(self):
        self.assertEqual(comments('REQUIRES "x # y"\n', Lang.CONFIGURATION), ['# y"'])

    def test_test_ccl_strings_end_at_the_line_and_backslash_escapes(self):
        self.assertEqual(comments('ABSTOL "a\n# b\nRELTOL \\# c # d\n', Lang.TEST), ["# b", "# d"])
        self.assertEqual(comments("ABSTOL 'it''s # x\n", Lang.TEST), [])

    def test_line_numbers(self):
        self.assertEqual([t.line for t in lex("a\n\n# b\nc # d\n", Lang.SCHEDULE) if t.kind is Kind.COMMENT], [3, 4])

    def test_crlf_is_not_part_of_the_comment(self):
        self.assertEqual(comments("# a\r\nx\r\n", Lang.INTERFACE), ["# a"])


class Par(unittest.TestCase):
    def test_whole_line_and_trailing(self):
        text = "#CarpetX::prolongation_order = 5\nA::b = 1  # c\n"
        self.assertEqual(comments(text, Lang.PAR), ["#CarpetX::prolongation_order = 5", "# c"])

    def test_a_hash_inside_a_string_is_content(self):
        self.assertEqual(comments('IO::out_dir = "x # y"  # z\n', Lang.PAR), ["# z"])

    def test_a_hash_after_the_first_line_of_a_string_is_a_comment(self):
        self.assertEqual(comments('A::v = "\n  x\n#  y\n"\n# z\n', Lang.PAR), ["#  y", "# z"])
        text = 'ActiveThorns = "\n    # SystemTopology\n    ADMBase  # why\n"\n'
        self.assertEqual(comments(text, Lang.PAR), ["# SystemTopology", "# why"])

    def test_a_multi_line_string_keeps_its_line_structure(self):
        text = 'ActiveThorns = "\n    # SystemTopology\n    ADMBase  # why\n"\n'
        self.assertEqual(
            kinds(text, Lang.PAR),
            [(Kind.CODE, "ActiveThorns"), (Kind.CODE, "="), (Kind.CODE, '"'), (Kind.NEWLINE, "\n"),
             (Kind.COMMENT, "# SystemTopology"), (Kind.NEWLINE, "\n"), (Kind.CODE, "ADMBase"), (Kind.COMMENT, "# why"),
             (Kind.NEWLINE, "\n"), (Kind.CODE, '"'), (Kind.NEWLINE, "\n")],
        )

    def test_a_hash_on_the_first_line_of_a_string_is_content(self):
        self.assertEqual(comments('A::v = "x # y\n  z # w\n"\n', Lang.PAR), ["# w"])
        self.assertEqual(comments('ActiveThorns = "# first\n  CarpetX\n"\n', Lang.PAR), [])

    def test_a_string_comment_ends_at_the_closing_quote(self):
        self.assertEqual(comments('A::v = "a\n  # b" # c\n', Lang.PAR), ["# b", "# c"])
        self.assertEqual(comments('A::v = "a\n  # b \\" c"\n', Lang.PAR), ['# b \\" c'])

    def test_a_hash_glued_to_text_on_a_later_line_is_a_comment(self):
        self.assertEqual(comments('A::v = "a\n  b#c\n"\n', Lang.PAR), ["#c"])

    def test_a_desc_string_is_not_reparsed(self):
        self.assertEqual(comments('!DESC "a\n # b" # c\n', Lang.PAR), ["# c"])
        self.assertEqual(comments('!DESC # x\n"a\n # b"\n', Lang.PAR), ["# x"])

    def test_string_comments_are_par_only(self):
        self.assertEqual(comments('STRING s "a\n  # b"\n{\n} ""\n', Lang.PARAM), [])

    def test_escaped_quote(self):
        self.assertEqual(comments('A::v = "a \\" # b" # c\n', Lang.PAR), ["# c"])

    def test_single_quotes_and_slashes_are_not_special(self):
        self.assertEqual(
            comments("A::p = /a/b # c\n# it's\nA::q = 1 # d\n", Lang.PAR), ["# c", "# it's", "# d"]
        )


class Make(unittest.TestCase):
    def test_whole_line_and_trailing(self):
        text = "# Main make.code.defn file\n\nSRCS = a.cxx # why\nSUBDIRS =\n"
        self.assertEqual(comments(text, Lang.MAKE), ["# Main make.code.defn file", "# why"])

    def test_a_backslash_newline_continues_a_comment(self):
        self.assertEqual(comments("SRCS = a.cxx \\\n  b.cxx # c \\\n  d.cxx\nX = 1\n", Lang.MAKE), ["# c \\\n  d.cxx"])
        self.assertEqual(comments("# a \\\\\nX = 1\n", Lang.MAKE), ["# a \\\\"])

    def test_an_escaped_hash_is_literal(self):
        self.assertEqual(comments("X = \\#1 # c\n", Lang.MAKE), ["# c"])
        self.assertEqual(comments("X = \\\\# c\n", Lang.MAKE), ["# c"])

    def test_a_hash_inside_a_variable_reference_is_literal(self):
        self.assertEqual(comments("X = $(info #) ${y #} $# # c\n", Lang.MAKE), ["# c"])
        self.assertEqual(comments("X = $(a (b #) c) # d\n", Lang.MAKE), ["# d"])

    def test_an_unterminated_reference_ends_with_its_line(self):
        self.assertEqual(comments("X = $(a # b\n# c\n", Lang.MAKE), ["# c"])

    def test_line_continuations_are_code(self):
        self.assertEqual(
            kinds("SRCS = \\\n\ta.cxx\n", Lang.MAKE),
            [(Kind.CODE, "SRCS"), (Kind.CODE, "="), (Kind.CODE, "\\"), (Kind.NEWLINE, "\n"), (Kind.CODE, "a.cxx"),
             (Kind.NEWLINE, "\n")],
        )


class Fortran(unittest.TestCase):
    def test_whole_line_and_trailing(self):
        text = "  !// test array group\n  x = 1 ! why\n"
        self.assertEqual(comments(text, Lang.FORTRAN), ["!// test array group", "! why"])

    def test_bang_inside_quotes(self):
        self.assertEqual(comments('call CCTK_WARN(1, "a ! b") ! c\n', Lang.FORTRAN), ["! c"])
        self.assertEqual(comments("print *, 'it''s ! not' ! yes\n", Lang.FORTRAN), ["! yes"])
        self.assertEqual(comments("print *, '\"!\"', \"'!'\" ! c\n", Lang.FORTRAN), ["! c"])

    def test_an_unterminated_string_ends_at_the_line_end(self):
        self.assertEqual(comments("s = 'a & b\n! c\n", Lang.FORTRAN), ["! c"])

    def test_a_string_continued_with_an_ampersand(self):
        text = "s = 'abc&\n! note\n\n   &def ! x' ! why\nt = \"a&\n  b ! y\" ! z\n"
        self.assertEqual(comments(text, Lang.FORTRAN), ["! note", "! why", "! z"])

    def test_a_continued_string_without_a_leading_ampersand_resumes_at_column_one(self):
        self.assertEqual(comments("s = 'abc&\nd ! x' ! y\n", Lang.FORTRAN), ["! y"])
        self.assertEqual(comments("s = 'abc&\n'! x\n", Lang.FORTRAN), ["! x"])

    def test_preprocessor_lines_hold_no_comments(self):
        text = "#include <cctk.h>\n  #if !defined(X)\n#define Y \\\n  !z\n#endif\n"
        self.assertEqual(comments(text, Lang.FORTRAN), [])
        self.assertEqual([k for k, _ in kinds(text, Lang.FORTRAN) if k is not Kind.NEWLINE], [Kind.DIRECTIVE] * 4)

    def test_openmp_sentinels_are_directives(self):
        text = "!$omp parallel do\n  !$ n = omp_get_num_threads() ! why\n! $omp is prose\nx = 1 !$omp after code\n"
        self.assertEqual(comments(text, Lang.FORTRAN), ["! why", "! $omp is prose", "!$omp after code"])
        self.assertEqual([t for k, t in kinds(text, Lang.FORTRAN) if k is Kind.DIRECTIVE], ["!$omp", "!$"])

    def test_operators_are_not_comments(self):
        self.assertEqual(comments("if (a /= b) s = 'x' // 'y'\n", Lang.FORTRAN), [])


class Hints(unittest.TestCase):
    def test_a_file_without_an_opener_needs_no_lexing(self):
        self.assertFalse(may_hold_comments("SCHEDULE X AT Y\n", Lang.SCHEDULE))
        self.assertTrue(may_hold_comments("/* x */", Lang.SCHEDULE))
        self.assertFalse(may_hold_comments("/* x */", Lang.PARAM))
        self.assertTrue(may_hold_comments("x # y", Lang.PAR))
        self.assertFalse(may_hold_comments("x # y", Lang.FORTRAN))
        self.assertTrue(may_hold_comments("x ! y", Lang.FORTRAN))


if __name__ == "__main__":
    unittest.main()
