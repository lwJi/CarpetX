from __future__ import annotations

import os
import shutil
import unittest
from unittest import mock

import lex_clang
from lex_clang import ToolMissing, lex
from tokens import Kind, join

HAVE_CLANG = shutil.which(lex_clang.CLANG) is not None


def kinds(text):
    return [(t.kind, t.text) for t in lex(text) if t.kind not in (Kind.SPACE, Kind.NEWLINE)]


def comments(text):
    return [t.text for t in lex(text) if t.kind is Kind.COMMENT]


@unittest.skipUnless(HAVE_CLANG, "needs clang++ on PATH")
class Tiling(unittest.TestCase):
    def test_the_tokens_join_back_into_the_file(self):
        for text in [
            "",
            "int a;",
            "int a; // x\n",
            "int a;\r\nint b; // c\r\n",
            "x\ry // d\rz\n",
            "a\n\rb\r\r\nc\n",
            "﻿int a; // bom\n",
            "int é = 1; x ⁻¹ y; // m s⁻¹\n",
            "auto s = \"\udcff\"; // \udcfe\n",
            "/* unterminated\nint a;\n",
            "#define F(x) \\\n  x // y\n\t\r\nint z;",
            "\f\vint a;\v\n",
        ]:
            with self.subTest(text=text):
                self.assertEqual(join(lex(text)), text)

    def test_starts_are_byte_offsets(self):
        text = "é; // c\n"
        tokens = lex(text)
        data = text.encode("utf-8")
        for t in tokens:
            self.assertEqual(data[t.start : t.start + len(t.text.encode("utf-8"))].decode("utf-8"), t.text)

    def test_whitespace_splits_into_spaces_and_newlines(self):
        self.assertEqual(
            [(t.kind, t.text) for t in lex("a\n\n  b\r\n")],
            [(Kind.CODE, "a"), (Kind.NEWLINE, "\n"), (Kind.NEWLINE, "\n"), (Kind.SPACE, "  "), (Kind.CODE, "b"),
             (Kind.NEWLINE, "\r\n")],
        )

    def test_a_line_splice_is_space_not_a_newline(self):
        self.assertEqual(
            [(t.kind, t.text) for t in lex("a \\\n  b")],
            [(Kind.CODE, "a"), (Kind.SPACE, " \\\n  "), (Kind.CODE, "b")],
        )

    def test_a_leading_bom_is_space(self):
        self.assertEqual(lex("﻿// c\n")[0].kind, Kind.SPACE)


@unittest.skipUnless(HAVE_CLANG, "needs clang++ on PATH")
class Comments(unittest.TestCase):
    def test_line_and_block_comments(self):
        self.assertEqual(comments("int a; // x\n/* y */ int b;\n"), ["// x", "/* y */"])

    def test_line_numbers(self):
        self.assertEqual([t.line for t in lex("int a;\n\n// x\nint b; /* y */\n") if t.kind is Kind.COMMENT], [3, 4])
        self.assertEqual([t.line for t in lex("a\rb // c\r\n// d\n") if t.kind is Kind.COMMENT], [2, 3])

    def test_non_ascii_text_keeps_positions(self):
        self.assertEqual(comments("// é\nauto s = \"⁻¹\"; // m s⁻¹\n"), ["// é", "// m s⁻¹"])
        self.assertEqual([t.line for t in lex("// é\nint a; // ü\n") if t.kind is Kind.COMMENT], [1, 2])

    def test_comment_markers_inside_strings_are_code(self):
        self.assertEqual(comments('const char *s = "a // b /* c */";\n'), [])
        self.assertEqual(comments('f("\\" // still a string");\n'), [])

    def test_raw_strings(self):
        self.assertEqual(comments('auto s = R"(a // b)";\n'), [])
        self.assertEqual(comments('auto s = R"x(a )" // b)x"; // c\n'), ["// c"])
        self.assertEqual(comments('auto s = u8R"(\n// not a comment\n)"; // c\n'), ["// c"])

    def test_a_raw_string_that_looks_like_a_dump_record(self):
        text = 'auto s = R"x(\ncomment \'// x\'\t\tLoc=<src.cxx:1:1>\nhash \'#\'\t [StartOfLine]\tLoc=<src.cxx:2:1>\n)x"; // c\n'
        self.assertEqual(comments(text), ["// c"])
        self.assertEqual(join(lex(text)), text)

    def test_char_literals_and_digit_separators(self):
        self.assertEqual(comments("char c = '\"'; // real\n"), ["// real"])
        self.assertEqual(comments("char a = '/', b = '*';\n"), [])
        self.assertEqual(comments("int a = 1'000; // c\n"), ["// c"])

    def test_a_comment_between_a_name_and_its_parenthesis(self):
        self.assertEqual(
            kinds("f /*x*/ (1);\n"),
            [(Kind.CODE, "f"), (Kind.COMMENT, "/*x*/"), (Kind.CODE, "("), (Kind.CODE, "1"), (Kind.CODE, ")"),
             (Kind.CODE, ";")],
        )

    def test_continued_line_comment(self):
        self.assertEqual(comments("// a \\\nstill the comment\nint x;\n"), ["// a \\\nstill the comment"])

    def test_block_comment_across_lines_and_to_end_of_file(self):
        self.assertEqual(comments("/* a\n b */ int x;\n"), ["/* a\n b */"])
        self.assertEqual(comments("int a; /* b\nc\n"), ["/* b\nc\n"])

    def test_crlf_is_not_part_of_the_comment(self):
        self.assertEqual(comments("int a; // c\r\nint b;\r\n"), ["// c"])


@unittest.skipUnless(HAVE_CLANG, "needs clang++ on PATH")
class Directives(unittest.TestCase):
    def test_a_directive_line_is_directive_tokens(self):
        self.assertEqual(
            kinds("#define X 1 // c\nint a;\n"),
            [(Kind.DIRECTIVE, "#"), (Kind.DIRECTIVE, "define"), (Kind.DIRECTIVE, "X"), (Kind.DIRECTIVE, "1"),
             (Kind.COMMENT, "// c"), (Kind.CODE, "int"), (Kind.CODE, "a"), (Kind.CODE, ";")],
        )

    def test_a_directive_continues_over_splices_and_block_comments(self):
        self.assertEqual(
            [k for k, _ in kinds("#define A 1 /* x\n y */ \\\n 2\nb\n")],
            [Kind.DIRECTIVE] * 4 + [Kind.COMMENT, Kind.DIRECTIVE, Kind.CODE],
        )

    def test_a_hash_after_a_leading_comment_opens_a_directive(self):
        self.assertEqual(kinds("/* e */ #define X 1\n")[1], (Kind.DIRECTIVE, "#"))
        self.assertEqual(kinds("/*\n*/ #define Y 2\n")[1], (Kind.DIRECTIVE, "#"))

    def test_a_hash_after_code_is_code(self):
        self.assertEqual(kinds("int b; /*\n*/ #define Z 3\n")[4], (Kind.CODE, "#"))


@unittest.skipUnless(HAVE_CLANG, "needs clang++ on PATH")
class HeaderNames(unittest.TestCase):
    def test_a_header_name_is_one_token(self):
        self.assertEqual(kinds("#include <a//b>\n"), [(Kind.DIRECTIVE, "#"), (Kind.DIRECTIVE, "include"),
                                                       (Kind.DIRECTIVE, "<a//b>")])
        self.assertEqual(kinds("#include <amrex/AMReX.H>\n")[2], (Kind.DIRECTIVE, "<amrex/AMReX.H>"))
        self.assertEqual(kinds("#  include_next <c/*d>\n")[2], (Kind.DIRECTIVE, "<c/*d>"))

    def test_header_names_hold_no_comments(self):
        self.assertEqual(comments("#include <a//b.h>\n#  include_next <c/*d>\nint a; // e\n"), ["// e"])
        self.assertEqual(comments("#include <a/*b.h> // c\nint x; /* d */\n"), ["// c", "/* d */"])
        self.assertEqual(comments("#include <vector> // std\n"), ["// std"])
        self.assertEqual(comments("#include /* c */ <a//b.h>\n"), ["/* c */"])

    def test_has_include_in_a_conditional(self):
        self.assertEqual(comments("#if __has_include(<a//b.h>)\n#endif\n"), [])
        self.assertEqual(comments("#if A && \\\n  __has_include(<a//b.h>)\n#elif __has_include(<c//d>)\n#endif\n"), [])

    def test_has_include_outside_a_conditional_is_not_a_header_name(self):
        self.assertEqual(comments("#define H __has_include(<a/*note*/b>)\n"), ["/*note*/"])
        self.assertEqual(comments("bool h = __has_include(<a/*c*/b>);\n"), ["/*c*/"])

    def test_a_less_than_is_not_a_header_name(self):
        self.assertEqual(comments("bool b = x < y; // c\n"), ["// c"])
        self.assertEqual(comments("#if A < B // c\n#endif\n"), ["// c"])

    def test_a_header_name_hidden_by_another_is_found(self):
        text = "#include <a/*b>\n#include <c//d> // e\nint x; /* f */\n"
        self.assertEqual(comments(text), ["// e", "/* f */"])
        self.assertEqual(join(lex(text)), text)


class Missing(unittest.TestCase):
    def test_no_clang_raises_tool_missing(self):
        with mock.patch.dict(os.environ, {"PATH": ""}):
            with self.assertRaises(ToolMissing):
                lex("int a; // x\n")


if __name__ == "__main__":
    unittest.main()
