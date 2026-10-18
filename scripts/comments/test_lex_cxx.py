import unittest

from lex_cxx import guard_groups, lex, scan


def spans(text):
    return [(s.kind, s.text) for s in lex(text)]


class Comments(unittest.TestCase):
    def test_line_and_block_comments(self):
        self.assertEqual(
            spans("int a; // x\n/* y */ int b;\n"),
            [("line", "// x"), ("block", "/* y */")],
        )

    def test_line_numbers(self):
        self.assertEqual([s.line for s in lex("int a;\n\n// x\nint b; /* y */\n")], [3, 4])

    def test_comment_markers_inside_strings_are_code(self):
        self.assertEqual(spans('const char *s = "a // b /* c */";\n'), [])
        self.assertEqual(spans('f("\\" // still a string");\n'), [])

    def test_raw_strings(self):
        self.assertEqual(spans('auto s = R"(a // b)";\n'), [])
        self.assertEqual(spans('auto s = R"x(a )" // b)x"; // c\n'), [("line", "// c")])
        self.assertEqual(spans('auto s = u8R"(\n// not a comment\n)"; // c\n'), [("line", "// c")])

    def test_identifier_ending_in_r_is_not_a_raw_string_prefix(self):
        self.assertEqual(spans('FOR"(" // c\n'), [("line", "// c")])

    def test_char_literals(self):
        self.assertEqual(spans("char c = '\"'; // real\n"), [("line", "// real")])
        self.assertEqual(spans("char a = '/', b = '*';\n"), [])
        self.assertEqual(spans("auto c = u8'/'; // c\n"), [("line", "// c")])

    def test_digit_separators(self):
        self.assertEqual(spans("int a = 1'0; // c\n"), [("line", "// c")])
        self.assertEqual(spans("int a = 0xFF'FF; // c\n"), [("line", "// c")])

    def test_continued_line_comment(self):
        self.assertEqual(
            spans("// a \\\nstill the comment\nint x;\n"),
            [("line", "// a \\\nstill the comment")],
        )

    def test_block_comment_across_lines(self):
        self.assertEqual(spans("/* a\n b */ int x;\n"), [("block", "/* a\n b */")])

    def test_unterminated_block_comment_runs_to_end_of_file(self):
        self.assertEqual(spans("int a; /* b\nc\n"), [("block", "/* b\nc\n")])

    def test_crlf_is_not_part_of_the_comment(self):
        self.assertEqual(spans("int a; // c\r\nint b;\r\n"), [("line", "// c")])


class ConstantGuards(unittest.TestCase):
    def test_if0_region_swallows_its_comments(self):
        self.assertEqual(
            spans("a;\n#if 0\nb; // c\n#endif\nd;\n"),
            [("disabled", "#if 0\nb; // c\n#endif\n")],
        )

    def test_if0_else(self):
        found = lex("#if 0\nb;\n#else\nc;\n#endif\n")
        self.assertEqual([(s.kind, s.text) for s in found],
                         [("disabled", "#if 0\nb;\n#else\n"), ("disabled", "#endif\n")])
        self.assertEqual(len(guard_groups(found)), 1)

    def test_if1(self):
        self.assertEqual(spans("#if 1\nc;\n#endif\n"),
                         [("disabled", "#if 1\n"), ("disabled", "#endif\n")])

    def test_if1_else(self):
        self.assertEqual(spans("#if 1\nc;\n#else\nb;\n#endif\n"),
                         [("disabled", "#if 1\n"), ("disabled", "#else\nb;\n#endif\n")])

    def test_spelled_variants(self):
        for opener in ["#if 0", "#  if (0)", "#if false", "#if 0 && defined __CUDACC__", "  #if 0 // off"]:
            self.assertEqual([s.kind for s in lex(f"{opener}\nx;\n#endif\n")], ["disabled"], opener)

    def test_real_conditions_are_code(self):
        text = "#ifdef A\n#if FOO\n#if 0x1\n#if 0 || B\nx;\n#endif\n#endif\n#endif\n#elif 0\n#endif\n"
        self.assertEqual(spans(text), [])
        self.assertEqual(scan(text).refusals, [])

    def test_nested_guards_in_a_dead_region_are_one_span(self):
        self.assertEqual(
            spans("#if 0\n#if 1\nx;\n#else\ny;\n#endif\n#endif\n"),
            [("disabled", "#if 0\n#if 1\nx;\n#else\ny;\n#endif\n#endif\n")],
        )

    def test_guards_in_a_live_branch_are_their_own(self):
        found = lex("#if 1\n#if 0\nx;\n#endif\ny;\n#endif\n")
        self.assertEqual([s.text for s in found], ["#if 1\n", "#if 0\nx;\n#endif\n", "#endif\n"])
        self.assertEqual([len(g) for g in guard_groups(found)], [2, 1])

    def test_guards_inside_real_conditionals(self):
        self.assertEqual(spans("#ifdef A\n#if 0\nx;\n#endif\n#endif\n"), [("disabled", "#if 0\nx;\n#endif\n")])
        self.assertEqual(spans("#ifdef A\n#else\n#if 0\nx;\n#endif\n#endif\n"), [("disabled", "#if 0\nx;\n#endif\n")])

    def test_directives_inside_comments_and_raw_strings_are_ignored(self):
        self.assertEqual(spans("/*\n#if 0\n*/\nint x;\n"), [("block", "/*\n#if 0\n*/")])
        self.assertEqual(spans('auto s = R"(\n#if 0\n)";\n'), [])

    def test_elif_under_a_constant_guard_is_refused(self):
        text = "a;\n#if 0\nx;\n#elif defined B\ny;\n#else\nz;\n#endif\n"
        result = scan(text)
        self.assertEqual([(s.kind, s.line) for s in result.spans], [("disabled", 2)])
        self.assertEqual([line for line, _ in result.refusals], [2])

    def test_unterminated_guard_is_left_alone(self):
        self.assertEqual(spans("#if 0\nx;\n"), [])


class TodoWarnings(unittest.TestCase):
    def test_a_todo_warning_is_a_span(self):
        found = lex('void f();\n\n#warning "TODO: Restrict"\n\nvoid g();\n')
        self.assertEqual([(s.kind, s.line, s.text) for s in found], [("warning", 3, '#warning "TODO: Restrict"')])

    def test_a_continued_todo_warning_is_one_span(self):
        text = '  {\n#warning                \\\n    "TODO: Allow different operators"\n    int rank = 0;\n'
        self.assertEqual(spans(text), [("warning", '#warning                \\\n    "TODO: Allow different operators"')])

    def test_spelled_variants(self):
        for line in ['# warning "TODO x"', "  #warning TODO bare words", '#warning "FIXME: x"', '#warning " TODO"']:
            self.assertEqual([s.kind for s in lex(f"{line}\nint a;\n")], ["warning"], line)

    def test_other_diagnostics_are_code(self):
        text = '#warning "SIMD is disabled"\n#error "TODO is not a #warning"\n#warning TODOS\nint TODO;\n'
        self.assertEqual(spans(text), [])

    def test_a_comment_on_the_directive_belongs_to_it(self):
        self.assertEqual(spans('#warning "TODO: x" // why\nint a;\n'), [("warning", '#warning "TODO: x" // why')])

    def test_a_block_comment_running_past_the_directive_is_refused(self):
        result = scan('#warning "TODO: x" /* a\n b */\nint a;\n')
        self.assertEqual([line for line, _ in result.refusals], [1])

    def test_inside_a_dead_region_it_goes_with_the_region(self):
        self.assertEqual(spans('#if 0\n#warning "TODO: x"\n#endif\n'), [("disabled", '#if 0\n#warning "TODO: x"\n#endif\n')])

    def test_inside_a_live_branch_it_is_its_own_span(self):
        found = lex('#if 1\n{\n#warning "TODO: x"\n  a;\n}\n#endif\n')
        self.assertEqual([s.kind for s in found], ["disabled", "warning", "disabled"])

    def test_inside_comments_and_raw_strings_it_is_ignored(self):
        self.assertEqual(spans('/*\n#warning "TODO: x"\n*/\n'), [("block", '/*\n#warning "TODO: x"\n*/')])
        self.assertEqual(spans('auto s = R"(\n#warning "TODO: x"\n)";\n'), [])


if __name__ == "__main__":
    unittest.main()
