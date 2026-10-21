from __future__ import annotations

import unittest
from unittest import mock

from files import Lang
from rules import comment_words, is_constant, units


def reported(text, lang=Lang.CXX):
    return [u.text for u in units(text, lang) if not u.allowed]


def placed(text, lang=Lang.CXX):
    return [(u.line, u.text) for u in units(text, lang) if not u.allowed]


def directives(text):
    return [(u.line, u.text) for u in units(text, Lang.CXX) if u.key[0] == "directive"]


def in_a_function(comment):
    return f"void f() {{\n  {comment}\n  a = 1;\n}}\n"


class CommentWords(unittest.TestCase):
    def test_openers_and_spacing_are_dropped(self):
        self.assertEqual(comment_words("//   a   b  "), ["a b"])
        self.assertEqual(comment_words("/// a"), ["a"])
        self.assertEqual(comment_words("# a"), ["a"])
        self.assertEqual(comment_words("!! a"), ["a"])
        self.assertEqual(comment_words("/* a */"), ["a"])

    def test_a_block_comment_has_one_entry_per_line(self):
        self.assertEqual(comment_words("/* a\n * b\n   c */"), ["a", "b", "c"])
        self.assertEqual(comment_words("/**\n * a\n */"), ["", "a", ""])
        self.assertEqual(comment_words("/***** a *****/"), ["a"])

    def test_empty_comments_have_no_words(self):
        for text in ["//", "//  ", "/**/", "/* */", "#", "!", "////////"]:
            with self.subTest(text=text):
                self.assertEqual(comment_words(text), [""])


class ToolMarkers(unittest.TestCase):
    def test_namespace_closers(self):
        self.assertEqual(reported("namespace CarpetX {\nint a;\n} // namespace CarpetX\n"), [])
        self.assertEqual(reported("namespace {\nint a;\n} // namespace\n"), [])
        self.assertEqual(reported("namespace A::B {\n} // namespace A::B\n"), [])

    def test_namespace_text_elsewhere_is_a_comment(self):
        self.assertEqual(reported("int a; // namespace X\n"), ["// namespace X"])
        self.assertEqual(reported("// namespace X\n"), ["// namespace X"])
        self.assertEqual(reported("} // end namespace X\n"), ["// end namespace X"])
        self.assertEqual(reported("} /* namespace X */\n"), ["/* namespace X */"])
        self.assertEqual(reported("}; // namespace X\n"), ["// namespace X"])
        self.assertEqual(reported("} // namespace X, the driver\n"), ["// namespace X, the driver"])

    def test_formatter_markers(self):
        self.assertEqual(reported("// clang-format off\nint a;\n// clang-format on\n"), [])
        self.assertEqual(reported("/* clang-format off */\nint a;\n/* clang-format on */\n"), [])

    def test_markers_carry_no_reason(self):
        self.assertEqual(reported("// clang-format off: table\n"), ["// clang-format off: table"])
        self.assertEqual(reported("//clang-format off\n"), ["//clang-format off"])

    def test_a_marker_stays_when_the_prose_beside_it_goes(self):
        self.assertEqual(reported("// prose\n// clang-format off\n// more\nint a;\n"), ["// prose", "// more"])

    def test_nolint_is_not_a_marker(self):
        self.assertEqual(reported("int b; // NOLINT\n"), ["// NOLINT"])
        self.assertEqual(reported("// NOLINTNEXTLINE(x)\nint d;\n"), ["// NOLINTNEXTLINE(x)"])


class EmptyComments(unittest.TestCase):
    def test_a_comment_with_no_words_is_not_reported(self):
        for text in ["f(a, //\n  b);\n", "f(a, //  \n  b);\n", "int a;\n//\nint b;\n", "int a; /* */\n", "int a; /**/\n"]:
            with self.subTest(text=text):
                self.assertEqual(units(text, Lang.CXX), [])

    def test_hash_and_bang_alone_are_not_reported(self):
        self.assertEqual(units("A::b = 1\n#\nA::c = 2\n", Lang.PAR), [])
        self.assertEqual(units("x = 1\n!\n", Lang.FORTRAN), [])


class EachLine(unittest.TestCase):
    def test_every_line_of_a_comment_is_judged_alone(self):
        text = (
            "// See <https://en.wikipedia.org/wiki/Brent%27s_method>\n"
            "// Returns a bracket containing the root. `iters` is the number of iterations\n"
            "// performed.\n"
            "int brent();\n"
        )
        self.assertEqual(
            placed(text),
            [(2, "// Returns a bracket containing the root. `iters` is the number of iterations"), (3, "// performed.")],
        )

    def test_the_lines_of_a_block_comment(self):
        self.assertEqual(placed("int a;\n/* first\n * second\n\n   third */\nint b;\n"),
                         [(2, "/* first"), (3, "* second"), (5, "third */")])

    def test_line_numbers(self):
        self.assertEqual([u.line for u in units("int a;\n\n// x\n// y\nint b; /* z */\n", Lang.CXX)], [3, 4, 5])

    def test_the_key_is_the_words(self):
        [a] = units("int a; //   one   two\n", Lang.CXX)
        [b] = units("/* one two */\nint b;\n", Lang.CXX)
        self.assertEqual(a.key, ("comment", "one two"))
        self.assertEqual(a.key, b.key)


class Citations(unittest.TestCase):
    def test_a_citation_is_allowed(self):
        for comment in [
            "// See arXiv:1111.2177 [gr-qc], (74-75)",
            "// arXiv:1111.2177 [gr-qc], (71)",
            "// arXiv:1111.2177v2, eqs. 74–75.",
            "// arXiv:gr-qc/0610128",
            "// https://stackoverflow.com/a/30968371",
            "// See <https://en.wikipedia.org/wiki/Brent%27s_method>",
            "// Taken from <https://en.cppreference.com/w/cpp/experimental/make_array>",
            "// From <https://en.wikipedia.org/wiki/List_of_Runge–Kutta_methods>",
            "// doi:10.1016/0021-9991(88)90177-5",
            "// cf. doi:10.1016/0021-9991(88)90177-5",
            "/* arXiv:1111.2177 (71) */",
            "/*\n * arXiv:1111.2177\n * https://example.org/paper\n */",
            "// arXiv:1111.2177 [gr-qc], (71)\n  //\n  // https://example.org/paper",
        ]:
            with self.subTest(comment):
                self.assertEqual(reported(in_a_function(comment)), [])

    def test_a_trailing_citation_is_allowed(self):
        self.assertEqual(reported("a = f(x); // arXiv:1111.2177 (71)\n"), [])

    def test_a_citation_needs_no_separator_from_prose(self):
        self.assertEqual(placed("// doi:10.1016/0021-9991(88)90177-5\n// Low-storage Shu-Osher form\nint a;\n"),
                         [(2, "// Low-storage Shu-Osher form")])

    def test_prose_is_not_a_citation(self):
        for comment in [
            "// Shu & Osher 1988",
            "// <https://en.wikipedia.org/wiki/Coroutine>. That is, each function",
            "// See <https://en.wikipedia.org/wiki/Brent%27s_method>\n  // Returns a bracket containing the root.",
            "// see the paper arXiv:1111.2177",
            "// arXiv:1111.2177 [gr-qc], (71) for the upwind stencil",
            "// Inspired by Athena-K; see\n  // <https://github.com/IAS-Astrophysics/athenak>",
            '// series->setSoftware("Einstein Toolkit <https://einsteintoolkit.org>");',
            "// AMReX_FluxRegister.cpp:188",
        ]:
            with self.subTest(comment):
                self.assertEqual(len(reported(in_a_function(comment))), 1)


class EverythingElse(unittest.TestCase):
    def test_prose_todo_labels_and_disabled_code(self):
        self.assertEqual(reported("// Must run before the restriction.\nint a;\n"), ["// Must run before the restriction."])
        self.assertEqual(reported("// TODO: Don't include files from other thorns\n"), ["// TODO: Don't include files from other thorns"])
        self.assertEqual(reported("f(cctkGH, /*vi=*/-1, /*tl=*/0);\n"), ["/*vi=*/", "/*tl=*/"])
        self.assertEqual(reported("// int b = 2;\n// f(b);\nint a;\n"), ["// int b = 2;", "// f(b);"])
        self.assertEqual(reported("int z; // tab\n"), ["// tab"])

    def test_non_ascii_prose(self):
        self.assertEqual(reported("double v; // m s⁻¹\n"), ["// m s⁻¹"])

    def test_no_comment_opener_skips_the_lexer(self):
        with mock.patch("rules.lex", side_effect=AssertionError("lexed")):
            self.assertEqual(units("int a = b / c * d;\n", Lang.CXX), [])


class ConstantConditions(unittest.TestCase):
    def test_numbers_and_defined_tests_only(self):
        for condition in [
            "0", "1", "(0)", "!1", "-1", "~0", "+1", "0x1F", "010", "0L", "1u", "1'000", "1 - 1", "2 > 1", "1 << 2",
            "0 ? 1 : 0", "0 && defined X", "defined X && 0", "defined(X) || 1", "1 || defined X", "defined ( X ) && 0",
            "0 || defined X", "1 && defined X",
        ]:
            with self.subTest(condition=condition):
                self.assertTrue(is_constant(condition))

    def test_a_macro_name_or_no_number(self):
        for condition in [
            "", "X", "!X", "F(1, 2)", "defined X", "!defined X", "defined(X) || defined(Y)", "X == 1", "true",
            "__cplusplus >= 201703L", "0 && X", "0 && F(1, 2)", "0 && __has_include(<x.h>)", "0 && defined",
            "0 && defined(X",
        ]:
            with self.subTest(condition=condition):
                self.assertFalse(is_constant(condition))


class Guards(unittest.TestCase):
    def test_constant_guards_and_messages_are_reported(self):
        found = units('#if 0\nint a;\n#endif\n#warning "TODO: Restrict"\n#pragma message("x")\n', Lang.CXX)
        self.assertEqual([(u.line, u.text, u.key[0], u.allowed) for u in found],
                         [(1, "#if 0", "directive", False), (4, '#warning "TODO: Restrict"', "directive", False),
                          (5, '#pragma message("x")', "directive", False)])

    def test_constant_if_and_elif(self):
        text = "#if 0\na\n#endif\n#if X\nb\n#elif 1\nc\n#endif\n#if 0 && defined __CUDACC__\nd\n#elif 0 && defined Y\ne\n#else\n#if 0 && defined Z\nf\n#endif\n#endif\n"
        self.assertEqual(directives(text), [(1, "#if 0"), (6, "#elif 1"), (9, "#if 0 && defined __CUDACC__"),
                                            (11, "#elif 0 && defined Y"), (14, "#if 0 && defined Z")])

    def test_conditions_that_depend_on_a_macro_are_not_guards(self):
        self.assertEqual(directives("#if X\n#elif defined Y\n#endif\n#ifdef Z\n#endif\n#ifndef W\n#endif\n#if 0\n#endif\n"),
                         [(8, "#if 0")])

    def test_every_constant_guard_is_reported_even_inside_disabled_code(self):
        self.assertEqual(directives("#if 0\n#if 0\na\n#endif\n#endif\n"), [(1, "#if 0"), (2, "#if 0")])
        self.assertEqual(directives("#if 1\n#elif 0\n#endif\n"), [(1, "#if 1"), (2, "#elif 0")])

    def test_comments_layout_splices_and_digraphs(self):
        self.assertEqual(directives("  #  if /* c */ 0 // d\n#endif\n"), [(1, "# if 0")])
        self.assertEqual(directives("/* c */ #if 0\n#endif\n"), [(1, "#if 0")])
        self.assertEqual(directives("#if 1 /* multi\nline */\nA\n#endif\n"), [(1, "#if 1")])
        self.assertEqual(directives("#if \\\n 0\n#endif\n"), [(1, "#if 0")])
        self.assertEqual(directives("%:if 0\n%:endif\n"), [(1, "%:if 0")])

    def test_a_hash_after_code_is_not_a_directive(self):
        self.assertEqual(directives("int b; /*\n*/ #if 0\n#endif\n"), [])

    def test_comments_in_and_on_a_guard_are_reported_too(self):
        self.assertEqual(placed("#if 0 // old\n// int a;\n#endif\n"), [(1, "// old"), (1, "#if 0"), (2, "// int a;")])

    def test_a_citation_does_not_excuse_a_guard(self):
        self.assertEqual(placed("#if 0 // arXiv:1111.2177\n#endif\n"), [(1, "#if 0")])

    def test_the_key_ignores_layout(self):
        [a] = units("#if 0\n#endif\n", Lang.CXX)
        [b] = units("  #  if   0\n#endif\n", Lang.CXX)
        self.assertEqual(a.key, ("directive", "# if 0"))
        self.assertEqual(a.key, b.key)


class Messages(unittest.TestCase):
    def test_compile_time_messages(self):
        text = (
            '#warning "TODO: Restrict"\n'
            "#warning                \\\n"
            '    "TODO: Allow different restriction operators"\n'
            '#pragma message("why")\n'
            '#pragma GCC warning "why"\n'
            '_Pragma("message(\\"why\\")") int a;\n'
            '#define W _Pragma("GCC warning \\"why\\"")\n'
            'int b = 1; _Pragma(R"(message("why"))")\n'
        )
        found = directives(text)
        self.assertEqual([line for line, _ in found], [1, 2, 4, 5, 6, 7, 8])
        self.assertEqual(found[1][1], '#warning "TODO: Allow different restriction operators"')

    def test_what_is_not_a_message(self):
        text = (
            '#error "Particles requires MPI support for I/O"\n'
            "#pragma omp parallel for\n"
            '#pragma GCC error "stop"\n'
            '_Pragma("omp parallel") int a;\n'
            '#pragma clang diagnostic push\n'
            'auto s = "#warning not a directive";\n'
        )
        self.assertEqual(directives(text), [])

    def test_a_message_inside_disabled_code_is_reported_too(self):
        text = '#if 0\n#warning "a"\n_Pragma("message(\\"b\\")")\n#else\n#warning "c"\n#endif\n'
        self.assertEqual([line for line, _ in directives(text)], [1, 2, 3, 5])

    def test_a_file_with_only_a_guard_or_message_is_lexed(self):
        self.assertEqual(len(units("#if 0\nint a;\n#endif\n", Lang.CXX)), 1)
        self.assertEqual(len(units('_Pragma("message(\\"x\\")")\n', Lang.CXX)), 1)

    def test_error_and_other_pragmas_stay(self):
        self.assertEqual(reported('#error "Particles requires MPI support for I/O"\n#pragma omp parallel\n'), [])


class Comments(unittest.TestCase):
    def ids(self, text, lang=Lang.CXX):
        return [(u.line, u.comment) for u in units(text, lang)]

    def test_adjacent_comment_lines_are_one_comment(self):
        self.assertEqual(self.ids("// a\n// b\n//\n// c\nint x;\n"), [(1, 0), (2, 0), (4, 0)])

    def test_a_blank_line_or_code_starts_a_new_one(self):
        self.assertEqual(self.ids("// a\n\n// b\nint x;\n// c\n"), [(1, 0), (3, 1), (5, 2)])

    def test_a_trailing_comment_starts_one_that_the_lines_below_continue(self):
        self.assertEqual(self.ids("int a; // a\nint b; // b\n       // c\n"), [(1, 0), (2, 1), (3, 1)])

    def test_a_block_comment_is_one_comment(self):
        self.assertEqual(self.ids("/* a\n   b */\n// c\nint x; /* d\n e */\n"), [(1, 0), (2, 0), (3, 0), (4, 1), (5, 1)])

    def test_hash_languages(self):
        self.assertEqual(self.ids("# a\n# b\nSRCS = x.cxx # c\n", Lang.MAKE), [(1, 0), (2, 0), (3, 1)])
        self.assertEqual(self.ids("! a\n  ! b\nx = 1 ! c\n", Lang.FORTRAN), [(1, 0), (2, 0), (3, 1)])

    def test_guards_belong_to_no_comment(self):
        self.assertEqual([u.comment for u in units("#if 0\n#endif\n", Lang.CXX)], [-1])


class HashLanguages(unittest.TestCase):
    def test_every_comment_line_is_reported(self):
        self.assertEqual(reported("# Parameter definitions\nINT a \"a\"\n{\n} 1 # why\n", Lang.PARAM),
                         ["# Parameter definitions", "# why"])
        self.assertEqual(reported("# a\n# b\nSRCS = x.cxx\n", Lang.MAKE), ["# a", "# b"])
        self.assertEqual(reported("  ! a\n  x = 1 ! b\n", Lang.FORTRAN), ["! a", "! b"])
        self.assertEqual(reported("/* a */ SCHEDULE X AT Y\n", Lang.SCHEDULE), ["/* a */"])

    def test_a_continued_make_comment_is_two_lines(self):
        self.assertEqual(placed("# a \\\n  b\nSRCS = x.cxx\n", Lang.MAKE), [(1, "# a \\"), (2, "b")])

    def test_commented_out_schedule_blocks(self):
        text = "# SCHEDULE A AT b\n# {\n#   LANG: C\n# } \"a\"\n# \n# SCHEDULE B AT c\n# {\n# } \"b\"\n\nSCHEDULE C AT d\n"
        self.assertEqual([u.line for u in units(text, Lang.SCHEDULE)], [1, 2, 3, 4, 6, 7, 8])

    def test_a_citation_line_next_to_prose(self):
        self.assertEqual(reported("# See <https://example.org/x>\n#\n# prose\nA::b = 1\n", Lang.PAR), ["# prose"])

    def test_citations_are_allowed(self):
        for lang, comment in [
            (Lang.PARAM, "  # See <https://en.wikipedia.org/wiki/List_of_Runge–Kutta_methods>"),
            (Lang.PAR, "# arXiv:1111.2177 [gr-qc], (74-75)"),
            (Lang.MAKE, "# doi:10.1016/0021-9991(88)90177-5"),
            (Lang.FORTRAN, "  ! https://example.org/paper"),
            (Lang.SCHEDULE, "/* arXiv:1111.2177 */"),
        ]:
            with self.subTest(lang=lang):
                self.assertEqual(reported(comment + "\nx\n", lang), [])

    def test_prose_next_to_a_url_is_not_a_citation(self):
        text = "# -ksp_type {bcgs|bcgsl|gmres|...}   <https://www.mcs.anl.gov/petsc/KSPType.html>\n"
        self.assertEqual(len(reported(text, Lang.PARAM)), 1)

    def test_tool_markers_are_c_and_c_plus_plus_only(self):
        self.assertEqual(reported("} # namespace X\n", Lang.PARAM), ["# namespace X"])
        self.assertEqual(reported("# clang-format off\n", Lang.PARAM), ["# clang-format off"])

    def test_par_string_comments(self):
        text = 'ActiveThorns = "\n    # SystemTopology\n    ADMBase  # why\n"\n'
        self.assertEqual(placed(text, Lang.PAR), [(2, "# SystemTopology"), (3, "# why")])

    def test_no_opener_skips_the_lexer(self):
        with mock.patch("rules.lex", side_effect=AssertionError("lexed")):
            self.assertEqual(units("SCHEDULE X AT Y\n{\n} \"x\"\n", Lang.SCHEDULE), [])
            self.assertEqual(units("x = 1 # not Fortran\n", Lang.FORTRAN), [])


if __name__ == "__main__":
    unittest.main()
