from __future__ import annotations

import shutil
import unittest
from unittest import mock

import lex_clang
from files import Lang
from rules import check, check_text

HAVE_CLANG = shutil.which(lex_clang.CLANG) is not None


def violations(text):
    return [v.region.text for v in check_text(text, Lang.CXX)]


def in_a_function(comment):
    return f"void f() {{\n  {comment}\n  a = 1;\n}}\n"


@unittest.skipUnless(HAVE_CLANG, "needs clang++ on PATH")
class ToolMarkers(unittest.TestCase):
    def test_namespace_closers(self):
        self.assertEqual(violations("namespace CarpetX {\nint a;\n} // namespace CarpetX\n"), [])
        self.assertEqual(violations("namespace {\nint a;\n} // namespace\n"), [])
        self.assertEqual(violations("namespace A::B {\n} // namespace A::B\n"), [])

    def test_namespace_text_elsewhere_is_a_comment(self):
        self.assertEqual(violations("int a; // namespace X\n"), ["// namespace X"])
        self.assertEqual(violations("// namespace X\n"), ["// namespace X"])
        self.assertEqual(violations("} // end namespace X\n"), ["// end namespace X"])
        self.assertEqual(violations("} /* namespace X */\n"), ["/* namespace X */"])
        self.assertEqual(violations("}; // namespace X\n"), ["// namespace X"])
        self.assertEqual(violations("} // namespace X, the driver\n"), ["// namespace X, the driver"])

    def test_line_break_guard(self):
        self.assertEqual(violations("f(a, //\n  b);\n"), [])
        self.assertEqual(violations("f(a, //  \n  b);\n"), [])

    def test_a_lone_empty_comment_line_is_not_reported(self):
        self.assertEqual(violations("int a;\n//\nint b;\n"), [])


@unittest.skipUnless(HAVE_CLANG, "needs clang++ on PATH")
class Separators(unittest.TestCase):
    def test_a_citation_survives_the_prose_after_its_separator(self):
        text = (
            "// See <https://en.wikipedia.org/wiki/Brent%27s_method>\n"
            "//\n"
            "// Returns a bracket containing the root. `iters` is the number of iterations\n"
            "// performed.\n"
            "int brent();\n"
        )
        found = check_text(text, Lang.CXX)
        self.assertEqual(
            [(v.line, v.region.text) for v in found],
            [(3, "// Returns a bracket containing the root. `iters` is the number of iterations\n// performed.")],
        )

    def test_prose_on_both_sides_is_two_runs(self):
        found = check_text("// first\n// still first\n//\n// second\nint a;\n", Lang.CXX)
        self.assertEqual([(v.line, v.region.text) for v in found], [(1, "// first\n// still first"), (4, "// second")])

    def test_a_separator_between_citations_keeps_both(self):
        self.assertEqual(violations("// arXiv:1111.2177 [gr-qc], (71)\n//\n// https://example.org/paper\nint a;\n"), [])

    def test_formatter_markers(self):
        self.assertEqual(violations("// clang-format off\nint a;\n// clang-format on\n"), [])
        self.assertEqual(violations("/* clang-format off */\nint a;\n/* clang-format on */\n"), [])

    def test_markers_carry_no_reason(self):
        self.assertEqual(violations("// clang-format off: table\n"), ["// clang-format off: table"])
        self.assertEqual(violations("//clang-format off\n"), ["//clang-format off"])

    def test_a_marker_stays_when_the_prose_beside_it_goes(self):
        self.assertEqual(
            violations("// prose\n// clang-format off\n// more\nint a;\n"), ["// prose", "// more"]
        )

    def test_nolint_is_not_a_marker(self):
        self.assertEqual(violations("int b; // NOLINT\n"), ["// NOLINT"])
        self.assertEqual(violations("// NOLINTNEXTLINE(x)\nint d;\n"), ["// NOLINTNEXTLINE(x)"])


@unittest.skipUnless(HAVE_CLANG, "needs clang++ on PATH")
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
                self.assertEqual(violations(in_a_function(comment)), [])

    def test_a_trailing_citation_is_allowed(self):
        self.assertEqual(violations("a = f(x); // arXiv:1111.2177 (71)\n"), [])

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
            "/* */",
            "/**/",
        ]:
            with self.subTest(comment):
                self.assertEqual(len(violations(in_a_function(comment))), 1)


@unittest.skipUnless(HAVE_CLANG, "needs clang++ on PATH")
class EverythingElse(unittest.TestCase):
    def test_prose_todo_labels_and_disabled_code(self):
        self.assertEqual(violations("// Must run before the restriction.\nint a;\n"), ["// Must run before the restriction."])
        self.assertEqual(violations("// TODO: Don't include files from other thorns\n"), ["// TODO: Don't include files from other thorns"])
        self.assertEqual(violations("f(cctkGH, /*vi=*/-1, /*tl=*/0);\n"), ["/*vi=*/", "/*tl=*/"])
        self.assertEqual(violations("// int b = 2;\n// f(b);\nint a;\n"), ["// int b = 2;\n// f(b);"])
        self.assertEqual(violations("int z; // tab\n"), ["// tab"])

    def test_non_ascii_prose(self):
        self.assertEqual(violations("double v; // m s⁻¹\n"), ["// m s⁻¹"])

    def test_violation_lines(self):
        found = check_text("int a;\n\n// x\n// y\nint b; /* z */\n", Lang.CXX)
        self.assertEqual([v.line for v in found], [3, 5])

    def test_no_comment_opener_skips_clang(self):
        with mock.patch("rules.lex", side_effect=AssertionError("lexed")):
            self.assertEqual(check_text("int a = b / c * d;\n", Lang.CXX), [])

    def test_check_takes_tokens(self):
        self.assertEqual([v.region.text for v in check(lex_clang.lex("int a; // x\n"), Lang.CXX)], ["// x"])


if __name__ == "__main__":
    unittest.main()
