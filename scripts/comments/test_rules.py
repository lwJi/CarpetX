import unittest

from files import Lang
from rules import check


def violations(text, lang=Lang.CXX):
    return [v.group.text for v in check(text, lang)]


class Exceptions(unittest.TestCase):
    def test_namespace_closers(self):
        self.assertEqual(violations("namespace X {\nint a;\n} // namespace X\n"), [])
        self.assertEqual(violations("namespace {\nint a;\n} // namespace\n"), [])
        self.assertEqual(violations("namespace A::B {\n} // namespace A::B\n"), [])

    def test_namespace_closer_with_the_wrong_name_still_passes(self):
        self.assertEqual(violations("namespace DG {\n} // namespace\n"), [])

    def test_namespace_text_elsewhere_is_a_comment(self):
        self.assertEqual(violations("int a; // namespace X\n"), ["// namespace X"])
        self.assertEqual(violations("// namespace X\n"), ["// namespace X"])
        self.assertEqual(violations("} // end namespace X\n"), ["// end namespace X"])
        self.assertEqual(violations("} /* namespace X */\n"), ["/* namespace X */"])

    def test_line_break_guard(self):
        self.assertEqual(violations("f(a, //\n  b);\n"), [])
        self.assertEqual(violations("int a;\n//\nint b;\n"), ["//"])

    def test_formatter_and_lint_markers(self):
        text = (
            "// clang-format off\nint a;\n// clang-format on\n"
            "int b; // NOLINT\nint c; // NOLINT(bugprone-foo,-cert-*)\n"
            "// NOLINTNEXTLINE(x)\nint d;\n/* clang-format off */\n"
        )
        self.assertEqual(violations(text), [])

    def test_markers_carry_no_reason(self):
        self.assertEqual(violations("int a; // NOLINT because\n"), ["// NOLINT because"])
        self.assertEqual(violations("// clang-format off: table\n"), ["// clang-format off: table"])

    def test_a_marker_stays_when_the_prose_beside_it_goes(self):
        self.assertEqual(violations("// prose\n// clang-format off\nint a;\n"), ["// prose"])


class Citations(unittest.TestCase):
    def test_a_citation_group_is_kept(self):
        for comment in [
            "// arXiv:1111.2177 [gr-qc], (71)",
            "// See arXiv:1111.2177 [gr-qc], (74-75)",
            "// arXiv:1111.2177v2, eqs. 74–75.",
            "// https://stackoverflow.com/a/30968371",
            "// See <https://en.wikipedia.org/wiki/Brent%27s_method>",
            "// Taken from <https://en.cppreference.com/w/cpp/experimental/make_array>",
            "// cf. doi:10.1016/0021-9991(88)90177-5",
            "/* arXiv:1111.2177 (71) */",
            "/*\n * arXiv:1111.2177\n * https://example.org/paper\n */",
            "// arXiv:1111.2177 [gr-qc], (71)\n//\n// https://example.org/paper",
        ]:
            with self.subTest(comment):
                self.assertEqual(violations(f"{{\n  {comment}\n  a = 1;\n}}\n"), [])

    def test_trailing_and_hash_citations_are_kept(self):
        self.assertEqual(violations("a = f(x); // arXiv:1111.2177 (71)\n"), [])
        text = 'KEYWORD m "m"\n{\n  # See <https://en.wikipedia.org/wiki/List_of_Runge–Kutta_methods>\n  "a" :: "a"\n} "a"\n'
        self.assertEqual(violations(text, Lang.CCL), [])

    def test_a_citation_inside_prose_goes_with_its_group(self):
        cases = [
            "// This class is based on NSIMD\n// <https://github.com/agenium-scale/nsimd>. The NSIMD library is very\n// close to what we need.",
            "// See <https://en.wikipedia.org/wiki/Brent%27s_method>\n//\n// Returns a bracket containing the root.",
            "// see the paper arXiv:1111.2177",
            "// arXiv:1111.2177 [gr-qc], (71) for the upwind stencil",
            "// Shu & Osher 1988",
            "// series->setSoftware(\"Einstein Toolkit <https://einsteintoolkit.org>\");",
        ]
        for comment in cases:
            with self.subTest(comment):
                self.assertEqual(violations(f"{comment}\nint a;\n"), [comment])
        prose = "# -ksp_type {bcgs|gmres}   <https://petsc.org/KSPType.html>"
        self.assertEqual(violations(f"{prose}\nBOOLEAN b \"b\"\n{{\n}} no\n", Lang.CCL), [prose])

    def test_an_empty_comment_is_not_a_citation(self):
        self.assertEqual(violations("int a;\n/* */\nint b;\n"), ["/* */"])


class Groups(unittest.TestCase):
    def test_adjacent_full_line_comments_form_one_group(self):
        self.assertEqual(violations("// a\n// b\nint x;\n\n// c\n"), ["// a\n// b", "// c"])

    def test_block_and_line_comments_join(self):
        self.assertEqual(violations("/* a\n b */\n// c\nint x;\n"), ["/* a\n b */\n// c"])

    def test_trailing_and_inline_comments_stand_alone(self):
        self.assertEqual(
            violations("// a\nint x; // b\n// c\nf(/*d*/ 1);\n"),
            ["// a", "// b", "// c", "/*d*/"],
        )

    def test_a_constant_guard_is_one_violation(self):
        found = check("#if 0\nx;\n#else\ny;\n#endif\n", Lang.CXX)
        self.assertEqual(len(found), 1)
        self.assertEqual(len(found[0].group.spans), 2)

    def test_hash_comments(self):
        self.assertEqual(violations("# a\n# b\n\nX y # c\n", Lang.CCL), ["# a\n# b", "# c"])
        self.assertEqual(violations("A::b = 1 # c\n", Lang.PAR), ["# c"])

    def test_cxx_exceptions_do_not_apply_to_hash_comments(self):
        self.assertEqual(violations("} # namespace X\n", Lang.CCL), ["# namespace X"])


if __name__ == "__main__":
    unittest.main()
