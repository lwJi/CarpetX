from __future__ import annotations

import shutil
import subprocess
import unittest

import lex_clang
from lex_clang import lex
from regions import comment_regions, constant_value, directive_regions
from tokens import Placement

HAVE_CLANG = shutil.which(lex_clang.CLANG) is not None
HAVE_UNIFDEF = shutil.which("unifdef") is not None
CONSTANT = [
    ("0", 0), ("1", 1), ("(0)", 0), ("!1", 0), ("!0", 1), ("-1", -1), ("~0", -1), ("0x1F", 31), ("010", 8),
    ("0 && defined X", 0), ("defined X && 0", 0), ("defined(X) || 1", 1), ("1 || defined X", 1),
    ("0 && F(1, 2)", 0), ("F((1), (2)) && 0", 0), ("0 && (A < B)", 0), ("0 && __has_include(<x.h>)", 0),
    ("0 && !X", 0), ("0 && X / 1", 0), ("0 && X % 1", 0), ("defined ( X ) && 0", 0), ("0 && defined", 0),
    ("1 - 1", 0), ("2 > 1", 1), ("1 << 2", 4), ("4 / 2 - 2", 0), ("-7 / 2 == -3", 1), ("-7 % 2 == -1", 1),
    ("0 == 0", 1), ("2 % 2", 0),
]
UNKNOWN = [
    "X", "!X", "F(1, 2)", "defined X", "!defined X", "0 || defined X", "1 && defined X", "X == X", "0 & X",
    "1 | X", "0 * X", "X / 1", "defined X || !defined X", "true", "0 ? 1 : 0", "0 garbage", "0L", "1u", "08",
    "0b1", "'a'", "1 / 0", "0 && 1 / 0", "X / 0", "1 / X", "0 && X / Y", "(0", "0)", "+1", "1 || F(",
    "0 && defined(X", "0 && X(", "",
]


def directive_texts(text):
    return [(r.placement, r.line, r.code) for r in directive_regions(lex(text))]


def regions(text, standalone=lambda t: False):
    return [(r.placement, r.text) for r in comment_regions(lex(text), standalone)]


@unittest.skipUnless(HAVE_CLANG, "needs clang++ on PATH")
class Placements(unittest.TestCase):
    def test_whole_line_trailing_and_inline(self):
        self.assertEqual(
            regions("// a\nint x; // b\n// c\nf(/*d*/ 1);\n"),
            [(Placement.WHOLE_LINE, "// a"), (Placement.TRAILING, "// b"), (Placement.WHOLE_LINE, "// c"),
             (Placement.INLINE, "/*d*/")],
        )

    def test_a_comment_before_code_on_its_line_is_inline(self):
        self.assertEqual(regions("/*constexpr*/ ARITH_INLINE int f();\n"), [(Placement.INLINE, "/*constexpr*/")])
        self.assertEqual(regions("/* a\n b */ int x;\n"), [(Placement.INLINE, "/* a\n b */")])

    def test_a_directive_counts_as_code(self):
        self.assertEqual(regions("#endif // FOO\n"), [(Placement.TRAILING, "// FOO")])
        self.assertEqual(regions("#include <a//b> // c\n"), [(Placement.TRAILING, "// c")])

    def test_each_trailing_comment_is_its_own_region(self):
        self.assertEqual(
            regions("int a; /* b */ // c\n"), [(Placement.TRAILING, "/* b */"), (Placement.TRAILING, "// c")]
        )


@unittest.skipUnless(HAVE_CLANG, "needs clang++ on PATH")
class Runs(unittest.TestCase):
    def test_adjacent_whole_line_comments_form_one_run(self):
        self.assertEqual(
            regions("// a\n// b\nint x;\n\n// c\n"), [(Placement.WHOLE_LINE, "// a\n// b"), (Placement.WHOLE_LINE, "// c")]
        )

    def test_a_blank_line_ends_a_run(self):
        self.assertEqual(
            regions("// a\n\n// b\n"), [(Placement.WHOLE_LINE, "// a"), (Placement.WHOLE_LINE, "// b")]
        )

    def test_an_empty_comment_line_ends_a_run_and_is_no_region(self):
        self.assertEqual(regions("// a\n//\n// b\n"), [(Placement.WHOLE_LINE, "// a"), (Placement.WHOLE_LINE, "// b")])
        self.assertEqual(regions("// a\n  //  \n// b\n"), [(Placement.WHOLE_LINE, "// a"), (Placement.WHOLE_LINE, "// b")])
        self.assertEqual(regions("int a;\n//\nint b;\n"), [])

    def test_an_empty_trailing_comment_is_still_a_region(self):
        self.assertEqual(regions("f(a, //\n  b);\n"), [(Placement.TRAILING, "//")])

    def test_block_and_line_comments_join(self):
        self.assertEqual(regions("/* a\n b */\n// c\nint x;\n"), [(Placement.WHOLE_LINE, "/* a\n b */\n// c")])

    def test_indentation_does_not_end_a_run(self):
        self.assertEqual(regions("  // a\n    // b\nint x;\n"), [(Placement.WHOLE_LINE, "// a\n    // b")])

    def test_comments_sharing_a_line_join(self):
        self.assertEqual(regions("/* a */ /* b */\n// c\n"), [(Placement.WHOLE_LINE, "/* a */ /* b */\n// c")])

    def test_code_between_comments_ends_a_run(self):
        self.assertEqual(
            regions("// a\nint x; // b\n// c\n"),
            [(Placement.WHOLE_LINE, "// a"), (Placement.TRAILING, "// b"), (Placement.WHOLE_LINE, "// c")],
        )

    def test_a_standalone_comment_splits_a_run(self):
        marker = lambda t: t.text == "// marker"
        self.assertEqual(
            regions("// a\n// marker\n// b\n", marker),
            [(Placement.WHOLE_LINE, "// a"), (Placement.WHOLE_LINE, "// marker"), (Placement.WHOLE_LINE, "// b")],
        )

    def test_region_indices_cover_its_tokens(self):
        tokens = lex("int x;\n// a\n// b\nint y; // c\n")
        for r in comment_regions(tokens):
            self.assertEqual(r.tokens, tuple(tokens[r.first : r.last + 1]))
        self.assertEqual([r.line for r in comment_regions(tokens)], [2, 4])


class ConstantValue(unittest.TestCase):
    def test_constant_conditions(self):
        for condition, value in CONSTANT:
            with self.subTest(condition=condition):
                self.assertEqual(constant_value(condition_words(condition)), value)

    def test_conditions_that_depend_on_a_macro_or_do_not_parse(self):
        for condition in UNKNOWN:
            with self.subTest(condition=condition):
                self.assertIsNone(constant_value(condition_words(condition)))


def condition_words(condition):
    return [t.text for t in lex("#if " + condition + "\n") if t.kind.name == "DIRECTIVE"][2:]


@unittest.skipUnless(HAVE_CLANG, "needs clang++ on PATH")
class Guards(unittest.TestCase):
    def test_constant_if_and_elif(self):
        text = "#if 0\na\n#endif\n#if X\nb\n#elif 1\nc\n#endif\n#if 0 && defined __CUDACC__\nd\n#elif 0 && defined Y\ne\n#else\n#if 0 && defined Z\nf\n#endif\n#endif\n"
        self.assertEqual(
            directive_texts(text),
            [(Placement.GUARD, 1, "#if 0"), (Placement.GUARD, 6, "#elif 1"), (Placement.GUARD, 9, "#if 0 && defined __CUDACC__"),
             (Placement.GUARD, 11, "#elif 0 && defined Y"), (Placement.GUARD, 14, "#if 0 && defined Z")],
        )

    def test_conditions_that_depend_on_a_macro_are_not_guards(self):
        self.assertEqual(directive_texts("#if X\n#elif defined Y\n#endif\n#ifdef Z\n#endif\n#ifndef W\n#endif\n"), [])

    def test_a_guard_inside_disabled_code_is_part_of_it(self):
        self.assertEqual(directive_texts("#if 0\n#if 0\na\n#endif\n#endif\n#if 0\n#else\n#if 0\n#endif\n#endif\n"),
                         [(Placement.GUARD, 1, "#if 0"), (Placement.GUARD, 6, "#if 0"), (Placement.GUARD, 8, "#if 0")])
        self.assertEqual(directive_texts("#if 1\n#else\n#if 0\n#endif\n#endif\n"), [(Placement.GUARD, 1, "#if 1")])
        self.assertEqual(directive_texts("#if 1\n#elif 0\n#endif\n"), [(Placement.GUARD, 1, "#if 1")])

    def test_a_guard_inside_live_code_of_a_guard_is_its_own(self):
        self.assertEqual(directive_texts("#if 1\n#if 0\n#endif\n#else\n#endif\n"),
                         [(Placement.GUARD, 1, "#if 1"), (Placement.GUARD, 2, "#if 0")])

    def test_comments_and_layout_in_a_guard(self):
        self.assertEqual(directive_texts("  #  if /* c */ 0 // d\n#endif\n"), [(Placement.GUARD, 1, "# if 0")])
        self.assertEqual(directive_texts("/* c */ #if 0\n#endif\n"), [(Placement.GUARD, 1, "#if 0")])

    def test_a_spliced_or_digraph_condition_is_not_a_guard(self):
        self.assertEqual(directive_texts("#if \\\n 0\n#endif\n#if 0 \\\n && X\n#endif\n%:if 0\n%:endif\n"), [])

    def test_the_guard_covers_its_directive_only(self):
        tokens = lex("int a;\n#if 0 // c\nint b;\n#endif\n")
        [r] = directive_regions(tokens)
        self.assertEqual((r.text, r.line), ("#if 0", 2))
        self.assertEqual(r.tokens, tuple(tokens[r.first : r.last + 1]))


@unittest.skipUnless(HAVE_CLANG, "needs clang++ on PATH")
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
        self.assertEqual(
            [(p, line) for p, line, _ in directive_texts(text)],
            [(Placement.MESSAGE, 1), (Placement.MESSAGE, 2), (Placement.MESSAGE, 4), (Placement.MESSAGE, 5),
             (Placement.MESSAGE, 6), (Placement.MESSAGE, 7), (Placement.MESSAGE, 8)],
        )
        self.assertEqual(directive_texts(text)[1][2], '#warning "TODO: Allow different restriction operators"')
        self.assertEqual(directive_texts(text)[4][2], '_Pragma("message(\\"why\\")")')

    def test_what_is_not_a_message(self):
        text = (
            '#error "Particles requires MPI support for I/O"\n'
            "#pragma omp parallel for\n"
            '#pragma GCC error "stop"\n'
            '_Pragma("omp parallel") int a;\n'
            '#pragma clang diagnostic push\n'
            'auto s = "#warning not a directive";\n'
        )
        self.assertEqual(directive_texts(text), [])

    def test_a_message_inside_disabled_code_is_part_of_it(self):
        text = '#if 0\n#warning "a"\n_Pragma("message(\\"b\\")")\n#else\n#warning "c"\n#endif\n'
        self.assertEqual([(p, line) for p, line, _ in directive_texts(text)],
                         [(Placement.GUARD, 1), (Placement.MESSAGE, 5)])


@unittest.skipUnless(HAVE_CLANG and HAVE_UNIFDEF, "needs clang++ and unifdef on PATH")
class AgreesWithUnifdef(unittest.TestCase):
    def folds(self, text):
        proc = subprocess.run(["unifdef", "-k"], input=text, capture_output=True, text=True)
        self.assertIn(proc.returncode, (0, 1), proc.stderr)
        return proc.stdout != text

    def test_a_guard_is_found_exactly_where_unifdef_folds(self):
        for condition in [c for c, _ in CONSTANT] + UNKNOWN:
            text = f"#if {condition}\nA\n#else\nB\n#endif\n"
            with self.subTest(condition=condition):
                self.assertEqual(bool(directive_regions(lex(text))), self.folds(text))

    def test_elif_chains_and_nesting(self):
        for text in [
            "#if X\nA\n#elif 0\nB\n#else\nC\n#endif\n",
            "#if X\nA\n#elif 1\nB\n#else\nC\n#endif\n",
            "#if 0\nA\n#elif X\nB\n#endif\n",
            "#ifdef X\n#elif 0\nB\n#endif\n",
            "#if X\n#if Y\n#endif\n#elif Z\n#endif\n",
            "/* c */ #if 0\nA\n#endif\n",
            "#if 1 /* multi\nline */\nA\n#endif\n",
            "%:if 0\nA\n%:endif\n",
            "#if \\\n 0\nA\n#endif\n",
        ]:
            with self.subTest(text=text):
                self.assertEqual(bool(directive_regions(lex(text))), self.folds(text))


if __name__ == "__main__":
    unittest.main()
