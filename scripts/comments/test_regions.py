from __future__ import annotations

import shutil
import unittest

import lex_clang
from lex_clang import lex
from regions import comment_regions
from tokens import Placement

HAVE_CLANG = shutil.which(lex_clang.CLANG) is not None


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


if __name__ == "__main__":
    unittest.main()
