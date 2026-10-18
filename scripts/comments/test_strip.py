import unittest

from files import Lang
from rules import check
from strip import Refused, strip

CXX_CASES = {
    "tdd example": (
        "int f(int x) {\n  // Restore the clocks first.\n\n  restore();  // see RestoreLevelClocks\n"
        "  g(/*order*/ 5, /*bool*/ true);\n}\n",
        "int f(int x) {\n  restore();\n  g(5, true);\n}\n",
    ),
    "two blank lines touching keep one": ("a;\n\n// c\n\nb;\n", "a;\n\nb;\n"),
    "blank line at the top of the file": ("// header\n\nint a;\n", "int a;\n"),
    "every blank line it leaves at the top of the file": ("// header\n\n\n\n// c\nint a;\n", "int a;\n"),
    "every blank line it leaves right after a brace": ("{\n  // c\n\n\n  a;\n}\n", "{\n  a;\n}\n"),
    "every blank line it leaves at the end of the file": ("a;\n\n\n// c\n", "a;\n"),
    "blank line right after a brace": ("{\n  // c\n\n  a;\n}\n", "{\n  a;\n}\n"),
    "blank line right before a brace": ("{\n  a;\n\n  // c\n}\n", "{\n  a;\n}\n"),
    "blank line at the end of the file": ("a;\n\n// c\n", "a;\n"),
    "existing blank lines elsewhere stay": ("a;\n\n\nb; // c\n", "a;\n\n\nb;\n"),
    "an existing blank line after a brace stays": ("{\n\n  // c\n  a;\n}\n", "{\n\n  a;\n}\n"),
    "namespace closer stays": (
        "namespace X {\n// c\nint a;\n} // namespace X\n",
        "namespace X {\nint a;\n} // namespace X\n",
    ),
    "line-break guard stays": ("f(a, // x\n  b, //\n  c);\n", "f(a,\n  b, //\n  c);\n"),
    "trailing block comment": ("int a; /* c */  \n", "int a;\n"),
    "trailing block comment across lines": ("int a; /* c\n d */\nint b;\n", "int a;\nint b;\n"),
    "inline after an open paren": ("f(/*a*/ 1);\n", "f(1);\n"),
    "inline after a space": ("g(1, /*b*/ 2);\n", "g(1, 2);\n"),
    "inline between words": ("int /*x*/y;\n", "int y;\n"),
    "inline after a word": ("a/*x*/;\n", "a;\n"),
    "inline without space keeps tokens apart": ("a/*x*/b;\nc+/*x*/+d;\n", "a b;\nc+ +d;\n"),
    "inline at the line start": ("  /* c */ int a;\n", "  int a;\n"),
    "inline before a closer or separator": (
        "assert(a == 1 /*x*/ || b == 5 /*y*/);\nf(IntVect{0} /* c */,\n  g(1 /*all*/), x /*d*/ ;\n"
        "int h(int /*n*/, int /*m*/) {\n}\n",
        "assert(a == 1 || b == 5);\nf(IntVect{0},\n  g(1), x;\nint h(int, int) {\n}\n",
    ),
    "inline before a closer at the line start keeps the indentation": ("  /* c */ );\n", "  );\n"),
    "inline before an operand keeps the space": ("report(gi, /*vi=*/-1, /*tl=*/0);\n", "report(gi, -1, 0);\n"),
    "inline block across lines": ("/* a\n b */ int x;\n", "int x;\n"),
    "comments in strings stay": ('s = "a // b"; // c\n', 's = "a // b";\n'),
    "two comments on one line": ("/* a */ // b\nint x;\n", "int x;\n"),
    "if 0": ("a;\n\n#if 0\nb; // c\n#endif\n\nc;\n", "a;\n\nc;\n"),
    "if 0 else keeps the live branch byte for byte": (
        "{\n#if 0\nb;\n#else\n\n  c;  \n\n#endif\n}\n",
        "{\n\n  c;  \n\n}\n",
    ),
    "if 1": ("#if 1\nc;\n#endif\n", "c;\n"),
    "if 1 else": ("#if 1\nc;\n#else\nb;\n#endif\n", "c;\n"),
    "guard in a live branch": ("#if 1\n#if 0\nx;\n#endif\ny;\n#endif\n", "y;\n"),
    "comment in a live branch": ("#if 0\nx;\n#else\n// c\ny;\n#endif\n", "y;\n"),
    "crlf line endings": ("int a; // c\r\n// d\r\nint b;\r\n", "int a;\r\nint b;\r\n"),
    "file without a final newline": ("a;\n// c", "a;\n"),
    "comment-only file": ("// a\n\n// b\n", ""),
    "a citation stays, a citation inside prose goes": (
        "{\n  // arXiv:1111.2177 [gr-qc], (71)\n  a = 1;\n"
        "  // Based on <https://example.org/lib>. Close to what we need.\n  b = 2;\n"
        "  c = 3; // https://stackoverflow.com/a/30968371\n}\n",
        "{\n  // arXiv:1111.2177 [gr-qc], (71)\n  a = 1;\n  b = 2;\n"
        "  c = 3; // https://stackoverflow.com/a/30968371\n}\n",
    ),
}

HASH_CASES = {
    "schedule.ccl": (
        Lang.CCL,
        "# Schedule definitions for thorn X\n\nSTORAGE: a\n\n# SCHEDULE Old AT y\n# {\n# }\n\n"
        "SCHEDULE X AT Y\n{\n  LANG: C\n  # why\n  SYNC: a\n} \"desc # kept\"\n",
        "STORAGE: a\n\nSCHEDULE X AT Y\n{\n  LANG: C\n  SYNC: a\n} \"desc # kept\"\n",
    ),
    "par trailing": (Lang.PAR, 'A::b = 1   # c\nA::c = "x # y"\n', 'A::b = 1\nA::c = "x # y"\n'),
    "par toggles": (Lang.PAR, "A::a = 1\n\n#A::b = 2\n\nA::c = 3\n", "A::a = 1\n\nA::c = 3\n"),
    "comment-only param.ccl": (Lang.CCL, "# Parameter definitions for thorn X\n", ""),
    "schedule.ccl whose header is followed by a blank run": (
        Lang.CCL,
        "# Schedule definitions for thorn X\n\n\n\n# why\nSCHEDULE GROUP G AT postinitial\n{\n} \"G\"\n",
        "SCHEDULE GROUP G AT postinitial\n{\n} \"G\"\n",
    ),
    "comment-only file with blank lines": (Lang.CCL, "# a\n\n# b\n\n", ""),
}


class Strip(unittest.TestCase):
    def test_cxx(self):
        for name, (before, after) in CXX_CASES.items():
            with self.subTest(name):
                self.assertEqual(strip(before, Lang.CXX), after)

    def test_hash(self):
        for name, (lang, before, after) in HASH_CASES.items():
            with self.subTest(name):
                self.assertEqual(strip(before, lang), after)

    def test_strip_leaves_nothing_for_check_and_is_idempotent(self):
        cases = [(Lang.CXX, b) for b, _ in CXX_CASES.values()] + [(l, b) for l, b, _ in HASH_CASES.values()]
        for lang, before in cases:
            with self.subTest(before=before):
                after = strip(before, lang)
                self.assertEqual(check(after, lang), [])
                self.assertEqual(strip(after, lang), after)

    def test_clean_text_is_unchanged(self):
        text = "namespace X {\n\nint a; //\n\n\n} // namespace X\n"
        self.assertEqual(strip(text, Lang.CXX), text)

    def test_elif_under_a_constant_guard_is_refused(self):
        with self.assertRaises(Refused) as caught:
            strip("a;\n#if 0 && defined A\nx;\n#elif 0 && defined B\ny;\n#else\nz;\n#endif\n", Lang.CXX)
        self.assertEqual(caught.exception.line, 2)

    def test_elif_chain_inside_a_real_conditional_is_refused(self):
        text = (
            "#ifndef SIMD_DISABLE\n  f();\n#else\n  if (m) {\n"
            "#if 0 && defined __CUDACC__\n    g();\n#elif 0 && defined __HIPCC__\n    h();\n#else\n"
            "#if 0 && defined __llvm__\n    h();\n#else\n    *ptr = x;\n#endif\n#endif\n  }\n#endif\n"
        )
        with self.assertRaises(Refused) as caught:
            strip(text, Lang.CXX)
        self.assertEqual(caught.exception.line, 5)


if __name__ == "__main__":
    unittest.main()
