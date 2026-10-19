from __future__ import annotations

import shutil
import unittest

import lex_clang
from files import Lang
from rules import check, lex
from strip import strip, unifdef_k
from tokens import Placement

HAVE_CLANG = shutil.which(lex_clang.CLANG) is not None
HAVE_UNIFDEF = shutil.which("unifdef") is not None

CXX = {
    "whole-line comments go with their lines": (
        "int f(int x) {\n  // Restore the clocks first.\n  restore();\n}\n",
        "int f(int x) {\n  restore();\n}\n",
    ),
    "a trailing comment goes with the space before it": ("int z;   // tab\n", "int z;\n"),
    "two trailing comments": ("int a; /* b */ // c\n", "int a;\n"),
    "an inline comment at a line start keeps the indentation": (
        "  /*constexpr*/ ARITH_INLINE int f();\n", "  ARITH_INLINE int f();\n"
    ),
    "an inline comment goes with one adjacent space": ("g(1, /*b*/ 2);\nint /*x*/y;\n", "g(1, 2);\nint y;\n"),
    "an inline label keeps the space before an operand": ("f(cctkGH, /*vi=*/-1, /*tl=*/0);\n", "f(cctkGH, -1, 0);\n"),
    "after an opener": ("f(/*a*/ 1);\n", "f(1);\n"),
    "before a closer or separator": (
        "f(IntVect{0} /* don't */,\n  g(1 /*all*/), x /*d*/ ;\nint h(int /*n*/, int /*m*/);\n",
        "f(IntVect{0},\n  g(1), x;\nint h(int, int);\n",
    ),
    "tokens that would glue keep one space": ("a/**/b;\nc+/*x*/+d;\n", "a b;\nc+ +d;\n"),
    "before a call or subscript after a name": (
        "y = var.copy(make_valid_int /*all*/ ());\nint f(const CCTK_REAL /*origin*/[]);\n",
        "y = var.copy(make_valid_int());\nint f(const CCTK_REAL[]);\n",
    ),
    "before parens after a keyword": ("if /*x*/ (a)\n  return /*y*/ (b);\n", "if (a)\n  return (b);\n"),
    "a multi-line inline comment": ("/* a\n b */ int x;\nint y; /* c\n d */\nint z;\n", "int x;\nint y;\nint z;\n"),
    "a directive keeps every gap": (
        "#define F/* c */(x)\n#define G(a) f(a /* c */)\n#define K(a,/*c*/b) a\n",
        "#define F (x)\n#define G(a) f(a )\n#define K(a, b) a\n",
    ),
    "a directive keeps its line splices": (
        "#define F(x) \\\n  g(x) /* c */ \\\n  /* d */ h(x)\n", "#define F(x) \\\n  g(x) \\\n  h(x)\n"
    ),
    "a trailing comment on a directive": (
        "#include <a.hxx> // why\n#ifdef FOO\n#endif // FOO\n", "#include <a.hxx>\n#ifdef FOO\n#endif\n"
    ),
    "a header name holds no comment": ("#include <a//b.h> // c\n", "#include <a//b.h>\n"),
    "comment openers in literals stay": (
        's = "a // b"; // c\nauto r = R"(x // y /* z)"; // d\nchar c = \'/\'; // e\n',
        's = "a // b";\nauto r = R"(x // y /* z)";\nchar c = \'/\';\n',
    ),
    "two blank runs that meet keep the longer": ("a;\n\n// c\n\nb;\nc;\n\n\n// d\n\ne;\n", "a;\n\nb;\nc;\n\n\ne;\n"),
    "blank lines elsewhere stay": ("a;\n\n\nb; // c\n", "a;\n\n\nb;\n"),
    "blank lines it leaves at the file's edges go": ("// header\n\n\n// c\nint a;\n\n// d\n", "int a;\n"),
    "blank lines it leaves after an opening brace go": ("{\n  // c\n\n\n  a;\n}\n", "{\n  a;\n}\n"),
    "blank lines it leaves before a closing brace go": ("{\n  a;\n\n  // c\n}\n", "{\n  a;\n}\n"),
    "a blank line already after an opening brace stays": ("{\n\n  // c\n  a;\n}\n", "{\n\n  a;\n}\n"),
    "after a brace that lost its trailing comment": (
        "} else { // a\n  // b\n\n  d;\n}\n} else { // a\n\n  d;\n}\n",
        "} else {\n  d;\n}\n} else {\n\n  d;\n}\n",
    ),
    "a blank line before a closing brace that lost its comment goes": (
        "  for (;;) {\n    x;\n\n  } // for i\n\n} // for j\n", "  for (;;) {\n    x;\n  }\n}\n"
    ),
    "no blank line after an access specifier": (
        "class A {\npublic:\n  // Member types\n\n  using T = int;\n};\n", "class A {\npublic:\n  using T = int;\n};\n"
    ),
    "a blank line before an access specifier that ends a block of members": (
        "class A {\n  int a;\n  // int b;\nprivate:\n  int c;\n};\n",
        "class A {\n  int a;\n\nprivate:\n  int c;\n};\n",
    ),
    "namespace closers and line-break guards stay": (
        "namespace X {\n// c\nf(a, // x\n  b, //\n  c);\n} // namespace X\n",
        "namespace X {\nf(a,\n  b, //\n  c);\n} // namespace X\n",
    ),
    "clang-format markers stay": ("// clang-format off\n// why\nint a;\n// clang-format on\n",
                                  "// clang-format off\nint a;\n// clang-format on\n"),
    "a citation stays and prose around it goes": (
        "{\n  // See arXiv:1111.2177 [gr-qc], (74-75)\n  a = 1;\n  // Based on <https://example.org/lib>. Close.\n"
        "  b = 2;\n  c = 3; // https://stackoverflow.com/a/30968371\n}\n",
        "{\n  // See arXiv:1111.2177 [gr-qc], (74-75)\n  a = 1;\n  b = 2;\n  c = 3; // https://stackoverflow.com/a/30968371\n}\n",
    ),
    "a separator goes with the prose after the citation it set off": (
        "f();\n// See <https://en.wikipedia.org/wiki/Brent%27s_method>\n//\n// Returns a bracket.\n// More.\nint g();\n",
        "f();\n// See <https://en.wikipedia.org/wiki/Brent%27s_method>\nint g();\n",
    ),
    "a separator goes with the prose before the citation": (
        "// Prose.\n//\n// doi:10.1016/0021-9991(88)90177-5\nint g();\n", "// doi:10.1016/0021-9991(88)90177-5\nint g();\n"
    ),
    "a separator between two citations stays": (
        "// arXiv:1111.2177\n//\n// doi:10.1016/0021-9991(88)90177-5\nint g();\n",
        "// arXiv:1111.2177\n//\n// doi:10.1016/0021-9991(88)90177-5\nint g();\n",
    ),
    "the separators between dropped runs go and the citations around them join": (
        "// arXiv:1111.2177\n//\n// prose\n//\n// arXiv:1203.4536\nint g();\n",
        "// arXiv:1111.2177\n// arXiv:1203.4536\nint g();\n",
    ),
    "a separator that had code on both sides stays": (
        "auto k = [\n#ifdef A\n    a = a,\n#endif\n    x = x,\n    //\n    cmin, cmax](int i) {};\nint x;\n//\n//\nint y;\n",
        "auto k = [\n#ifdef A\n    a = a,\n#endif\n    x = x,\n    //\n    cmin, cmax](int i) {};\nint x;\n//\n//\nint y;\n",
    ),
    "a separator orphaned between code lines by a dropped run goes": (
        "int x;\n// prose\n//\n// more prose\nint y;\nint z;\n//\n// prose\nint w;\n",
        "int x;\nint y;\nint z;\nint w;\n",
    ),
    "a separator beside a blank line goes": ("int x;\n\n//\nint y;\n//\n\nint z;\n", "int x;\n\nint y;\n\nint z;\n"),
    "a compile-time message stays for a hand edit": ('#warning "TODO: x" // c\nint a;\n', '#warning "TODO: x"\nint a;\n'),
    "crlf line endings": ("int a; // c\r\n// d\r\nint b;\r\n", "int a;\r\nint b;\r\n"),
    "a file without a final newline": ("a;\n// c", "a;\n"),
    "a comment-only file": ("// a\n\n// b\n", ""),
    "non-ASCII text in a comment": ("int x; // L'Hôpital ⁻¹\n// Résumé\nint y;\n", "int x;\nint y;\n"),
}

GUARDS = {
    "an if 0 block and the blank runs it joined": ("a;\n\n#if 0\nb; // c\n#endif\n\nc;\n", "a;\n\nc;\n"),
    "an if 1 block keeps its live branch": ("#if 1\nc;\n#else\nb;\n#endif\n", "c;\n"),
    "a guard in a live branch": ("#if 1\n#if 0\nx;\n#endif\ny;\n#endif\n", "y;\n"),
    "a comment in the live branch": ("#if 0\nx;\n#else\n// c\ny;\n#endif\n", "y;\n"),
    "an elif chain unifdef folds": (
        "  if (m) {\n#if 0 && defined __CUDACC__\n    g(); // why\n#elif 0 && defined __HIPCC__\n    h();\n#else\n"
        "    // CPU\n#if 0 && defined __llvm__\n    h();\n#else\n    *ptr = x;\n#endif\n#endif\n  }\n",
        "  if (m) {\n    *ptr = x;\n  }\n",
    ),
    "an elif that becomes first, as unifdef rewrites it": (
        "#if 0\nx;\n#elif defined A\ny;\n#endif\n", "#if   defined A\ny;\n#endif\n"
    ),
    "a guard that depends on a macro stays": (
        "#if 0 || defined A\nx;\n#endif\n#ifdef B\ny;\n#endif\n", "#if 0 || defined A\nx;\n#endif\n#ifdef B\ny;\n#endif\n"
    ),
    "a raw string that unifdef cannot lex": (
        'const char *s = R"EOS(\n  "a": 1,\n#if 0\n)EOS";\n#if 0\nx;\n#endif\nint n = 1\'000;\n',
        'const char *s = R"EOS(\n  "a": 1,\n#if 0\n)EOS";\nint n = 1\'000;\n',
    ),
}

HASH = {
    "schedule.ccl": (
        Lang.SCHEDULE,
        "# Schedule definitions for thorn X\n\nSTORAGE: a\n\n# SCHEDULE Old AT y\n# {\n# }\n\n"
        "SCHEDULE X AT Y\n{\n  LANG: C /* why */\n  # why\n  SYNC: a\n} \"desc # kept\"\n",
        "STORAGE: a\n\nSCHEDULE X AT Y\n{\n  LANG: C\n  SYNC: a\n} \"desc # kept\"\n",
    ),
    "a comment-only param.ccl": (Lang.PARAM, "# Parameter definitions for thorn X\n", ""),
    "interface.ccl": (
        Lang.INTERFACE,
        "# Interface definition for thorn X\n\nIMPLEMENTS: X\n\nCCTK_REAL a TYPE=gf # why\n{\n  u\n} \"it's # kept\"\n",
        "IMPLEMENTS: X\n\nCCTK_REAL a TYPE=gf\n{\n  u\n} \"it's # kept\"\n",
    ),
    "par trailing comments and disabled settings": (
        Lang.PAR,
        "# header\n\nA::b = 1   # c\nA::c = \"x # y\"\n\n#A::d = 2\n\nA::e = 3\n",
        "A::b = 1\nA::c = \"x # y\"\n\nA::e = 3\n",
    ),
    "a par string keeps its first line and its blank lines": (
        Lang.PAR,
        'ActiveThorns = "\n    # SystemTopology\n    ADMBase\n"\nA::v = "x # y\n  a  # b\n\n  # c\n\n  d\n"\n',
        'ActiveThorns = "\n    ADMBase\n"\nA::v = "x # y\n  a\n\n\n  d\n"\n',
    ),
    "a par citation stays": (Lang.PAR, "# See <https://example.org/a>\nA::b = 1\n", "# See <https://example.org/a>\nA::b = 1\n"),
    "make.code.defn": (
        Lang.MAKE,
        "# Main make.code.defn file for thorn X\n\n# Source files in this directory\nSRCS = a.cxx # c\n\n"
        "# Subdirectories containing source files\nSUBDIRS =\n",
        "SRCS = a.cxx\n\nSUBDIRS =\n",
    ),
    "test.ccl": (
        Lang.TEST,
        "# Silo checkpoints\n\nNPROCS 1\n\nTEST a\n{\n  # ridiculous\n  ABSTOL 2.0e-4 # why\n}\nEXTENSIONS \"x # y\"\n",
        "NPROCS 1\n\nTEST a\n{\n  ABSTOL 2.0e-4\n}\nEXTENSIONS \"x # y\"\n",
    ),
    "fortran": (
        Lang.FORTRAN,
        "#include <cctk.h>\n\nsubroutine S()\n  integer :: i ! why\n\n  ! check\n"
        "  if (i /= 1) call CCTK_WARN(1, \"a ! b\")\n  s = 'abc&\n! note\n   &def ! x' ! z\n!$omp barrier\nend subroutine S\n",
        "#include <cctk.h>\n\nsubroutine S()\n  integer :: i\n\n"
        "  if (i /= 1) call CCTK_WARN(1, \"a ! b\")\n  s = 'abc&\n   &def ! x'\n!$omp barrier\nend subroutine S\n",
    ),
}


def leftover(text, lang):
    return [v for v in check(lex(text, lang), lang) if v.region.placement is not Placement.MESSAGE]


class Hash(unittest.TestCase):
    def test_cases(self):
        for name, (lang, before, after) in HASH.items():
            with self.subTest(name):
                self.assertEqual(strip(before, lang), after)
                self.assertEqual(leftover(after, lang), [])
                self.assertEqual(strip(after, lang), after)

    def test_clean_text_is_unchanged(self):
        self.assertEqual(strip("A::b = 1\n\n\nA::c = 2\n", Lang.PAR), "A::b = 1\n\n\nA::c = 2\n")


@unittest.skipUnless(HAVE_CLANG and HAVE_UNIFDEF, "needs clang++ and unifdef on PATH")
class Cxx(unittest.TestCase):
    def test_cases(self):
        for name, (before, after) in {**CXX, **GUARDS}.items():
            with self.subTest(name):
                self.assertEqual(strip(before, Lang.CXX), after)
                self.assertEqual(leftover(after, Lang.CXX), [])
                self.assertEqual(strip(after, Lang.CXX), after)

    def test_clean_text_is_unchanged(self):
        text = "namespace X {\n\nint a; //\n\n\n#ifdef A\n#endif\n} // namespace X\n"
        self.assertEqual(strip(text, Lang.CXX), text)

    def test_unifdef_sees_literals_masked_and_gets_them_back(self):
        text = 'auto a = "__comments_literal_0_";\nauto r = R"(\n#endif\n)";\n#if 0\nx;\n#endif\n'
        self.assertEqual(unifdef_k(text), 'auto a = "__comments_literal_0_";\nauto r = R"(\n#endif\n)";\n')


if __name__ == "__main__":
    unittest.main()
