import os
import re
import shutil
import unittest

from files import Lang
from strip import strip
from verify import peg_for, verify_file

HAVE_CXX_TOOLS = shutil.which("clang++") is not None and shutil.which("unifdef") is not None
FLESH = os.environ.get("CACTUSX")
HAVE_FLESH = (
    shutil.which("perl") is not None
    and FLESH is not None
    and os.path.isfile(os.path.join(FLESH, "lib", "sbin", "Piraha.pm"))
)


class PegFor(unittest.TestCase):
    def test_grammar_per_file(self):
        self.assertEqual(peg_for("X/interface.ccl"), "interface.peg")
        self.assertEqual(peg_for("X/param.ccl"), "param.peg")
        self.assertEqual(peg_for("X/schedule.ccl"), "schedule.peg")
        self.assertEqual(peg_for("X/configuration.ccl"), "config.peg")
        self.assertEqual(peg_for("X/test/a.par"), "par.peg")
        self.assertEqual(peg_for("X/test/a.par-DISABLED"), "par.peg")
        self.assertIsNone(peg_for("X/test/test.ccl"))

    def test_uncovered_and_unchecked_files_fail(self):
        self.assertFalse(verify_file("X/README", "a", "b", None).ok)
        self.assertFalse(verify_file("X/test/test.ccl", "# a\nNPROCS 1\n", "NPROCS 1\n", None).ok)


@unittest.skipUnless(HAVE_CXX_TOOLS, "needs clang++ and unifdef")
class Cxx(unittest.TestCase):
    BEFORE = (
        "#include <a.hxx> // why\n"
        'const char *s = "a // b"; /* c */\n'
        "auto r = R\"(x // y)\";\n"
        "int n = 1'000; // d\n"
        "#if 0\nold();\n#endif\n"
        "#if 1\nkept();\n#else\ndropped();\n#endif\n"
        "f(/*order*/ 5);\n"
    )

    def test_our_strip_passes(self):
        result = verify_file("X/src/a.cxx", self.BEFORE, strip(self.BEFORE, Lang.CXX), None)
        self.assertTrue(result.ok, result.detail)

    def test_negative_control_a_strip_that_eats_slashes_in_a_string_fails(self):
        naive = re.sub(r"[ \t]*//.*", "", self.BEFORE)
        self.assertIn('const char *s = "a\n', naive)
        self.assertFalse(verify_file("X/src/a.cxx", self.BEFORE, naive, None).ok)

    def test_a_code_change_fails(self):
        self.assertFalse(verify_file("X/src/a.cxx", "int a = 1;\n", "int a = 2;\n", None).ok)

    def test_joining_a_directive_with_the_next_line_fails(self):
        result = verify_file("X/src/a.cxx", "#define A 1\nint x;\n", "#define A 1 int x;\n", None)
        self.assertFalse(result.ok)

    def test_dropping_a_live_branch_fails(self):
        before = "#if 0\nx;\n#else\ny;\n#endif\n"
        self.assertFalse(verify_file("X/src/a.cxx", before, "", None).ok)

    def test_removing_a_todo_warning_passes(self):
        before = (
            '#warning "TODO: Restrict"\nvoid f();\n#if 1\n{\n#warning           \\\n'
            '    "TODO: conservative"\n  int a;\n}\n#endif\n'
        )
        after = strip(before, Lang.CXX)
        self.assertNotIn("#warning", after)
        result = verify_file("X/src/a.cxx", before, after, None)
        self.assertTrue(result.ok, result.detail)
        self.assertIn("#warning TODO directives dropped: 2", result.evidence)

    def test_removing_another_warning_fails(self):
        before = '#warning "SIMD is disabled"\nvoid f();\n'
        self.assertFalse(verify_file("X/src/a.cxx", before, "void f();\n", None).ok)

    def test_removing_the_line_after_a_todo_warning_fails(self):
        before = '#warning "TODO: x"\nvoid f();\nvoid g();\n'
        self.assertFalse(verify_file("X/src/a.cxx", before, "void g();\n", None).ok)

    def test_a_hand_resolved_elif_chain_passes(self):
        before = (
            "  if (m) {\n#if 0 && defined __CUDACC__\n    g(); // why\n#elif 0 && defined __HIPCC__\n"
            "    h();\n#else\n    // CPU\n#if 0 && defined __llvm__\n    h();\n#else\n    *ptr = x;\n"
            "#endif\n#endif\n  }\n"
        )
        result = verify_file("X/src/a.cxx", before, "  if (m) {\n    *ptr = x;\n  }\n", None)
        self.assertTrue(result.ok, result.detail)
        self.assertFalse(verify_file("X/src/a.cxx", before, "  if (m) {\n    h();\n  }\n", None).ok)


@unittest.skipUnless(HAVE_FLESH, "needs perl and a Cactus flesh in $CACTUSX")
class Piraha(unittest.TestCase):
    def test_par_comment_removal_passes(self):
        before = '# header\n\nA::b = 1  # why\n#A::c = 2\nA::d = "x # y"\n'
        result = verify_file("X/test/a.par", before, strip(before, Lang.PAR), FLESH)
        self.assertTrue(result.ok, result.detail)

    def test_par_value_change_fails(self):
        self.assertFalse(verify_file("X/test/a.par", "A::b = 1\n", "A::b = 2\n", FLESH).ok)

    def test_removing_a_hash_inside_a_par_string_fails(self):
        self.assertFalse(verify_file("X/test/a.par", 'A::d = "x # y"\n', 'A::d = "x "\n', FLESH).ok)

    def test_schedule_comment_removal_passes(self):
        before = (
            "# Schedule definitions for thorn X\n\nSTORAGE: a\n\n# SCHEDULE Old AT initial\n"
            "SCHEDULE X_Init AT initial\n{\n  LANG: C\n  # why\n  WRITES: a(interior)\n"
            '} "Init # kept"\n'
        )
        result = verify_file("X/schedule.ccl", before, strip(before, Lang.CCL), FLESH)
        self.assertTrue(result.ok, result.detail)

    def test_schedule_description_change_fails(self):
        before = 'SCHEDULE X_Init AT initial\n{\n  LANG: C\n} "Init # kept"\n'
        after = 'SCHEDULE X_Init AT initial\n{\n  LANG: C\n} "Init"\n'
        self.assertFalse(verify_file("X/schedule.ccl", before, after, FLESH).ok)

    def test_unparseable_file_fails(self):
        self.assertFalse(verify_file("X/param.ccl", "BOOLEAN {\n", "BOOLEAN {\n", FLESH).ok)

    def test_a_comment_only_file_stripped_to_nothing_passes(self):
        for name in ["param.ccl", "schedule.ccl", "configuration.ccl", "test/a.par"]:
            for before in ["# Parameter definitions for thorn X\n", "# a\n\n  # b\n\n"]:
                with self.subTest(name=name, before=before):
                    after = strip(before, Lang.PAR if name.endswith(".par") else Lang.CCL)
                    self.assertEqual(after, "")
                    result = verify_file(f"X/{name}", before, after, FLESH)
                    self.assertTrue(result.ok, result.detail)
                    self.assertIn("empty after", result.evidence)

    def test_emptying_a_file_with_code_fails(self):
        before = '# Parameter definitions for thorn X\nBOOLEAN b "b"\n{\n} no\n'
        self.assertFalse(verify_file("X/param.ccl", before, "", FLESH).ok)

    def test_changing_a_comment_only_file_without_emptying_it_fails(self):
        self.assertFalse(verify_file("X/param.ccl", "# a\n# b\n", "# b\n", FLESH).ok)


if __name__ == "__main__":
    unittest.main()
