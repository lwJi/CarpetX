import os
import re
import shutil
import unittest

from files import Lang
from strip import strip
from verify import only_string_comments_removed, peg_for, verify_file

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

    def test_uncovered_files_fail(self):
        self.assertFalse(verify_file("X/README", "a", "b", None).ok)


class TestCclAudit(unittest.TestCase):
    BEFORE = (
        "# Silo checkpoints can only be recovered on the rank count\n# they were written with.\n\n"
        "NPROCS 1\n\nTEST a\n{\n  # These tolerances are ridiculous.\n  ABSTOL 2.0e-4 # why\n"
        '  RELTOL 1\n}\nEXTENSIONS "x # y"\n'
    )

    def check(self, before, after, name="X/test/test.ccl"):
        return verify_file(name, before, after, None)

    def test_our_strip_passes(self):
        after = strip(self.BEFORE, Lang.CCL)
        self.assertEqual(after, 'NPROCS 1\n\nTEST a\n{\n  ABSTOL 2.0e-4\n  RELTOL 1\n}\nEXTENSIONS "x # y"\n')
        result = self.check(self.BEFORE, after)
        self.assertTrue(result.ok, result.detail)
        self.assertIn("line audit", result.evidence)
        self.assertTrue(self.check("NPROCS 1\nABSTOL 1e-12 # a\n", "NPROCS 1\nABSTOL 1e-12\n",
                                   "TestInterpolate/test/tests.ccl").ok)

    def test_a_comment_only_file_stripped_to_nothing_passes(self):
        self.assertTrue(self.check("# a\n\n# b\n", "").ok)

    def test_a_changed_value_fails(self):
        self.assertFalse(self.check("NPROCS 1\n", "NPROCS 2\n").ok)
        self.assertFalse(self.check("ABSTOL 1 # a\n", "ABSTOL 2\n").ok)

    def test_a_deleted_code_line_fails(self):
        self.assertFalse(self.check("# a\nNPROCS 1\nABSTOL 1\n", "NPROCS 1\n").ok)

    def test_an_added_line_fails(self):
        self.assertFalse(self.check("NPROCS 1\n", "NPROCS 1\nABSTOL 1\n").ok)
        self.assertFalse(self.check("# a\n", "NPROCS 1\n").ok)

    def test_cutting_inside_a_string_fails(self):
        self.assertFalse(self.check('EXTENSIONS "x # y"\n', 'EXTENSIONS "x\n').ok)

    def test_a_line_emptied_but_not_deleted_fails(self):
        self.assertFalse(self.check("NPROCS 1 # a\n", "\n").ok)
        self.assertFalse(self.check("NPROCS 1 # a\n", "# a\n").ok)


class StringModel(unittest.TestCase):
    def ok(self, before, after):
        return only_string_comments_removed(before.replace("<", "\x01").replace(">", "\x02"), after)

    def test_a_comment_line_goes_with_its_newline(self):
        self.assertTrue(self.ok("\n  a\n  <# c>\n  b\n", "\n  a\n  b\n"))
        self.assertTrue(self.ok("\n  a\n\n  <# c>\n\n  b\n", "\n  a\n\n\n  b\n"))

    def test_a_trailing_comment_goes_with_the_blanks_before_it(self):
        self.assertTrue(self.ok("\n  a  <# c>\n", "\n  a\n"))
        self.assertTrue(self.ok("\n  a <# c>\n", "\n  a \n"))
        self.assertTrue(self.ok("a\n  <# c>", "a\n  "))

    def test_anything_else_fails(self):
        self.assertFalse(self.ok("\n  a\n\n  b\n", "\n  a\n  b\n"))
        self.assertFalse(self.ok("\n  a <# c>\n  b\n", "\n  \n  b\n"))
        self.assertFalse(self.ok("\n  a\n  <# c>\n", "\n  a\n  A\n"))
        self.assertFalse(self.ok("x # y\n  z\n", "x\n  z\n"))


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

    def test_par_string_comment_removal_passes(self):
        before = (
            'ActiveThorns = "\n    # SystemTopology\n    ADMBase\n"\n'
            'A::v = "x # y\n  a # b\n\n  # c\n\n  d\n"\n'
        )
        after = strip(before, Lang.PAR)
        self.assertEqual(after, 'ActiveThorns = "\n    ADMBase\n"\nA::v = "x # y\n  a\n\n\n  d\n"\n')
        result = verify_file("X/test/a.par", before, after, FLESH)
        self.assertTrue(result.ok, result.detail)
        self.assertIn("but for 3 string lines that par.peg's stringcomment drops", result.evidence)

    def test_removing_a_hash_on_the_first_line_of_a_string_fails(self):
        for before, after in [
            ('A::v = "x # y\n  z\n"\n', 'A::v = "x\n  z\n"\n'),
            ('ActiveThorns = "# first\n  B\n"\n', 'ActiveThorns = "\n  B\n"\n'),
        ]:
            with self.subTest(before=before):
                self.assertFalse(verify_file("X/test/a.par", before, after, FLESH).ok)

    def test_changing_a_string_beyond_its_comments_fails(self):
        for before, after in [
            ('A::v = "\n  a\n  # c\n  b\n"\n', 'A::v = "\n  a\n  B\n"\n'),
            ('A::v = "\n  a\n\n  b\n"\n', 'A::v = "\n  a\n  b\n"\n'),
            ('A::v = "\n  a # c\n"\n', 'A::v = "\n  \n"\n'),
        ]:
            with self.subTest(before=before):
                self.assertFalse(verify_file("X/test/a.par", before, after, FLESH).ok)

    def test_text_next_to_a_variable_in_a_string_counts(self):
        self.assertFalse(verify_file("X/test/a.par", 'A::v = "x $parfile"\n', 'A::v = "y $parfile"\n', FLESH).ok)

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
