from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import unittest

import lex_clang
from files import Lang
from strip import strip
from upstream import listed, verify_skips
from verify import line_audit, only_string_comments_removed, verify_cxx, verify_files, verify_revs

HAVE_CXX_TOOLS = shutil.which(lex_clang.CLANG) is not None and shutil.which("unifdef") is not None
FLESH = os.environ.get("CACTUSX")
HAVE_FLESH = (
    shutil.which("perl") is not None
    and FLESH is not None
    and os.path.isfile(os.path.join(FLESH, "lib", "sbin", "Piraha.pm"))
)


def audit(before, after, lang):
    return line_audit("X/a", lang, before, after)


class MakeAudit(unittest.TestCase):
    BEFORE = "# Main make.code.defn file for thorn X\n\n# Source files\nSRCS = a.cxx b.cxx # c\n\nSUBDIRS =\n"

    def test_our_strip_passes(self):
        after = strip(self.BEFORE, Lang.MAKE)
        result = audit(self.BEFORE, after, Lang.MAKE)
        self.assertTrue(result.ok, result.why)
        self.assertIn("line audit", result.how)

    def test_a_changed_line_fails(self):
        self.assertFalse(audit("SRCS = a.cxx # c\n", "SRCS = b.cxx\n", Lang.MAKE).ok)
        self.assertFalse(audit("SRCS = a.cxx\n", "SRCS = a.cxx b.cxx\n", Lang.MAKE).ok)

    def test_an_escaped_hash_or_one_in_a_reference_is_not_a_comment(self):
        self.assertFalse(audit("X = a\\# b\n", "X = a\\\n", Lang.MAKE).ok)
        self.assertFalse(audit("X = $(subst #,y,z)\n", "X = $(subst\n", Lang.MAKE).ok)
        self.assertTrue(audit("X = a\\\\# b\n", "X = a\\\\\n", Lang.MAKE).ok)

    def test_a_continued_comment_goes_whole(self):
        self.assertTrue(audit("# a \\\n b\nX = 1\n", "X = 1\n", Lang.MAKE).ok)
        self.assertFalse(audit("# a\n b\nX = 1\n", "X = 1\n", Lang.MAKE).ok)

    def test_a_blank_line_may_go_only_beside_a_removed_comment(self):
        self.assertTrue(audit("A = 1\n\n# c\n\nB = 2\n", "A = 1\n\nB = 2\n", Lang.MAKE).ok)
        self.assertFalse(audit("A = 1\n\nB = 2\n", "A = 1\nB = 2\n", Lang.MAKE).ok)

    def test_a_line_emptied_but_kept_fails(self):
        self.assertFalse(audit("A = 1 # c\n", "\n", Lang.MAKE).ok)


class FortranAudit(unittest.TestCase):
    BEFORE = (
        "#include <cctk.h>\n#if !defined(X)\n#endif\n\nsubroutine S()\n  !// test\n"
        "  ! check that grid variable is of rank 3.\n  if(RANK(t) /= 3) then ! note rank 3\n"
        '      call CCTK_ERROR("a ! b")\n  endif\n  s = \'abc&\n! note\n   &def ! x\' ! why\n'
        "!$omp barrier ! c\nend subroutine S\n"
    )

    def test_our_strip_passes(self):
        after = strip(self.BEFORE, Lang.FORTRAN)
        self.assertNotIn("note rank", after)
        result = audit(self.BEFORE, after, Lang.FORTRAN)
        self.assertTrue(result.ok, result.why)

    def test_a_code_change_fails(self):
        self.assertFalse(audit("  x = 1 ! a\n", "  x = 2\n", Lang.FORTRAN).ok)

    def test_directives_and_sentinels_are_not_comments(self):
        self.assertFalse(audit("#if !defined(X)\n#endif\n", "#if\n#endif\n", Lang.FORTRAN).ok)
        self.assertFalse(audit("!$omp barrier\nx = 1\n", "x = 1\n", Lang.FORTRAN).ok)

    def test_cutting_inside_a_string_fails(self):
        self.assertFalse(audit('  call f("a ! b")\n', '  call f("a\n', Lang.FORTRAN).ok)
        self.assertTrue(audit("  call f('it''s') ! b\n", "  call f('it''s')\n", Lang.FORTRAN).ok)
        self.assertFalse(audit("s = 'abc&\n   &def ! x'\n", "s = 'abc&\n   &def\n", Lang.FORTRAN).ok)


class TestCclAudit(unittest.TestCase):
    def test_our_strip_passes(self):
        before = "# Silo\n\nNPROCS 1\n\nTEST a\n{\n  # These\n  ABSTOL 2.0e-4 # why\n}\nEXTENSIONS \"x # y\"\n"
        result = audit(before, strip(before, Lang.TEST), Lang.TEST)
        self.assertTrue(result.ok, result.why)

    def test_cutting_inside_a_string_fails(self):
        self.assertFalse(audit('EXTENSIONS "x # y"\n', 'EXTENSIONS "x\n', Lang.TEST).ok)
        self.assertFalse(audit("EXTENSIONS 'x # y'\n", "EXTENSIONS 'x\n", Lang.TEST).ok)
        self.assertTrue(audit('EXTENSIONS "it\\"s" # y\n', 'EXTENSIONS "it\\"s"\n', Lang.TEST).ok)

    def test_a_comment_only_file_stripped_to_nothing_passes(self):
        self.assertTrue(audit("# a\n\n# b\n", "", Lang.TEST).ok)


class StringModel(unittest.TestCase):
    def ok(self, before, after):
        return only_string_comments_removed(before.replace("<", "\x01").replace(">", "\x02"), after)

    def test_a_comment_line_goes_with_its_newline(self):
        self.assertTrue(self.ok("\n  a\n  <# c>\n  b\n", "\n  a\n  b\n"))
        self.assertTrue(self.ok("\n  a\n\n  <# c>\n\n  b\n", "\n  a\n\n\n  b\n"))

    def test_a_trailing_comment_goes_with_or_without_the_blanks_before_it(self):
        self.assertTrue(self.ok("\n  a  <# c>\n", "\n  a\n"))
        self.assertTrue(self.ok("\n  a <# c>", "\n  a "))

    def test_anything_else_fails(self):
        self.assertFalse(self.ok("\n  a\n\n  b\n", "\n  a\n  b\n"))
        self.assertFalse(self.ok("\n  a <# c>\n  b\n", "\n  \n  b\n"))
        self.assertFalse(self.ok("\n  a\n  <# c>\n", "\n  a\n  A\n"))
        self.assertFalse(self.ok("x # y\n  z\n", "x\n  z\n"))


@unittest.skipUnless(HAVE_CXX_TOOLS, "needs clang++ and unifdef on PATH")
class Cxx(unittest.TestCase):
    BEFORE = (
        "#include <a//b.h> // why\n"
        'const char *s = "a // b"; /* c */\n'
        "auto r = R\"(x // y)\";\n"
        "int n = 1'000; // d\n"
        "#if 0\nold();\n#endif\n"
        "#if 1\nkept();\n#else\ndropped();\n#endif\n"
        "#define F/* c */(x)\n#define G(a) f(a /* c */)\n"
        "f(/*order*/ 5, IntVect{0} /* e */, a/**/b);\n"
    )

    def check(self, before, after):
        return verify_cxx("X/src/a.cxx", before, after)

    def test_our_strip_passes(self):
        result = self.check(self.BEFORE, strip(self.BEFORE, Lang.CXX))
        self.assertTrue(result.ok, result.why)

    def test_negative_control_a_strip_that_eats_slashes_in_a_string_fails(self):
        naive = re.sub(r"[ \t]*//.*", "", self.BEFORE)
        self.assertIn('const char *s = "a\n', naive)
        self.assertFalse(self.check(self.BEFORE, naive).ok)

    def test_a_code_change_fails(self):
        self.assertFalse(self.check("int a = 1;\n", "int a = 2;\n").ok)

    def test_gluing_two_tokens_fails(self):
        result = self.check("int a/**/b;\n", "int ab;\n")
        self.assertFalse(result.ok)
        self.assertIn("'ab'", result.why)

    def test_joining_a_directive_with_the_next_line_fails(self):
        self.assertFalse(self.check("#define A 1 // c\nint x;\n", "#define A 1 int x;\n").ok)

    def test_a_macro_turned_function_like_fails(self):
        result = self.check("#define F/* c */(x)\n", "#define F(x)\n")
        self.assertFalse(result.ok)
        self.assertIn("no gap", result.why)

    def test_gaps_count_in_directives_and_between_operands_but_not_beside_punctuation(self):
        self.assertFalse(self.check("#define G(a) f(a /* c */)\n", "#define G(a) f(a)\n").ok)
        self.assertFalse(self.check("S(a/* c */+b);\n", "S(a+b);\n").ok)
        self.assertTrue(self.check("f(a /* c */, b);\ng(/* d */ c);\n", "f(a, b);\ng(c);\n").ok)

    def test_a_guard_strip_left_in_place_fails(self):
        result = self.check("#if 0 && defined FOO\nx;\n#endif\n", "#if 0 && defined FOO\nx;\n#endif\n")
        self.assertFalse(result.ok)

    def test_dropping_a_live_branch_or_a_message_fails(self):
        self.assertFalse(self.check("#if 0\nx;\n#else\ny;\n#endif\n", "").ok)
        self.assertFalse(self.check('#warning "TODO: x"\nvoid f();\n', "void f();\n").ok)

    def test_changing_a_header_name_fails(self):
        self.assertFalse(self.check("#include <a//b.h>\n", "#include <a\n").ok)
        self.assertFalse(self.check("#include <a//b.h>\n", "#include <a/b.h>\n").ok)


@unittest.skipUnless(HAVE_FLESH, "needs perl and a Cactus flesh in $CACTUSX")
class Piraha(unittest.TestCase):
    def check(self, name, lang, before, after):
        [result] = verify_files([(name, lang, before, after)], FLESH)
        return result

    def test_par_comment_removal_passes(self):
        before = '# header\n\nA::b = 1  # why\n#A::c = 2\nA::d = "x # y"\n'
        result = self.check("X/test/a.par", Lang.PAR, before, strip(before, Lang.PAR))
        self.assertTrue(result.ok, result.why)

    def test_par_value_change_fails(self):
        self.assertFalse(self.check("X/test/a.par", Lang.PAR, "A::b = 1\n", "A::b = 2\n").ok)
        self.assertFalse(self.check("X/test/a.par", Lang.PAR, 'A::d = "x # y"\n', 'A::d = "x "\n').ok)

    def test_par_string_comment_removal_passes(self):
        before = 'ActiveThorns = "\n    # SystemTopology\n    ADMBase\n"\nA::v = "x # y\n  a # b\n\n  # c\n\n  d\n"\n'
        after = strip(before, Lang.PAR)
        result = self.check("X/test/a.par", Lang.PAR, before, after)
        self.assertTrue(result.ok, result.why)
        self.assertIn("but for 3 string lines", result.how)

    def test_changing_a_string_beyond_its_comments_fails(self):
        for before, after in [
            ('A::v = "x # y\n  z\n"\n', 'A::v = "x\n  z\n"\n'),
            ('A::v = "\n  a\n  # c\n  b\n"\n', 'A::v = "\n  a\n  B\n"\n'),
            ('A::v = "\n  a\n\n  b\n"\n', 'A::v = "\n  a\n  b\n"\n'),
            ('A::v = "x $parfile"\n', 'A::v = "y $parfile"\n'),
        ]:
            with self.subTest(before=before):
                self.assertFalse(self.check("X/test/a.par", Lang.PAR, before, after).ok)

    def test_ccl_comment_removal_passes(self):
        for name, lang, before in [
            ("X/schedule.ccl", Lang.SCHEDULE,
             "# Schedule\n\nSTORAGE: a\n\n# SCHEDULE Old AT initial\nSCHEDULE X AT initial\n{\n  LANG: C /* c */\n"
             "  # why\n  WRITES: a(interior)\n} \"Init # kept\"\n"),
            ("X/param.ccl", Lang.PARAM, '# Parameters\n\nBOOLEAN b "b # kept" # why\n{\n} no\n'),
            ("X/interface.ccl", Lang.INTERFACE, "# Interface\n\nIMPLEMENTS: X # why\n"),
            ("X/configuration.ccl", Lang.CONFIGURATION, "# Configuration\n\nREQUIRES Loop # why\n"),
        ]:
            with self.subTest(name):
                after = strip(before, lang)
                self.assertNotIn("why", after)
                result = self.check(name, lang, before, after)
                self.assertTrue(result.ok, result.why)

    def test_a_description_change_fails(self):
        before = 'SCHEDULE X AT initial\n{\n  LANG: C\n} "Init # kept"\n'
        self.assertFalse(self.check("X/schedule.ccl", Lang.SCHEDULE, before, before.replace(" # kept", "")).ok)

    def test_an_unparseable_file_fails(self):
        result = self.check("X/param.ccl", Lang.PARAM, "BOOLEAN {\n", "BOOLEAN {\n")
        self.assertFalse(result.ok)
        self.assertIn("does not parse", result.why)

    def test_a_comment_only_file_stripped_to_nothing_passes(self):
        for name, lang in [("X/param.ccl", Lang.PARAM), ("X/schedule.ccl", Lang.SCHEDULE), ("X/test/a.par", Lang.PAR)]:
            with self.subTest(name):
                before = "# Parameter definitions for thorn X\n\n  # b\n"
                self.assertEqual(strip(before, lang), "")
                result = self.check(name, lang, before, "")
                self.assertTrue(result.ok, result.why)
                self.assertFalse(self.check(name, lang, '# a\nBOOLEAN b "b"\n{\n} no\n', "").ok)


def run_git(root, *args):
    env = {**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"}
    proc = subprocess.run(
        ["git", "-C", root, "-c", "user.name=t", "-c", "user.email=t@example.org", "-c", "commit.gpgsign=false",
         "-c", "core.hooksPath=/dev/null", *args],
        capture_output=True, text=True, check=True, env=env,
    )
    return proc.stdout.strip()


def commit(root, files, message):
    for name, text in files.items():
        path = os.path.join(root, name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", newline="") as f:
            f.write(text)
    run_git(root, "add", "-A")
    run_git(root, "commit", "-q", "-m", message)
    return run_git(root, "rev-parse", "HEAD")


@unittest.skipUnless(HAVE_CXX_TOOLS, "needs clang++ and unifdef on PATH")
class History(unittest.TestCase):
    SOURCE = "namespace X {\n// why\nint a; // c\n#if 0\nint b;\n#endif\n} // namespace X\n"

    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="comments-test-")
        self.addCleanup(shutil.rmtree, self.root)
        run_git(self.root, "init", "-q")
        self.base = commit(self.root, {"X/interface.ccl": "IMPLEMENTS: X\n", "X/src/a.cxx": self.SOURCE,
                                       "README": "a\n"}, "base")
        self.strip = commit(self.root, {"X/src/a.cxx": strip(self.SOURCE, Lang.CXX)}, "X: remove comments")

    def test_a_strip_commit_is_only_comments(self):
        results = verify_revs(self.root, self.base, self.strip, None)
        self.assertEqual([(r.name, r.ok) for r in results], [("X/src/a.cxx", True)])

    def test_an_added_file_or_an_uncovered_file_fails(self):
        bad = commit(self.root, {"X/src/b.cxx": "int b;\n", "README": "b\n"}, "more")
        results = verify_revs(self.root, self.strip, bad, None)
        self.assertEqual([(r.name, r.ok) for r in results], [("README", False), ("X/src/b.cxx", False)])

    def skips(self, base):
        lines = []
        return verify_skips(self.root, base, None, lines.append), "\n".join(lines)

    def test_verify_skips_proves_each_new_entry(self):
        commit(self.root, {".git-blame-ignore-revs": f"# X: remove comments\n{self.strip}\n"}, "ignore")
        ok, shown = self.skips(self.base)
        self.assertTrue(ok, shown)
        self.assertIn("X: remove comments", shown)
        ok, shown = self.skips("HEAD")
        self.assertTrue(ok, shown)
        self.assertIn("no new commits", shown)

    def test_verify_skips_refuses_a_code_change(self):
        code = commit(self.root, {"X/src/a.cxx": "namespace X {\nint a = 1;\n} // namespace X\n"}, "code")
        commit(self.root, {".git-blame-ignore-revs": f"{self.strip}\n{code}\n"}, "ignore")
        ok, shown = self.skips(self.base)
        self.assertFalse(ok)
        self.assertIn("1 of 2", shown)

    def test_verify_skips_refuses_a_commit_outside_the_branch_or_an_abbreviation(self):
        run_git(self.root, "checkout", "-q", "-b", "side", self.base)
        side = commit(self.root, {"README": "side\n"}, "side")
        run_git(self.root, "checkout", "-q", "-")
        for entry, why in [(side, "not in this branch's history"), (self.strip[:12], "not a full commit SHA")]:
            with self.subTest(why):
                commit(self.root, {".git-blame-ignore-revs": entry + "\n"}, "ignore " + why)
                ok, shown = self.skips(self.base)
                self.assertFalse(ok)
                self.assertIn(why, shown)

    def test_listed_skips_comments_and_repeats(self):
        self.assertEqual(listed("# a\n\nabc # b\nabc\n  def\n"), ["abc", "def"])


if __name__ == "__main__":
    unittest.main()
