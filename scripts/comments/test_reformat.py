from __future__ import annotations

import os
import shutil
import stat
import tempfile
import unittest
from unittest import mock

import lex_clang
import reformat
from files import PROJECT, Lang, read_text
from strip import strip
from test_verify import commit, run_git
from upstream import verify_skips
from verify import verify_revs


def pinned_on_path() -> bool:
    try:
        reformat.clang_format()
    except lex_clang.ToolMissing:
        return False
    return True


HAVE_TOOLS = shutil.which(lex_clang.CLANG) is not None and shutil.which("unifdef") is not None
HAVE_PINNED = HAVE_TOOLS and pinned_on_path()


class Version(unittest.TestCase):
    def test_only_llvm_22_1_1_passes(self):
        for output in ["clang-format version 22.1.1\n", "Ubuntu clang-format version 22.1.1 (1ubuntu1)\n"]:
            self.assertIsNone(reformat.version_problem(output), output)
        for output in ["clang-format version 22.1.0\n", "clang-format version 22.1.10\n",
                       "Ubuntu clang-format version 18.1.3 (1ubuntu1)\n",
                       "Apple clang-format version 21.0.0 (clang-2100.3.34.2)\n",
                       "Apple clang-format version 22.1.1\n", "", "usage: clang-format\n"]:
            self.assertIsNotNone(reformat.version_problem(output), output)

    def test_a_missing_tool_names_the_pinned_wheel(self):
        with mock.patch.object(reformat.shutil, "which", return_value=None):
            with self.assertRaises(lex_clang.ToolMissing) as caught:
                reformat.clang_format()
        self.assertIn("pip install clang-format==22.1.1", str(caught.exception))

    def test_another_version_is_refused_by_name(self):
        tmp = tempfile.mkdtemp(prefix="comments-test-")
        self.addCleanup(shutil.rmtree, tmp)
        fake = os.path.join(tmp, "clang-format")
        with open(fake, "w") as f:
            f.write("#!/bin/sh\necho 'Ubuntu clang-format version 18.1.3 (1ubuntu1)'\n")
        os.chmod(fake, os.stat(fake).st_mode | stat.S_IXUSR)
        with mock.patch.object(reformat.shutil, "which", return_value=fake):
            with self.assertRaises(lex_clang.ToolMissing) as caught:
                reformat.clang_format()
        message = str(caught.exception)
        self.assertIn("18.1.3", message)
        self.assertIn("pip install clang-format==22.1.1", message)


@unittest.skipUnless(HAVE_PINNED, "needs clang++, unifdef and clang-format 22.1.1 on PATH")
class Reformat(unittest.TestCase):
    DRAFTS = {
        "X/src/a.cxx": "void g(int a, int b);\nvoid f() {\n  g(1, // first\n    2);\n}\n",
        "X/src/c.cxx": "int y; // c\n",
    }
    UNFORMATTED = "int  x; // c\nvoid  h(int a, // first\n  int b);\n"

    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="comments-test-")
        self.addCleanup(shutil.rmtree, self.root)
        self.exe = reformat.clang_format()
        shutil.copy(os.path.join(PROJECT, ".clang-format"), os.path.join(self.root, ".clang-format"))
        run_git(self.root, "init", "-q")
        sources = {name: reformat.formatted(self.exe, self.root, name, text) for name, text in self.DRAFTS.items()}
        sources["X/src/b.cxx"] = self.UNFORMATTED
        self.assertIn("// first\n", sources["X/src/a.cxx"])
        self.assertNotEqual(reformat.formatted(self.exe, self.root, "X/src/b.cxx", self.UNFORMATTED), self.UNFORMATTED)
        self.base = commit(self.root, {"X/interface.ccl": "IMPLEMENTS: X\n", **sources}, "base")
        self.stripped = {name: strip(text, Lang.CXX) for name, text in sources.items()}
        self.strip = commit(self.root, self.stripped, "X: remove comments")

    def run_reformat(self, *thorns):
        lines = []
        return reformat.reformat(self.root, list(thorns), lines.append), lines

    def text(self, name):
        return read_text(os.path.join(self.root, name))

    def test_only_a_file_clean_before_the_strip_is_reformatted(self):
        [plan], lines = self.run_reformat("X")
        self.assertEqual((plan.reformat, plan.unformatted, plan.clean),
                         (["X/src/a.cxx"], ["X/src/b.cxx"], ["X/src/c.cxx"]))
        self.assertIn("g(1, 2);", self.text("X/src/a.cxx"))
        self.assertEqual(self.text("X/src/b.cxx"), self.stripped["X/src/b.cxx"])
        self.assertEqual(self.text("X/src/c.cxx"), self.stripped["X/src/c.cxx"])
        self.assertEqual(lines[0], "X/src/a.cxx")
        self.assertIn("reformatted 1 of the 3", lines[-1])

    def test_a_second_run_changes_nothing(self):
        self.run_reformat("X")
        [plan], _ = self.run_reformat("X")
        self.assertEqual(plan.reformat, [])

    def test_the_reformat_commit_is_proven_by_layout_not_by_comments(self):
        self.run_reformat("X")
        layout = commit(self.root, {}, "X: clang-format after removing comments")
        self.assertTrue(all(r.ok for r in verify_revs(self.root, self.strip, layout, None, layout=True)))
        self.assertFalse(all(r.ok for r in verify_revs(self.root, self.strip, layout, None)))
        commit(self.root, {".git-blame-ignore-revs": f"{self.strip}\n{layout}\n"}, "ignore")
        lines = []
        self.assertTrue(verify_skips(self.root, self.base, None, lines.append), "\n".join(lines))

    def test_a_thorn_is_named_from_the_repository_or_by_its_path(self):
        elsewhere = tempfile.mkdtemp(prefix="comments-test-")
        self.addCleanup(shutil.rmtree, elsewhere)
        os.makedirs(os.path.join(elsewhere, "X"))
        cwd = os.getcwd()
        self.addCleanup(os.chdir, cwd)
        os.chdir(elsewhere)
        for arg in ["X", "X/", os.path.join(self.root, "X")]:
            with self.subTest(arg):
                [plan], _ = self.run_reformat(arg)
                self.assertEqual(plan.thorn, "X")

    def test_a_thorn_without_a_strip_commit_or_a_non_thorn_is_refused(self):
        os.makedirs(os.path.join(self.root, "Y"))
        with open(os.path.join(self.root, "Y", "interface.ccl"), "w") as f:
            f.write("IMPLEMENTS: Y\n")
        for arg, why in [("Y", "no commit with the subject"), ("X/src", "is not a thorn"), ("Z", "is not a thorn")]:
            with self.subTest(arg):
                with self.assertRaises(reformat.ReformatError) as caught:
                    self.run_reformat(arg)
                self.assertIn(why, str(caught.exception))


if __name__ == "__main__":
    unittest.main()
