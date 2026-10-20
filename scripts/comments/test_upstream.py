from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import unittest
from unittest import mock

import lex_clang
from files import Lang, list_tree, read_text
from rules import check_text
from strip import strip
from test_files import ISOLATED_GIT
from upstream import TRAILER, UpstreamError, newest_strip, stripped_from, strip_upstream, verify_skips

HAVE_CXX_TOOLS = shutil.which(lex_clang.CLANG) is not None and shutil.which("unifdef") is not None
IDENTITY = {
    "GIT_AUTHOR_NAME": "Test", "GIT_AUTHOR_EMAIL": "test@example.org",
    "GIT_COMMITTER_NAME": "Test", "GIT_COMMITTER_EMAIL": "test@example.org",
}

ET0 = (
    "namespace X {\n"
    "// Upstream's note on f.\n"
    "int f() { return 1; }\n"
    "int g() { return 2; } // two\n"
    "int h() { return 3; }\n"
    "\n"
    "int k() { return 5; }\n"
    "} // namespace X\n"
)
ET1 = (
    "namespace X {\n"
    "// Upstream's note on f, reworded.\n"
    "int f() { return 1; }\n"
    "int g() { return 2; } // the number two\n"
    "int h() { return 3; }\n"
    "\n"
    "int k() { return 50; } // fifty\n"
    "// A new helper.\n"
    "int m() { return 6; }\n"
    "} // namespace X\n"
)
MERGED = (
    "namespace X {\n"
    "int f() { return 1; }\n"
    "int g() { return 20; }\n"
    "int h() { return 3; }\n"
    "\n"
    "int k() { return 50; }\n"
    "int m() { return 6; }\n"
    "} // namespace X\n"
)


def git(root, *args, check=True):
    proc = subprocess.run(["git", "-C", root, *args], capture_output=True, text=True)
    if check and proc.returncode != 0:
        raise AssertionError(f"git {' '.join(args)} failed: {proc.stderr}")
    return proc.stdout.strip() if check else proc


@unittest.skipUnless(HAVE_CXX_TOOLS, "needs clang++ and unifdef on PATH")
class Lineage(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="comments-test-")
        self.addCleanup(shutil.rmtree, self.root)
        patcher = mock.patch.dict(os.environ, {**ISOLATED_GIT, **IDENTITY})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.clock = 1700000000
        git(self.root, "init", "-q", "-b", "main")
        git(self.root, "config", "commit.gpgsign", "false")
        git(self.root, "config", "core.hooksPath", os.devnull)
        self.et0 = self.commit({"X/interface.ccl": "IMPLEMENTS: X\n", "X/src/a.cxx": ET0, "README": "x\n"}, "ET0")
        git(self.root, "switch", "-q", "-c", "dev")
        self.dev_strip = self.commit({"X/src/a.cxx": strip(ET0, Lang.CXX)}, "X: remove comments")
        self.lines = []

    def tick(self):
        self.clock += 60
        stamp = f"{self.clock} +0000"
        os.environ["GIT_AUTHOR_DATE"] = os.environ["GIT_COMMITTER_DATE"] = stamp

    def commit(self, files, message):
        self.tick()
        for name, text in files.items():
            path = os.path.join(self.root, name)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", newline="") as f:
                f.write(text)
        git(self.root, "add", "-A")
        git(self.root, "commit", "-q", "-m", message)
        return git(self.root, "rev-parse", "HEAD")

    def upstream(self, files, message):
        git(self.root, "switch", "-q", "main")
        sha = self.commit(files, message)
        git(self.root, "switch", "-q", "dev")
        return sha

    def strip_upstream(self, ref, seed=False):
        self.tick()
        return strip_upstream(self.root, ref, seed, None, self.lines.append)

    def seed(self):
        s0 = self.strip_upstream(self.et0, seed=True)
        self.tick()
        git(self.root, "merge", "-q", "-s", "ours", "-m", "Seed", s0)
        return s0

    def parents(self, rev):
        return git(self.root, "log", "-1", "--format=%P", rev).split()

    def test_the_seed_strips_et0_onto_et0_alone(self):
        s0 = self.seed()
        self.assertEqual(self.parents(s0), [self.et0])
        self.assertEqual(stripped_from(self.root, s0), self.et0)
        self.assertEqual(git(self.root, "log", "-1", "--format=%s", s0), "Strip comments from EinsteinToolkit/CarpetX main "
                         + git(self.root, "rev-parse", "--short", self.et0))
        self.assertEqual(git(self.root, "show", f"{s0}:X/src/a.cxx") + "\n", strip(ET0, Lang.CXX))
        self.assertEqual(git(self.root, "rev-parse", f"{s0}^{{tree}}"), git(self.root, "rev-parse", "HEAD^{tree}"))
        self.assertEqual(newest_strip(self.root), s0)
        self.assertIn(f"✓ verify {self.et0[:12]} {s0[:12]}: only comments (1 file changed)", self.lines)

    def test_without_a_seed_there_is_no_lineage_to_strip_onto(self):
        with self.assertRaises(UpstreamError) as caught:
            self.strip_upstream(self.et0)
        self.assertIn("--seed", str(caught.exception))

    def test_an_upstream_that_has_not_moved_gives_back_the_last_strip_commit(self):
        s0 = self.seed()
        self.assertEqual(self.strip_upstream("main"), s0)
        self.assertEqual(git(self.root, "merge", "--no-ff", s0), "Already up to date.")

    def test_a_fork_edit_beside_an_upstream_comment_merges_cleanly(self):
        s0 = self.seed()
        fork = self.commit({"X/src/a.cxx": strip(ET0, Lang.CXX).replace("return 2;", "return 20;")}, "fork edit")
        et1 = self.upstream({"X/src/a.cxx": ET1}, "ET1")
        s1 = self.strip_upstream("main")
        self.assertEqual(self.parents(s1), [et1, s0])
        self.assertEqual(stripped_from(self.root, s1), et1)
        self.tick()
        merge = git(self.root, "merge", "--no-ff", "-m", "Merge upstream", s1, check=False)
        self.assertEqual(merge.returncode, 0, merge.stdout + merge.stderr)
        self.assertEqual(read_text(os.path.join(self.root, "X/src/a.cxx")), MERGED)
        self.assertEqual([(name, check_text(read_text(path), lang)) for name, path, lang in list_tree(self.root)],
                         [("X/interface.ccl", []), ("X/src/a.cxx", [])])
        self.assertEqual(newest_strip(self.root), s1)

        blamed = self.blame("X/src/a.cxx", [self.dev_strip, s0, s1])
        self.assertEqual(blamed, [
            ("namespace X {", self.et0),
            ("int f() { return 1; }", self.et0),
            ("int g() { return 20; }", fork),
            ("int h() { return 3; }", self.et0),
            ("", self.et0),
            ("int k() { return 50; }", et1),
            ("int m() { return 6; }", et1),
            ("} // namespace X", self.et0),
        ])

    def test_negative_control_a_strip_commit_without_the_lineage_conflicts(self):
        self.seed()
        self.commit({"X/src/a.cxx": strip(ET0, Lang.CXX).replace("return 2;", "return 20;")}, "fork edit")
        et1 = self.upstream({"X/src/a.cxx": ET1}, "ET1")
        tree = git(self.root, "rev-parse", f"{self.strip_upstream(et1, seed=True)}^{{tree}}")
        alone = git(self.root, "commit-tree", tree, "-p", et1, "-m", "strip of ET1 alone")
        merge = git(self.root, "merge", "--no-ff", "-m", "Merge upstream", alone, check=False)
        self.assertNotEqual(merge.returncode, 0)
        self.assertIn("CONFLICT", merge.stdout)
        git(self.root, "merge", "--abort")

    def blame(self, name, ignored):
        args = ["blame", "--line-porcelain"] + [f"--ignore-rev={rev}" for rev in ignored] + ["--", name]
        found = []
        sha = None
        for line in git(self.root, *args).split("\n"):
            if sha is None:
                sha = line.split(" ")[0]
            elif line.startswith("\t"):
                found.append((line[1:], sha))
                sha = None
        return found

    def test_verify_skips_proves_each_strip_commit_and_its_trailer(self):
        s0 = self.seed()
        self.upstream({"X/src/a.cxx": ET1}, "ET1")
        s1 = self.strip_upstream("main")
        self.tick()
        git(self.root, "merge", "-q", "--no-ff", "-m", "Merge upstream", s1)
        self.commit({".git-blame-ignore-revs": f"{self.dev_strip}\n{s0}\n{s1}\n"}, "ignore")
        lines = []
        ok = verify_skips(self.root, self.et0, None, lines.append)
        self.assertTrue(ok, "\n".join(lines))
        self.assertEqual(sum(f"is the stripped ET commit its {TRAILER}: trailer names" in line for line in lines), 2)

    def test_verify_skips_refuses_a_strip_commit_whose_first_parent_is_not_its_trailer(self):
        self.seed()
        et1 = self.upstream({"X/src/a.cxx": ET1}, "ET1")
        tree = git(self.root, "rev-parse", f"{self.strip_upstream(et1, seed=True)}^{{tree}}")
        wrong = git(self.root, "commit-tree", tree, "-p", self.et0, "-m", "Strip", "-m", f"{TRAILER}: {et1}")
        self.tick()
        git(self.root, "merge", "-q", "-s", "ours", "-m", "Record", wrong)
        self.commit({".git-blame-ignore-revs": f"{wrong}\n"}, "ignore")
        lines = []
        self.assertFalse(verify_skips(self.root, self.et0, None, lines.append))
        self.assertIn("is not the stripped ET commit", "\n".join(lines))


if __name__ == "__main__":
    unittest.main()
