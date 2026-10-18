import os
import subprocess
import sys
import tempfile
import unittest

TOOL = os.path.dirname(os.path.abspath(__file__))
FILLER = "int x1;\nint x2;\nint x3;\nint x4;\n"


class Remerge(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.repo = os.path.join(self.tmp.name, "repo")
        os.makedirs(self.repo)
        config = os.path.join(self.tmp.name, "gitconfig")
        open(config, "w").close()
        self.env = dict(os.environ)
        self.env.update(
            GIT_CONFIG_GLOBAL=config,
            GIT_CONFIG_NOSYSTEM="1",
            GIT_AUTHOR_NAME="t",
            GIT_AUTHOR_EMAIL="t@t",
            GIT_COMMITTER_NAME="t",
            GIT_COMMITTER_EMAIL="t@t",
        )
        self.git("init", "-q", "-b", "dev")

    def tearDown(self):
        self.tmp.cleanup()

    def git(self, *args, check=True):
        return subprocess.run(["git", *args], cwd=self.repo, env=self.env, capture_output=True, text=True, check=check)

    def write(self, files):
        for name, text in files.items():
            path = os.path.join(self.repo, name)
            if text is None:
                os.remove(path)
                continue
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                f.write(text)

    def commit(self, files, message):
        self.write(files)
        self.git("add", "-A")
        self.git("commit", "-q", "--allow-empty", "-m", message)

    def merge(self, base, ours, theirs):
        self.commit(base, "base")
        self.git("switch", "-q", "-c", "strip-upstream")
        self.commit(theirs, "strip upstream")
        self.git("switch", "-q", "dev")
        self.commit(ours, "fork")
        return self.git("merge", "-q", "strip-upstream", check=False)

    def unmerged(self):
        return self.git("diff", "--name-only", "--diff-filter=U").stdout.split()

    def read(self, name):
        with open(os.path.join(self.repo, name)) as f:
            return f.read()

    def run_tool(self):
        return subprocess.run(
            [sys.executable, TOOL, "remerge"], cwd=self.repo, env=self.env, capture_output=True, text=True
        )

    def test_a_fork_change_under_an_upstream_comment_merges_clean(self):
        merge = self.merge(
            base={
                "Thorn/src/a.cxx": "namespace Thorn {\n// why c\nint c = 1;\n} // namespace Thorn\n",
                "Thorn/test/t.par": "# toggle\nThorn::b = no\n",
            },
            ours={
                "Thorn/src/a.cxx": "namespace Thorn {\nint c = 2;\n} // namespace Thorn\n",
                "Thorn/test/t.par": "Thorn::b = yes\n",
            },
            theirs={
                "Thorn/src/a.cxx": "namespace Thorn {\nint c = 1;\n} // namespace Thorn\n",
                "Thorn/test/t.par": "Thorn::b = no\n",
            },
        )
        self.assertNotEqual(merge.returncode, 0)
        self.assertEqual(self.unmerged(), ["Thorn/src/a.cxx", "Thorn/test/t.par"])

        result = self.run_tool()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("merged    Thorn/src/a.cxx", result.stdout)
        self.assertIn("remerge: 2 of 2 unmerged paths merged, 0 left to resolve by hand", result.stdout)
        self.assertEqual(self.unmerged(), [])
        self.assertEqual(self.read("Thorn/src/a.cxx"), "namespace Thorn {\nint c = 2;\n} // namespace Thorn\n")
        self.assertEqual(self.read("Thorn/test/t.par"), "Thorn::b = yes\n")
        self.git("commit", "-q", "--no-edit")
        self.assertEqual(self.git("diff", "HEAD~1", "HEAD").stdout, "")

    def test_a_real_conflict_keeps_its_markers_and_loses_the_comment_one(self):
        self.merge(
            base={"Thorn/src/a.cxx": "// why a\nint a = 1;\n" + FILLER + "// why b\nint b = 1;\n"},
            ours={"Thorn/src/a.cxx": "int a = 2;\n" + FILLER + "int b = 2;\n"},
            theirs={"Thorn/src/a.cxx": "int a = 1;\n" + FILLER + "int b = 3;\n"},
        )
        self.assertEqual(self.read("Thorn/src/a.cxx").count("<<<<<<<"), 2)

        result = self.run_tool()
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("conflict  Thorn/src/a.cxx  1 conflict left", result.stdout)
        self.assertEqual(self.unmerged(), ["Thorn/src/a.cxx"])
        self.assertEqual(
            self.read("Thorn/src/a.cxx"),
            "int a = 2;\n" + FILLER + "<<<<<<< HEAD\nint b = 2;\n=======\nint b = 3;\n>>>>>>> MERGE_HEAD\n",
        )

    def test_the_conflict_style_follows_git_config(self):
        self.git("config", "merge.conflictStyle", "diff3")
        self.merge(
            base={"Thorn/src/a.cxx": "// why\nint b = 1;\n"},
            ours={"Thorn/src/a.cxx": "int b = 2;\n"},
            theirs={"Thorn/src/a.cxx": "int b = 3;\n"},
        )
        self.run_tool()
        self.assertIn("||||||| merge base, stripped\nint b = 1;\n=======\n", self.read("Thorn/src/a.cxx"))

    def test_paths_that_are_not_covered_are_left_as_git_left_them(self):
        self.merge(
            base={"Thorn/README": "# why\nold\n", "Thorn/src/a.cxx": "// why\nint c = 1;\n"},
            ours={"Thorn/README": "ours\n", "Thorn/src/a.cxx": "int c = 2;\n"},
            theirs={"Thorn/README": "old\n", "Thorn/src/a.cxx": "int c = 1;\n"},
        )
        conflicted = self.read("Thorn/README")

        result = self.run_tool()
        self.assertEqual(result.returncode, 1)
        self.assertIn("skipped   Thorn/README     not a covered file", result.stdout)
        self.assertIn("remerge: 1 of 2 unmerged paths merged, 1 left to resolve by hand", result.stdout)
        self.assertEqual(self.read("Thorn/README"), conflicted)
        self.assertEqual(self.unmerged(), ["Thorn/README"])

    def test_covered_paths_without_all_three_versions_are_left_alone(self):
        self.merge(
            base={"Thorn/src/gone.cxx": "// why\nint g = 1;\n"},
            ours={"Thorn/src/new.cxx": "int n = 1;\n", "Thorn/src/gone.cxx": None},
            theirs={"Thorn/src/new.cxx": "int n = 2;\n", "Thorn/src/gone.cxx": "int g = 2;\n"},
        )
        new, gone = self.read("Thorn/src/new.cxx"), self.read("Thorn/src/gone.cxx")

        result = self.run_tool()
        self.assertEqual(result.returncode, 1)
        self.assertIn("Thorn/src/new.cxx   no base version", result.stdout)
        self.assertIn("Thorn/src/gone.cxx  no ours version", result.stdout)
        self.assertEqual(self.read("Thorn/src/new.cxx"), new)
        self.assertEqual(self.read("Thorn/src/gone.cxx"), gone)
        self.assertEqual(self.unmerged(), ["Thorn/src/gone.cxx", "Thorn/src/new.cxx"])

    def test_paths_git_merged_are_not_touched(self):
        self.merge(
            base={"Thorn/src/a.cxx": "// why\nint c = 1;\n", "Thorn/src/b.cxx": "int b = 1;\n"},
            ours={"Thorn/src/a.cxx": "int c = 2;\n"},
            theirs={"Thorn/src/a.cxx": "int c = 1;\n", "Thorn/src/b.cxx": "// upstream\nint b = 1;\n"},
        )
        staged = self.git("ls-files", "-s", "Thorn/src/b.cxx").stdout

        result = self.run_tool()
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertNotIn("b.cxx", result.stdout)
        self.assertEqual(self.read("Thorn/src/b.cxx"), "// upstream\nint b = 1;\n")
        self.assertEqual(self.git("ls-files", "-s", "Thorn/src/b.cxx").stdout, staged)

    def test_a_conflict_resolved_by_hand_is_left_alone(self):
        self.merge(
            base={"Thorn/src/a.cxx": "// why\nint c = 1;\n"},
            ours={"Thorn/src/a.cxx": "int c = 2;\n"},
            theirs={"Thorn/src/a.cxx": "int c = 1;\n"},
        )
        self.write({"Thorn/src/a.cxx": "int c = 4;\n"})

        result = self.run_tool()
        self.assertEqual(result.returncode, 1)
        self.assertIn("no conflict markers in the working tree file", result.stdout)
        self.assertEqual(self.read("Thorn/src/a.cxx"), "int c = 4;\n")
        self.assertEqual(self.unmerged(), ["Thorn/src/a.cxx"])

    def test_a_version_strip_refuses_leaves_the_conflict(self):
        guard = "#if 0\nx;\n#elif defined A\ny;\n#endif\n"
        self.merge(
            base={"Thorn/src/a.cxx": "// why\nint c = 1;\n" + guard},
            ours={"Thorn/src/a.cxx": "int c = 2;\n" + guard},
            theirs={"Thorn/src/a.cxx": "int c = 1;\n" + guard},
        )
        conflicted = self.read("Thorn/src/a.cxx")

        result = self.run_tool()
        self.assertEqual(result.returncode, 1)
        self.assertIn("strip refused the base version at line 3", result.stdout)
        self.assertEqual(self.read("Thorn/src/a.cxx"), conflicted)

    def test_nothing_to_do_outside_a_conflict(self):
        self.commit({"Thorn/src/a.cxx": "// why\nint c = 1;\n"}, "base")
        result = self.run_tool()
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "remerge: no unmerged paths\n")
        self.assertEqual(self.read("Thorn/src/a.cxx"), "// why\nint c = 1;\n")


if __name__ == "__main__":
    unittest.main()
