import os
import subprocess
import sys
import tempfile
import unittest

from test_verify import FLESH, HAVE_CXX_TOOLS, HAVE_FLESH

TOOL = os.path.dirname(os.path.abspath(__file__))
FILES = {
    "Thorn/src/a.cxx": "namespace Thorn {\n// why\nint a = 1; // what\n} // namespace Thorn\n",
    "Thorn/param.ccl": "# Parameter definitions for thorn Thorn\n\nBOOLEAN b \"b\"\n{\n} no\n",
    "Thorn/test/t.par": "ActiveThorns = \"Thorn\"\n\n# toggle\nThorn::b = yes\n",
    "Thorn/README": "# not covered\n",
}


class Cli(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.repo = self.tmp.name
        for name, text in FILES.items():
            path = os.path.join(self.repo, name)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                f.write(text)
        self.git("init", "-q")
        self.commit("start")

    def tearDown(self):
        self.tmp.cleanup()

    def git(self, *args):
        return subprocess.run(
            ["git", "-c", "user.name=t", "-c", "user.email=t@t", *args],
            cwd=self.repo, capture_output=True, text=True, check=True,
        ).stdout

    def commit(self, message):
        self.git("add", "-A")
        self.git("commit", "-q", "-m", message)

    def run_tool(self, *args):
        env = dict(os.environ)
        if FLESH:
            env["CACTUSX"] = FLESH
        return subprocess.run([sys.executable, TOOL, *args], cwd=self.repo, capture_output=True, text=True, env=env)

    def read(self, name):
        with open(os.path.join(self.repo, name)) as f:
            return f.read()

    def test_check_strip_check(self):
        before = self.run_tool("check")
        self.assertEqual(before.returncode, 1, before.stderr)
        self.assertIn("Thorn/src/a.cxx:2   // why", before.stdout)
        self.assertEqual(before.stdout.splitlines()[-1], "✗ no-comments: 4 comments outside the allowed exceptions")

        stripped = self.run_tool("strip", "Thorn")
        self.assertEqual(stripped.returncode, 0, stripped.stderr)
        self.assertEqual(self.read("Thorn/src/a.cxx"), "namespace Thorn {\nint a = 1;\n} // namespace Thorn\n")
        self.assertEqual(self.read("Thorn/README"), FILES["Thorn/README"])

        after = self.run_tool("check", "Thorn")
        self.assertEqual(after.returncode, 0, after.stdout)

    def test_check_takes_files(self):
        result = self.run_tool("check", "Thorn/test/t.par")
        self.assertEqual(result.returncode, 1)
        self.assertIn("Thorn/test/t.par:3   # toggle", result.stdout)

    def test_missing_path(self):
        self.assertEqual(self.run_tool("check", "Nope").returncode, 2)

    def test_an_internal_error_is_not_a_finding(self):
        self.assertEqual(self.run_tool("verify", "no-such-rev", "HEAD").returncode, 2)

    def test_strip_refuses_an_elif_chain_and_leaves_the_file(self):
        text = "#if 0\nx;\n#elif defined A\ny;\n#endif\n"
        with open(os.path.join(self.repo, "Thorn/src/b.cxx"), "w") as f:
            f.write(text)
        result = self.run_tool("strip", "Thorn/src/b.cxx")
        self.assertEqual(result.returncode, 1)
        self.assertIn("Thorn/src/b.cxx:1: strip refused", result.stderr)
        self.assertEqual(self.read("Thorn/src/b.cxx"), text)

    @unittest.skipUnless(HAVE_CXX_TOOLS and HAVE_FLESH, "needs clang++, unifdef, perl and $CACTUSX")
    def test_verify_a_strip_commit_and_a_code_change(self):
        self.run_tool("strip", "Thorn")
        self.commit("Thorn: remove comments")
        result = self.run_tool("verify", "HEAD~1", "HEAD")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("3 files changed, only comments", result.stdout)

        with open(os.path.join(self.repo, "Thorn/src/a.cxx"), "a") as f:
            f.write("int b = 2;\n")
        self.commit("Thorn: add b")
        result = self.run_tool("verify", "HEAD~1", "HEAD")
        self.assertEqual(result.returncode, 1)
        self.assertIn("FAIL Thorn/src/a.cxx", result.stdout)

    @unittest.skipUnless(HAVE_CXX_TOOLS and HAVE_FLESH, "needs clang++, unifdef, perl and $CACTUSX")
    def test_a_comment_only_file_is_kept_empty_and_verifies(self):
        with open(os.path.join(self.repo, "Thorn/configuration.ccl"), "w") as f:
            f.write("# Configuration definitions for thorn Thorn\n")
        self.commit("configuration.ccl")
        self.assertEqual(self.run_tool("strip", "Thorn").returncode, 0)
        self.assertEqual(self.read("Thorn/configuration.ccl"), "")
        self.commit("Thorn: remove comments")
        result = self.run_tool("verify", "HEAD~1", "HEAD")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("ok   Thorn/configuration.ccl", result.stdout)

    @unittest.skipUnless(HAVE_CXX_TOOLS and HAVE_FLESH, "needs clang++, unifdef, perl and $CACTUSX")
    def test_verify_rejects_uncovered_changes(self):
        with open(os.path.join(self.repo, "Thorn/README"), "w") as f:
            f.write("changed\n")
        self.commit("README")
        result = self.run_tool("verify", "HEAD~1", "HEAD")
        self.assertEqual(result.returncode, 1)
        self.assertIn("FAIL Thorn/README", result.stdout)


if __name__ == "__main__":
    unittest.main()
