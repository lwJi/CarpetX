import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import hooks

TOOL = os.path.dirname(os.path.abspath(__file__))
FILES = {
    "Thorn/src/a.cxx": "namespace Thorn {\nint a = 1;\n} // namespace Thorn\n",
    "Thorn/param.ccl": "BOOLEAN b \"b\"\n{\n} no\n",
    "Thorn/README": "not covered\n",
    ".gitignore": "scratch/\n",
}


def isolated_git_env(tmp: str) -> dict[str, str]:
    config = os.path.join(tmp, "gitconfig")
    open(config, "w").close()
    return {
        "GIT_CONFIG_GLOBAL": config,
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@t",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@t",
    }


class Project(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.repo = os.path.join(self.tmp.name, "repo")
        os.makedirs(self.repo)
        patcher = mock.patch.dict(os.environ, isolated_git_env(self.tmp.name))
        patcher.start()
        self.addCleanup(patcher.stop)
        self.write(FILES)
        subprocess.run(["git", "init", "-q"], cwd=self.repo, check=True)
        subprocess.run(["git", "add", "-A"], cwd=self.repo, check=True)

    def path(self, name):
        return os.path.join(self.repo, name)

    def write(self, files):
        for name, text in files.items():
            os.makedirs(os.path.dirname(self.path(name)), exist_ok=True)
            with open(self.path(name), "w") as f:
                f.write(text)

    def run_hook(self, hook, payload):
        out = io.StringIO()
        raw = payload if isinstance(payload, str) else json.dumps(payload)
        self.assertEqual(hook(raw, out, root=self.repo), 0)
        return out.getvalue()

    def decision(self, output):
        self.assertTrue(output, "the hook printed nothing")
        return json.loads(output)


class HookEdit(Project):
    def edit(self, name, old, new, replace_all=False, path=None):
        tool_input = {"file_path": path or self.path(name), "old_string": old, "new_string": new}
        if replace_all:
            tool_input["replace_all"] = True
        return self.run_hook(hooks.hook_edit, {"tool_name": "Edit", "tool_input": tool_input, "cwd": self.repo})

    def write_tool(self, name, content):
        payload = {"tool_name": "Write", "tool_input": {"file_path": self.path(name), "content": content}}
        return self.run_hook(hooks.hook_edit, payload)

    def assert_denied(self, output, *lines):
        out = self.decision(output)["hookSpecificOutput"]
        self.assertEqual(out["hookEventName"], "PreToolUse")
        self.assertEqual(out["permissionDecision"], "deny")
        for line in lines:
            self.assertIn(line, out["permissionDecisionReason"].split("\n"))
        return out["permissionDecisionReason"]

    def test_denies_an_added_comment(self):
        reason = self.assert_denied(
            self.edit("Thorn/src/a.cxx", "int a = 1;\n", "int a = 1; // why\n"),
            "  Thorn/src/a.cxx:2   // why",
        )
        self.assertIn("Put the explanation in", reason)
        self.assertTrue(reason.endswith("✗ no-comments: 1 comment outside the allowed exceptions"))

    def test_allows_a_namespace_closer(self):
        self.assertEqual(
            self.edit("Thorn/src/a.cxx", "} // namespace Thorn\n", "} // namespace Thorn\nnamespace B {\n} // namespace B\n"),
            "",
        )

    def test_denies_a_write_with_a_comment(self):
        self.assert_denied(self.write_tool("Thorn/test/t.par", "# toggle\nThorn::b = yes\n"), "  Thorn/test/t.par:1   # toggle")
        self.assert_denied(self.write_tool("Thorn/schedule.ccl", "SCHEDULE f AT init\n{\n} \"f\" # why\n"))

    def test_allows_a_write_without_comments(self):
        self.assertEqual(self.write_tool("Thorn/src/b.cxx", "namespace Thorn {\nint b;\n} // namespace Thorn\n"), "")

    def test_an_edit_that_creates_a_file_checks_new_string(self):
        self.assert_denied(self.edit("Thorn/src/new.cxx", "", "int n; /* why */\n"), "  Thorn/src/new.cxx:1   /* why */")

    def test_prints_nothing_for_an_uncovered_path(self):
        self.assertEqual(self.edit("Thorn/README", "not covered\n", "// still not covered\n"), "")
        self.assertEqual(self.write_tool("scripts/x.py", "# a python comment\n"), "")

    def test_prints_nothing_for_an_ignored_path(self):
        self.assertEqual(self.write_tool("scratch/src/x.cxx", "// why\n"), "")

    def test_prints_nothing_outside_the_project(self):
        with tempfile.TemporaryDirectory() as other:
            path = os.path.join(other, "Thorn", "src", "a.cxx")
            self.assertEqual(self.edit(None, "", "// why\n", path=path), "")

    def test_no_decision_when_old_string_is_missing_from_the_file(self):
        self.assertEqual(self.edit("Thorn/src/a.cxx", "int nope;\n", "int nope; // why\n"), "")

    def test_no_decision_when_old_string_is_not_unique_unless_replace_all(self):
        self.write({"Thorn/src/b.cxx": "int x;\nint x;\n"})
        self.assertEqual(self.edit("Thorn/src/b.cxx", "int x;", "int x; // why"), "")
        self.assert_denied(
            self.edit("Thorn/src/b.cxx", "int x;", "int x; // why", replace_all=True),
            "  Thorn/src/b.cxx:1   // why",
            "  Thorn/src/b.cxx:2   // why",
        )

    def test_comments_already_in_the_file_are_not_the_edit_s(self):
        self.write({"Thorn/src/b.cxx": "// one\nint x;\n// two\nint y;\n"})
        self.assertEqual(self.edit("Thorn/src/b.cxx", "// two\n", ""), "")
        self.write({"Thorn/src/b.cxx": "// one\nint x;\nint y;\n"})
        self.assertEqual(self.edit("Thorn/src/b.cxx", "// one\nint x;\n", "{\n  // one\n  int x;\n}\n"), "")
        self.write({"Thorn/src/b.cxx": "{\n  // one\n  int x;\n}\nint y;\n"})
        reason = self.assert_denied(self.edit("Thorn/src/b.cxx", "int y;\n", "int y; // three\n"), "  Thorn/src/b.cxx:5   // three")
        self.assertNotIn("// one", reason)

    def test_an_internal_error_becomes_a_system_message(self):
        for payload in [
            "not json",
            {"tool_name": "Edit", "tool_input": {"file_path": self.path("Thorn/src/a.cxx"), "new_string": "x"}},
            {"tool_name": "Teleport", "tool_input": {"file_path": self.path("Thorn/src/a.cxx")}},
        ]:
            out = self.decision(self.run_hook(hooks.hook_edit, payload))
            self.assertEqual(list(out), ["systemMessage"], payload)
            self.assertTrue(out["systemMessage"].startswith("comment check skipped: "))
        with mock.patch("rules.check", side_effect=RuntimeError("lexer crash")):
            out = self.decision(self.edit("Thorn/src/a.cxx", "int a = 1;\n", "int a = 2;\n"))
        self.assertEqual(out, {"systemMessage": "comment check skipped: RuntimeError: lexer crash"})


class HookStop(Project):
    def stop(self, **payload):
        return self.run_hook(hooks.hook_stop, {"hook_event_name": "Stop", **payload})

    def assert_blocked(self, output, *lines):
        out = self.decision(output)
        self.assertEqual(out["decision"], "block")
        for line in lines:
            self.assertIn(line, out["reason"].split("\n"))
        return out["reason"]

    def test_a_clean_tree_prints_nothing(self):
        self.assertEqual(self.stop(stop_hook_active=False), "")

    def test_catches_an_untracked_file(self):
        self.write({"Thorn/src/new.cxx": "int n;\n// why\n"})
        self.assert_blocked(self.stop(), "  Thorn/src/new.cxx:2   // why")

    def test_catches_a_tracked_file(self):
        self.write({"Thorn/param.ccl": "# why\nBOOLEAN b \"b\"\n{\n} no\n"})
        reason = self.assert_blocked(self.stop(), "  Thorn/param.ccl:1   # why")
        self.assertTrue(reason.endswith("✗ no-comments: 1 comment outside the allowed exceptions"))

    def test_ignores_ignored_and_uncovered_files(self):
        self.write({"scratch/src/x.cxx": "// why\n", "Thorn/README": "// why\n"})
        self.assertEqual(self.stop(), "")

    def test_stop_hook_active_skips_the_check(self):
        self.write({"Thorn/src/new.cxx": "// why\n"})
        self.assertEqual(self.stop(stop_hook_active=True), "")

    def test_an_internal_error_becomes_a_system_message(self):
        out = self.decision(self.run_hook(hooks.hook_stop, "[]"))
        self.assertEqual(list(out), ["systemMessage"])
        with mock.patch("rules.check", side_effect=RuntimeError("lexer crash")):
            out = self.decision(self.stop())
        self.assertEqual(out, {"systemMessage": "comment check skipped: RuntimeError: lexer crash"})


class HookCli(unittest.TestCase):
    def run_tool(self, command, stdin):
        return subprocess.run([sys.executable, TOOL, command], input=stdin, capture_output=True, text=True)

    def test_every_outcome_exits_0(self):
        for command in ["hook-edit", "hook-stop"]:
            result = self.run_tool(command, "not json")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("systemMessage", json.loads(result.stdout))

    def test_an_uncovered_path_prints_nothing(self):
        payload = {"tool_name": "Write", "tool_input": {"file_path": os.path.join(TOOL, "x.py"), "content": "# x\n"}}
        result = self.run_tool("hook-edit", json.dumps(payload))
        self.assertEqual((result.returncode, result.stdout, result.stderr), (0, "", ""))


if __name__ == "__main__":
    unittest.main()
