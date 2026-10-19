from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import hooks
import lex_clang
from files import PROJECT, Lang
from hooks import added, simulate_edit

HAVE_CLANG = shutil.which(lex_clang.CLANG) is not None
TOOL = os.path.dirname(os.path.abspath(__file__))
SETTINGS = os.path.join(PROJECT, ".claude", "settings.json")
FILES = {
    "Thorn/interface.ccl": "IMPLEMENTS: Thorn\n",
    "Thorn/src/a.cxx": "namespace Thorn {\nint a = 1;\n} // namespace Thorn\n",
    "Thorn/README": "not covered\n",
    "Other/src/b.cxx": "int b;\n",
}


def write_files(root, files):
    for name, text in files.items():
        path = os.path.join(root, name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(text)


class SimulateEdit(unittest.TestCase):
    def edit(self, before, old, new, replace_all=False):
        tool_input = {"file_path": "/x", "old_string": old, "new_string": new}
        if replace_all:
            tool_input["replace_all"] = True
        return simulate_edit("Edit", tool_input, before)

    def test_write_takes_the_content(self):
        self.assertEqual(simulate_edit("Write", {"file_path": "/x", "content": "c"}, "old"), "c")

    def test_edit_replaces_one_occurrence(self):
        self.assertEqual(self.edit("a b c", "b", "B"), "a B c")

    def test_edit_that_does_not_match_is_not_judged(self):
        self.assertIsNone(self.edit("a b c", "x", "X"))
        self.assertIsNone(self.edit(None, "x", "X"))

    def test_edit_that_is_not_unique_needs_replace_all(self):
        self.assertIsNone(self.edit("a a", "a", "b"))
        self.assertEqual(self.edit("a a", "a", "b", replace_all=True), "b b")

    def test_edit_with_an_empty_old_string_creates_a_file(self):
        self.assertEqual(self.edit(None, "", "new"), "new")
        self.assertEqual(self.edit("", "", "new"), "new")
        self.assertIsNone(self.edit("exists", "", "new"))

    def test_other_tools_are_errors(self):
        with self.assertRaises(ValueError):
            simulate_edit("NotebookEdit", {}, None)


@unittest.skipUnless(HAVE_CLANG, "needs clang++ on PATH")
class Added(unittest.TestCase):
    def texts(self, before, after):
        return [(v.line, v.token.text) for v in added(before, after, Lang.CXX)]

    def test_a_new_comment_is_added(self):
        self.assertEqual(self.texts("int a;\n", "int a; // why\n"), [(1, "// why")])

    def test_comments_already_in_the_file_are_not_the_edit_s(self):
        self.assertEqual(self.texts("// one\nint x;\n// two\nint y;\n", "// one\nint x;\nint y;\n"), [])
        self.assertEqual(self.texts("// one\nint x;\n", "{\n  // one\n  int x;\n}\n"), [])
        self.assertEqual(self.texts("// a\nint x;\n// b\n", "// a\n// b\n"), [])
        self.assertEqual(self.texts("// a\n// b\n", "// a\nint y;\n// b\n"), [])

    def test_only_the_new_comment_is_shown(self):
        self.assertEqual(self.texts("// one\n// two\nint x;\n", "// one\n// two\n// three\nint x;\n"), [(3, "// three")])

    def test_a_duplicate_of_an_existing_comment_is_added(self):
        self.assertEqual(self.texts("int a; // x\n", "int a; // x\nint b; // x\n"), [(2, "// x")])

    def test_an_allowed_comment_does_not_hide_a_new_violation(self):
        before = "namespace X {\n} // namespace X\n"
        after = "// namespace X\nnamespace X {\n} // namespace X\n"
        self.assertEqual(self.texts(before, after), [(1, "// namespace X")])

    def test_an_allowed_comment_is_not_added(self):
        self.assertEqual(self.texts("int a;\n", "namespace X {\nint a;\n} // namespace X\n"), [])


class Project(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = os.path.realpath(tmp.name)
        write_files(self.root, FILES)

    def path(self, name):
        return os.path.join(self.root, name)

    def run_hook(self, payload):
        out = io.StringIO()
        raw = payload if isinstance(payload, str) else json.dumps(payload)
        self.assertEqual(hooks.hook_edit(raw, out, root=self.root), 0)
        return out.getvalue()

    def edit(self, name, old, new, replace_all=False, path=None):
        tool_input = {"file_path": path or self.path(name), "old_string": old, "new_string": new}
        if replace_all:
            tool_input["replace_all"] = True
        return self.run_hook({"tool_name": "Edit", "tool_input": tool_input, "cwd": self.root})

    def write(self, name, content):
        return self.run_hook({"tool_name": "Write", "tool_input": {"file_path": self.path(name), "content": content}})

    def denied(self, output):
        self.assertTrue(output, "the hook printed nothing")
        decision = json.loads(output)["hookSpecificOutput"]
        self.assertEqual(decision["hookEventName"], "PreToolUse")
        self.assertEqual(decision["permissionDecision"], "deny")
        return decision["permissionDecisionReason"].split("\n")

    def skipped(self, output):
        message = json.loads(output)
        self.assertEqual(list(message), ["systemMessage"])
        self.assertTrue(message["systemMessage"].startswith("comment check skipped: "), message)
        return message["systemMessage"]


@unittest.skipUnless(HAVE_CLANG, "needs clang++ on PATH")
class HookEdit(Project):
    def test_denies_an_added_comment(self):
        lines = self.denied(self.edit("Thorn/src/a.cxx", "int a = 1;\n", "int a = 1; // why\n"))
        self.assertEqual(lines[0], "Thorn/src/a.cxx:2   // why")
        self.assertEqual(lines[-1], "✗ no-comments: 1 comment")

    def test_allows_an_edit_without_a_new_comment(self):
        self.assertEqual(self.edit("Thorn/src/a.cxx", "int a = 1;\n", "int a = 2;\n"), "")
        self.assertEqual(
            self.edit("Thorn/src/a.cxx", "} // namespace Thorn\n", "} // namespace Thorn\nnamespace B {\n} // namespace B\n"),
            "",
        )

    def test_write(self):
        self.denied(self.write("Thorn/src/new.cxx", "/* why */\nint n;\n"))
        self.assertEqual(self.write("Thorn/src/b.cxx", "namespace Thorn {\nint b;\n} // namespace Thorn\n"), "")

    def test_an_edit_that_creates_a_file_checks_new_string(self):
        lines = self.denied(self.edit("Thorn/src/new.cxx", "", "int n; /* why */\n"))
        self.assertEqual(lines[0], "Thorn/src/new.cxx:1   /* why */")

    def test_a_relative_path_is_taken_from_cwd(self):
        self.denied(self.edit(None, "int a = 1;\n", "int a = 1; // why\n", path="Thorn/src/a.cxx"))

    def test_replace_all_reports_every_copy(self):
        write_files(self.root, {"Thorn/src/b.cxx": "int x;\nint x;\n"})
        self.assertEqual(self.edit("Thorn/src/b.cxx", "int x;", "int x; // why"), "")
        lines = self.denied(self.edit("Thorn/src/b.cxx", "int x;", "int x; // why", replace_all=True))
        self.assertEqual(lines[:2], ["Thorn/src/b.cxx:1   // why", "Thorn/src/b.cxx:2   // why"])

    def test_an_old_string_that_does_not_match_is_left_to_the_edit_tool(self):
        self.assertEqual(self.edit("Thorn/src/a.cxx", "int nope;\n", "int nope; // why\n"), "")

    def test_a_tool_failure_is_a_system_message(self):
        with mock.patch("hooks.lex", side_effect=RuntimeError("lexer crash")):
            output = self.edit("Thorn/src/a.cxx", "int a = 1;\n", "int a = 1; // why\n")
        self.assertEqual(self.skipped(output), "comment check skipped: RuntimeError: lexer crash")


class HookEditWithoutClang(Project):
    def test_files_that_are_not_covered_pass_untouched(self):
        with mock.patch.dict(os.environ, {"PATH": ""}):
            self.assertEqual(self.edit("Thorn/README", "not covered\n", "// still not covered\n"), "")
            self.assertEqual(self.write("Other/src/b.cxx", "int b; // no interface.ccl\n"), "")
            self.assertEqual(self.write("Thorn/test/x.py", "# python\n"), "")
            self.assertEqual(self.write("Thorn/a.cxx", "// not under src\n"), "")
            self.assertEqual(self.write("Thorn/src/a.cxx.orig", "// not a C/C++ suffix\n"), "")
            with tempfile.TemporaryDirectory() as other:
                self.assertEqual(self.edit(None, "", "// why\n", path=os.path.join(other, "Thorn", "src", "a.cxx")), "")

    def test_no_clang_is_a_system_message(self):
        with mock.patch.dict(os.environ, {"PATH": ""}):
            output = self.edit("Thorn/src/a.cxx", "int a = 1;\n", "int a = 1; // why\n")
        self.assertEqual(self.skipped(output), "comment check skipped: clang++ not found on PATH")

    def test_a_bad_payload_is_a_system_message(self):
        for payload in [
            "not json",
            "[]",
            {"tool_name": "Edit", "tool_input": {"file_path": self.path("Thorn/src/a.cxx"), "new_string": "x"}},
            {"tool_name": "Teleport", "tool_input": {"file_path": self.path("Thorn/src/a.cxx")}},
        ]:
            with self.subTest(payload=payload):
                self.skipped(self.run_hook(payload))


def copy_tool(root):
    shutil.copytree(TOOL, os.path.join(root, "scripts", "comments"), ignore=shutil.ignore_patterns("__pycache__"))


def edit_payload(root):
    return {
        "tool_name": "Edit",
        "tool_input": {
            "file_path": os.path.join(root, "Thorn/src/a.cxx"),
            "old_string": "int a = 1;\n",
            "new_string": "int a = 1; // x\n",
        },
        "cwd": root,
    }


class Cli(Project):
    def run_tool(self, args, stdin="", env=None):
        return subprocess.run(
            [sys.executable, os.path.join(self.root, "scripts", "comments"), *args],
            input=stdin, capture_output=True, text=True, env=env,
        )

    def setUp(self):
        super().setUp()
        copy_tool(self.root)

    @unittest.skipUnless(HAVE_CLANG, "needs clang++ on PATH")
    def test_hook_edit_denies(self):
        result = self.run_tool(["hook-edit"], json.dumps(edit_payload(self.root)))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["hookSpecificOutput"]["permissionDecision"], "deny")

    def test_hook_edit_without_clang_exits_0_with_a_system_message(self):
        result = self.run_tool(["hook-edit"], json.dumps(edit_payload(self.root)), env=dict(os.environ, PATH=""))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout), {"systemMessage": "comment check skipped: clang++ not found on PATH"})

    def test_hook_edit_with_a_bad_payload_exits_0(self):
        result = self.run_tool(["hook-edit"], "not json")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("systemMessage", json.loads(result.stdout))

    @unittest.skipUnless(HAVE_CLANG, "needs clang++ on PATH")
    def test_check_exit_codes(self):
        write_files(self.root, {"Thorn/src/c.cxx": "int c; // why\n"})
        result = self.run_tool(["check", os.path.join(self.root, "Thorn")])
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertEqual(result.stdout.split("\n")[0], "Thorn/src/c.cxx:1   // why")
        self.assertEqual(result.stdout.rstrip("\n").split("\n")[-1], "✗ no-comments: 1 comment")
        result = self.run_tool(["check", os.path.join(self.root, "Thorn/src/a.cxx")])
        self.assertEqual((result.returncode, result.stdout), (0, "✓ no-comments: 0 comments\n"))

    def test_check_without_clang_exits_2(self):
        result = self.run_tool(["check", os.path.join(self.root, "Thorn")], env=dict(os.environ, PATH=""))
        self.assertEqual(result.returncode, 2)
        self.assertIn("clang++ not found on PATH", result.stderr)

    def test_check_of_a_missing_path_exits_2(self):
        self.assertEqual(self.run_tool(["check", os.path.join(self.root, "nope")]).returncode, 2)


def configured_pretooluse():
    with open(SETTINGS) as f:
        entries = json.load(f)["hooks"].get("PreToolUse", [])
    return [(entry.get("matcher"), hook["command"]) for entry in entries for hook in entry["hooks"]]


@unittest.skipUnless(os.path.isfile(SETTINGS), "needs .claude/settings.json")
class Settings(Project):
    def run_command(self, command, project, payload):
        env = {k: v for k, v in os.environ.items() if k != "CLAUDE_PROJECT_DIR"}
        if project is not None:
            env["CLAUDE_PROJECT_DIR"] = project
        return subprocess.run(["sh", "-c", command], input=json.dumps(payload), env=env, capture_output=True, text=True)

    def test_one_edit_hook_runs_hook_edit(self):
        [(matcher, command)] = configured_pretooluse()
        self.assertEqual(matcher, "Edit|Write")
        self.assertTrue(command.endswith('python3 "$d" hook-edit'), command)

    def test_the_command_is_silent_where_the_tool_is_absent(self):
        [(_, command)] = configured_pretooluse()
        write_files(self.root, {"scripts/comments/__pycache__/hooks.pyc": ""})
        for project in [self.root, None]:
            result = self.run_command(command, project, edit_payload(self.root))
            self.assertEqual((result.returncode, result.stdout, result.stderr), (0, "", ""), project)

    @unittest.skipUnless(HAVE_CLANG, "needs clang++ on PATH")
    def test_the_command_runs_the_project_s_tool(self):
        [(_, command)] = configured_pretooluse()
        copy_tool(self.root)
        result = self.run_command(command, self.root, edit_payload(self.root))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["hookSpecificOutput"]["permissionDecision"], "deny")


if __name__ == "__main__":
    unittest.main()
