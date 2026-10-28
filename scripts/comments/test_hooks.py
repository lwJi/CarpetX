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
from files import PROJECT
from hooks import simulate_edit
from report import CHANGE_RULE, REDO
from test_files import ISOLATED_GIT, git

TOOL = os.path.dirname(os.path.abspath(__file__))
SETTINGS = os.path.join(PROJECT, ".claude", "settings.json")
FILES = {
    "Thorn/interface.ccl": "IMPLEMENTS: Thorn\n",
    "Thorn/src/a.cxx": "namespace Thorn {\nint a = 1;\n} // namespace Thorn\n",
    "Thorn/src/c.cxx": "// why c\nint c = 1;\n\nint d = 1;\n",
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


class HookEdit(Project):
    def test_denies_a_new_comment(self):
        lines = self.denied(self.edit("Thorn/src/a.cxx", "int a = 1;\n", "int a = 1; // why\n"))
        self.assertEqual(lines[0], "Thorn/src/a.cxx:2   new       // why")
        self.assertEqual(lines[1:-1], CHANGE_RULE.split("\n") + [REDO])
        self.assertEqual(lines[-1], "✗ no-comments: 1 new comment line")

    def test_denies_a_code_change_that_keeps_the_comment_over_it(self):
        lines = self.denied(self.edit("Thorn/src/c.cxx", "int c = 1;\n", "int c = 2;\n"))
        self.assertEqual(lines[0], "Thorn/src/c.cxx:1   touched   // why c")
        self.assertEqual(lines[-1], "✗ no-comments: 1 touched comment")

    def test_allows_the_change_with_the_comment_deleted(self):
        self.assertEqual(self.edit("Thorn/src/c.cxx", "// why c\nint c = 1;\n", "int c = 2;\n"), "")

    def test_allows_an_edit_to_code_no_comment_covers(self):
        self.assertEqual(self.edit("Thorn/src/c.cxx", "int d = 1;\n", "int d = 2;\n"), "")
        self.assertEqual(self.edit("Thorn/src/a.cxx", "int a = 1;\n", "int a = 2;\n"), "")
        self.assertEqual(
            self.edit("Thorn/src/a.cxx", "} // namespace Thorn\n", "} // namespace Thorn\nnamespace B {\n} // namespace B\n"),
            "",
        )

    def test_denies_deleting_some_lines_of_a_comment(self):
        write_files(self.root, {"Thorn/src/c.cxx": "// one\n// two\n// three\nint c;\n"})
        self.assertEqual(self.denied(self.edit("Thorn/src/c.cxx", "// two\n", ""))[0], "Thorn/src/c.cxx:1-2   touched   // one")
        self.assertEqual(self.edit("Thorn/src/c.cxx", "// one\n// two\n// three\n", ""), "")

    def test_a_reworded_line_of_a_longer_comment_touches_it(self):
        write_files(self.root, {"Thorn/src/c.cxx": "// one\n// two\nint c;\n"})
        lines = self.denied(self.edit("Thorn/src/c.cxx", "// two\n", "// too\n"))
        self.assertEqual(lines[0], "Thorn/src/c.cxx:1-2   touched   // one")
        self.assertEqual(lines[-1], "✗ no-comments: 1 touched comment")

    def test_write(self):
        self.denied(self.write("Thorn/src/new.cxx", "/* why */\nint n;\n"))
        self.assertEqual(self.write("Thorn/src/b.cxx", "namespace Thorn {\nint b;\n} // namespace Thorn\n"), "")
        self.denied(self.write("Thorn/src/c.cxx", "// why c\nint c = 2;\n\nint d = 1;\n"))

    def test_an_edit_that_creates_a_file_checks_new_string(self):
        lines = self.denied(self.edit("Thorn/src/new.cxx", "", "int n; /* why */\n"))
        self.assertEqual(lines[0], "Thorn/src/new.cxx:1   new       /* why */")

    def test_a_relative_path_is_taken_from_cwd(self):
        self.denied(self.edit(None, "int a = 1;\n", "int a = 1; // why\n", path="Thorn/src/a.cxx"))

    def test_replace_all_reports_every_copy(self):
        write_files(self.root, {"Thorn/src/b.cxx": "int x;\nint x;\n"})
        self.assertEqual(self.edit("Thorn/src/b.cxx", "int x;", "int x; // why"), "")
        lines = self.denied(self.edit("Thorn/src/b.cxx", "int x;", "int x; // why", replace_all=True))
        self.assertEqual(lines[:2], ["Thorn/src/b.cxx:1   new       // why", "Thorn/src/b.cxx:2   new       // why"])

    def test_an_old_string_that_does_not_match_is_left_to_the_edit_tool(self):
        self.assertEqual(self.edit("Thorn/src/a.cxx", "int nope;\n", "int nope; // why\n"), "")

    def test_a_tool_failure_is_a_system_message(self):
        with mock.patch("spans.lex", side_effect=RuntimeError("lexer crash")):
            output = self.edit("Thorn/src/a.cxx", "int a = 1;\n", "int a = 1; // why\n")
        self.assertEqual(self.skipped(output), "comment check skipped: RuntimeError: lexer crash")

    def test_files_that_are_not_covered_pass_untouched(self):
        self.assertEqual(self.edit("Thorn/README", "not covered\n", "// still not covered\n"), "")
        self.assertEqual(self.write("Other/src/b.cxx", "int b; // no interface.ccl\n"), "")
        self.assertEqual(self.write("Thorn/test/x.py", "# python\n"), "")
        self.assertEqual(self.write("Thorn/a.cxx", "// not under src\n"), "")
        self.assertEqual(self.write("Thorn/src/a.cxx.orig", "// not a C/C++ suffix\n"), "")
        with tempfile.TemporaryDirectory() as other:
            self.assertEqual(self.edit(None, "", "// why\n", path=os.path.join(other, "Thorn", "src", "a.cxx")), "")

    def test_hash_languages_and_fortran_are_judged(self):
        write_files(self.root, {"Thorn/param.ccl": "# old\nINT n \"n\"\n{\n} 1\n\nINT m \"m\"\n{\n} 1\n"})
        lines = self.denied(self.edit("Thorn/param.ccl", "} 1\n\n", "} 1 # why\n\n"))
        self.assertEqual(lines[0], "Thorn/param.ccl:4   new       # why")
        self.assertEqual(self.denied(self.edit("Thorn/param.ccl", "} 1\n\n", "} 2\n\n"))[0],
                         "Thorn/param.ccl:1   touched   # old")
        self.assertEqual(self.edit("Thorn/param.ccl", "\nINT m \"m\"\n{\n} 1\n", "\nINT m \"m\"\n{\n} 2\n"), "")
        self.assertEqual(self.write("Thorn/param.ccl", "INT n \"n\"\n{\n} 2\n"), "")
        self.denied(self.write("Thorn/test/t.par", "A::b = 1\n# why\n"))
        self.denied(self.write("Thorn/src/make.code.defn", "# why\nSRCS = a.cxx\n"))
        self.denied(self.write("Thorn/src/f.F90", "x = 1 ! why\n"))
        self.assertEqual(self.write("Thorn/test/t.par", "A::b = \"# not a comment\"\n"), "")

    def test_a_bad_payload_is_a_system_message(self):
        for payload in [
            "not json",
            "[]",
            {"tool_name": "Edit", "tool_input": {"file_path": self.path("Thorn/src/a.cxx"), "new_string": "x"}},
            {"tool_name": "Teleport", "tool_input": {"file_path": self.path("Thorn/src/a.cxx")}},
        ]:
            with self.subTest(payload=payload):
                self.skipped(self.run_hook(payload))


PARAM = "# old prose\nINT n \"n\"\n{\n  *:* :: \"\"\n} 1\n\nINT m \"m\"\n{\n  *:* :: \"\"\n} 1\n"
STOP_FILES = {
    "Thorn/interface.ccl": "IMPLEMENTS: Thorn\n",
    "Thorn/param.ccl": PARAM,
    "Thorn/src/a.cxx": "namespace Thorn {\nint a = 1; // old\n} // namespace Thorn\n",
    "Thorn/src/make.code.defn": "SRCS = a.cxx\n",
    "Thorn/test/t.par": "Thorn::n = 2\n",
    "Thorn/README": "not covered\n",
}


BIG = "// why big\nint keep = 1;\n\n" + "".join(f"int f{n} = {n};\n" for n in range(40))
SHRUNK = "// why big\nint keep = 1;\n"


def with_m(default):
    return PARAM[: -len("} 1\n")] + f"}} {default}\n"


class Branch(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = os.path.realpath(tmp.name)
        patcher = mock.patch.dict(os.environ, ISOLATED_GIT)
        patcher.start()
        self.addCleanup(patcher.stop)
        git(self.root, "init", "-q", "-b", "dev")
        self.commit(STOP_FILES)
        git(self.root, "switch", "-q", "-c", "work")

    def commit(self, files):
        write_files(self.root, files)
        git(self.root, "add", "-A")
        git(self.root, "commit", "-q", "-m", "c")

    def merge(self, branch):
        git(self.root, "merge", "--no-ff", "-q", "-m", "m", branch)

    def merge_upstream(self, files):
        git(self.root, "switch", "-q", "-c", "upstream", "dev")
        self.commit(files)
        git(self.root, "switch", "-q", "work")
        self.commit({"Thorn/test/t.par": "Thorn::n = 3\n"})
        self.merge("upstream")

    def on_dev(self, files):
        git(self.root, "switch", "-q", "dev")
        self.commit(files)
        git(self.root, "switch", "-q", "work")
        git(self.root, "merge", "-q", "--ff-only", "dev")

    def rename_then_shrink(self, shrunk):
        self.on_dev({"Thorn/src/big.cxx": BIG})
        git(self.root, "mv", "Thorn/src/big.cxx", "Thorn/src/kept.cxx")
        git(self.root, "commit", "-q", "-m", "rename")
        self.commit({"Thorn/src/kept.cxx": shrunk})

    def append(self, name, text):
        with open(os.path.join(self.root, name), "a") as f:
            f.write(text)

    def read(self, name):
        with open(os.path.join(self.root, name)) as f:
            return f.read()


class HookStop(Branch):
    def run_hook(self, payload=None):
        out = io.StringIO()
        payload = {"hook_event_name": "Stop", "stop_hook_active": False} if payload is None else payload
        raw = payload if isinstance(payload, str) else json.dumps(payload)
        self.assertEqual(hooks.hook_stop(raw, out, root=self.root), 0)
        return out.getvalue()

    def blocked(self, output):
        self.assertTrue(output, "the hook printed nothing")
        decision = json.loads(output)
        self.assertEqual(sorted(decision), ["decision", "reason"])
        self.assertEqual(decision["decision"], "block")
        return decision["reason"].split("\n")

    def test_a_clean_branch_stops(self):
        self.assertEqual(self.run_hook(), "")
        self.commit({"Thorn/param.ccl": with_m(2)})
        self.assertEqual(self.run_hook(), "")

    def test_a_comment_written_to_the_working_tree_blocks(self):
        self.append("Thorn/param.ccl", "# x\n")
        lines = self.blocked(self.run_hook())
        self.assertEqual(lines[0], "Thorn/param.ccl:11   new       # x")
        self.assertEqual(lines[1:-1], CHANGE_RULE.split("\n"))
        self.assertEqual(lines[-1], "✗ no-comments: 1 new comment line")

    def test_code_changed_under_a_comment_blocks_until_the_comment_goes(self):
        self.commit({"Thorn/param.ccl": PARAM.replace("} 1\n\n", "} 2\n\n")})
        lines = self.blocked(self.run_hook())
        self.assertEqual(lines[0], "Thorn/param.ccl:1   touched   # old prose")
        self.assertEqual(lines[-1], "✗ no-comments: 1 touched comment")
        write_files(self.root, {"Thorn/param.ccl": PARAM.replace("} 1\n\n", "} 2\n\n").replace("# old prose\n", "")})
        self.assertEqual(self.run_hook(), "")

    def test_stop_hook_active_allows_the_stop(self):
        self.append("Thorn/param.ccl", "# x\n")
        self.assertEqual(self.run_hook({"hook_event_name": "Stop", "stop_hook_active": True}), "")
        self.assertEqual(self.run_hook({"hook_event_name": "SubagentStop", "stop_hook_active": True}), "")
        self.blocked(self.run_hook({"hook_event_name": "SubagentStop", "stop_hook_active": False}))

    def test_a_comment_committed_on_the_branch_blocks(self):
        self.commit({"Thorn/test/t.par": "Thorn::n = 2 # why\n"})
        self.assertEqual(self.blocked(self.run_hook())[0], "Thorn/test/t.par:1   new       # why")

    def test_every_kind_of_file_is_checked(self):
        self.append("Thorn/src/a.cxx", "#if 0\n#endif\n")
        self.append("Thorn/src/make.code.defn", "# m\n")
        write_files(self.root, {"Thorn/src/new.cxx": "int n; // n\n", "Thorn/schedule.ccl": "/* s */\n"})
        lines = self.blocked(self.run_hook())
        self.assertEqual(lines[:4], ["Thorn/schedule.ccl:1         new       /* s */",
                                     "Thorn/src/a.cxx:4            new       #if 0",
                                     "Thorn/src/make.code.defn:2   new       # m",
                                     "Thorn/src/new.cxx:1          new       // n"])

    def test_comments_that_dev_already_has_are_not_reported(self):
        self.append("Thorn/src/a.cxx", "int b = 2;\n")
        self.commit({"Thorn/param.ccl": PARAM + "\nINT k \"k\"\n{\n} 1\n"})
        self.assertEqual(self.run_hook(), "")

    def test_a_renamed_file_keeps_its_old_comments(self):
        git(self.root, "mv", "Thorn/param.ccl", "Thorn/test/test.ccl")
        self.assertEqual(self.run_hook(), "")
        self.append("Thorn/test/test.ccl", "# new\n")
        self.assertEqual(self.blocked(self.run_hook())[0], "Thorn/test/test.ccl:11   new       # new")

    def test_a_renamed_file_shrunk_in_a_later_commit_keeps_its_comments(self):
        self.rename_then_shrink(SHRUNK)
        self.assertEqual(self.run_hook(), "")

    def test_a_comment_added_to_a_renamed_and_shrunk_file_is_reported(self):
        self.rename_then_shrink(SHRUNK)
        self.append("Thorn/src/kept.cxx", "int z = 0; // added\n")
        lines = self.blocked(self.run_hook())
        self.assertEqual(lines[0], "Thorn/src/kept.cxx:3   new       // added")
        self.assertEqual(lines[-1], "✗ no-comments: 1 new comment line")

    def test_a_renamed_and_rewritten_file_is_judged_as_new(self):
        git(self.root, "mv", "Thorn/param.ccl", "Thorn/test/test.ccl")
        write_files(self.root, {"Thorn/test/test.ccl": "# old prose\nTEST t\n{\n  NPROCS 2\n}\n"})
        self.assertEqual(self.blocked(self.run_hook())[0], "Thorn/test/test.ccl:1   new       # old prose")

    def test_comments_merged_in_from_another_branch_are_not_reported(self):
        self.merge_upstream({
            "Thorn/param.ccl": PARAM + "# upstream\n",
            "Thorn/schedule.ccl": "# upstream schedule\n",
        })
        self.assertEqual(self.run_hook(), "")

    def test_code_another_branch_changed_under_a_comment_is_not_reported(self):
        self.merge_upstream({"Thorn/param.ccl": PARAM.replace("} 1\n\n", "} 2\n\n")})
        self.assertEqual(self.run_hook(), "")

    def test_the_merge_resolution_itself_is_judged(self):
        self.merge_upstream({"Thorn/param.ccl": PARAM + "# upstream\n"})
        merged = self.read("Thorn/param.ccl")
        write_files(self.root, {"Thorn/param.ccl": merged.replace("} 1\n#", "} 1 # resolution\n#")})
        git(self.root, "commit", "-q", "--amend", "--no-edit", "-a")
        lines = self.blocked(self.run_hook())
        self.assertEqual(lines[0], "Thorn/param.ccl:10   new       # resolution")
        self.assertEqual(lines[-1], "✗ no-comments: 1 new comment line")

    def test_a_merge_on_the_branch_s_own_history_is_walked_first_parent(self):
        git(self.root, "switch", "-q", "-c", "side")
        self.commit({"Thorn/test/t.par": "Thorn::n = 2 # side\n"})
        git(self.root, "switch", "-q", "work")
        self.commit({"Thorn/param.ccl": with_m("1 # work")})
        self.merge("side")
        lines = self.blocked(self.run_hook())
        self.assertEqual(lines[0], "Thorn/param.ccl:10   new       # work")
        self.assertEqual(lines[-1], "✗ no-comments: 1 new comment line")

    def test_a_merged_in_comment_is_found_under_the_file_s_old_name(self):
        git(self.root, "switch", "-q", "-c", "upstream", "dev")
        self.commit({"Thorn/param.ccl": PARAM + "# upstream\n"})
        git(self.root, "switch", "-q", "work")
        git(self.root, "mv", "Thorn/param.ccl", "Thorn/test/test.ccl")
        git(self.root, "commit", "-q", "-m", "rename")
        self.merge("upstream")
        self.assertIn("# upstream\n", self.read("Thorn/test/test.ccl"))
        self.assertEqual(self.run_hook(), "")

    def test_a_comment_moved_to_another_file_is_new_there(self):
        self.commit({"Thorn/param.ccl": PARAM.replace("# old prose\n", ""), "Thorn/schedule.ccl": "# old prose\n"})
        self.assertEqual(self.blocked(self.run_hook())[0], "Thorn/schedule.ccl:1   new       # old prose")

    def test_a_file_merged_in_and_then_renamed_keeps_its_comments(self):
        self.merge_upstream({"Thorn/schedule.ccl": "# upstream schedule\n"})
        git(self.root, "mv", "Thorn/schedule.ccl", "Thorn/test/test.ccl")
        git(self.root, "commit", "-q", "-m", "rename")
        self.assertEqual(self.run_hook(), "")

    def test_a_renamed_thorn_keeps_its_comments(self):
        git(self.root, "mv", "Thorn", "Renamed")
        self.assertEqual(self.run_hook(), "")
        self.append("Renamed/param.ccl", "# new\n")
        self.assertEqual(self.blocked(self.run_hook())[0], "Renamed/param.ccl:11   new       # new")

    def test_files_that_are_not_covered_or_are_gone_are_skipped(self):
        self.append("Thorn/README", "# not covered\n")
        write_files(self.root, {"scripts/x.par": "# not a thorn\n", "Other/param.ccl": "# no interface.ccl\n"})
        os.remove(os.path.join(self.root, "Thorn/test/t.par"))
        os.remove(os.path.join(self.root, "Thorn/param.ccl"))
        self.assertEqual(self.run_hook(), "")

    def test_ignored_files_are_skipped(self):
        self.commit({".gitignore": "*.par\n"})
        write_files(self.root, {"Thorn/test/u.par": "# ignored\n"})
        self.assertEqual(self.run_hook(), "")

    def test_without_dev_only_uncommitted_changes_count(self):
        self.commit({"Thorn/test/t.par": "Thorn::n = 2 # committed\n"})
        git(self.root, "branch", "-q", "-D", "dev")
        self.assertEqual(self.run_hook(), "")
        self.append("Thorn/test/t.par", "# x\n")
        self.assertEqual(self.blocked(self.run_hook())[0], "Thorn/test/t.par:2   new       # x")

    def test_a_bad_payload_or_no_repository_is_a_system_message(self):
        for payload in ["not json", "[]"]:
            with self.subTest(payload=payload):
                self.assertIn("systemMessage", json.loads(self.run_hook(payload)))
        with tempfile.TemporaryDirectory() as other:
            out = io.StringIO()
            hooks.hook_stop(json.dumps({"stop_hook_active": False}), out, root=other)
            self.assertIn("systemMessage", json.loads(out.getvalue()))


class HookEditOnABranch(Branch):
    def run_edit(self, tool_name, tool_input):
        out = io.StringIO()
        payload = {"tool_name": tool_name, "tool_input": tool_input, "cwd": self.root}
        self.assertEqual(hooks.hook_edit(json.dumps(payload), out, root=self.root), 0)
        output = out.getvalue()
        if output and "permissionDecision" in json.loads(output).get("hookSpecificOutput", {}):
            return output
        path = os.path.join(self.root, tool_input["file_path"])
        before = self.read(tool_input["file_path"]) if os.path.isfile(path) else None
        write_files(self.root, {tool_input["file_path"]: hooks.simulate_edit(tool_name, tool_input, before)})
        return output

    def edit(self, name, old, new):
        return self.run_edit("Edit", {"file_path": name, "old_string": old, "new_string": new})

    def write(self, name, content):
        return self.run_edit("Write", {"file_path": name, "content": content})

    def denied(self, output):
        self.assertTrue(output, "the hook printed nothing")
        return json.loads(output)["hookSpecificOutput"]["permissionDecisionReason"].split("\n")

    def stop(self):
        out = io.StringIO()
        hooks.hook_stop(json.dumps({"stop_hook_active": False}), out, root=self.root)
        return out.getvalue()

    def test_a_comment_moved_by_two_edits_is_new_where_it_lands(self):
        self.assertEqual(self.edit("Thorn/param.ccl", "# old prose\n", ""), "")
        lines = self.denied(self.edit("Thorn/param.ccl", "} 1\n\n", "} 1\n# old prose\n\n"))
        self.assertEqual(lines[0], "Thorn/param.ccl:5   new       # old prose")

    def test_a_copy_is_denied(self):
        lines = self.denied(self.write("Thorn/schedule.ccl", "# old prose\n"))
        self.assertEqual(lines[0], "Thorn/schedule.ccl:1   new       # old prose")
        self.assertEqual(lines[-1], "✗ no-comments: 1 new comment line")

    def test_a_comment_the_branch_deleted_can_be_put_back_where_it_was(self):
        self.commit({"Thorn/param.ccl": PARAM.replace("# old prose\n", "")})
        self.assertEqual(self.edit("Thorn/param.ccl", "INT n", "# old prose\nINT n"), "")
        self.assertEqual(self.stop(), "")

    def test_a_comment_deleted_from_a_renamed_and_shrunk_file_can_be_put_back(self):
        self.rename_then_shrink("int keep = 1;\n")
        self.assertEqual(self.edit("Thorn/src/kept.cxx", "int keep", "// why big\nint keep"), "")
        self.assertEqual(self.stop(), "")

    def test_code_the_branch_changed_under_a_comment_can_be_changed_back(self):
        self.commit({"Thorn/param.ccl": PARAM.replace("} 1\n\n", "} 2\n\n")})
        self.assertTrue(self.stop())
        self.assertEqual(self.edit("Thorn/param.ccl", "} 2\n\n", "} 1\n\n"), "")
        self.assertEqual(self.stop(), "")

    def test_an_edit_is_not_blamed_for_what_the_branch_did_before_it(self):
        self.commit({"Thorn/param.ccl": PARAM.replace("} 1\n\n", "} 2\n\n")})
        self.assertEqual(self.edit("Thorn/param.ccl", "INT m", "INT mm"), "")
        lines = self.denied(self.edit("Thorn/src/a.cxx", "int a = 1;", "int a = 2;"))
        self.assertEqual(lines[0], "Thorn/src/a.cxx:2   touched   // old")

    def test_the_report_names_only_the_edited_file(self):
        write_files(self.root, {"Thorn/schedule.ccl": "# written by a shell command\n"})
        lines = self.denied(self.edit("Thorn/param.ccl", "} 1\n\n", "} 1\n\n# new\n# also new\n"))
        self.assertEqual(lines[:2], ["Thorn/param.ccl:7   new       # new", "Thorn/param.ccl:8   new       # also new"])
        self.assertEqual(lines[-1], "✗ no-comments: 2 new comment lines")

    def test_without_the_branch_the_edit_is_judged_alone(self):
        self.assertEqual(self.edit("Thorn/param.ccl", "# old prose\n", ""), "")
        with mock.patch("hooks.merge_base", side_effect=RuntimeError("git crashed")):
            lines = self.denied(self.edit("Thorn/param.ccl", "INT n", "# old prose\nINT n"))
        self.assertEqual(lines[0], "Thorn/param.ccl:1   new       # old prose")


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

    def test_hook_edit_denies(self):
        result = self.run_tool(["hook-edit"], json.dumps(edit_payload(self.root)))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["hookSpecificOutput"]["permissionDecision"], "deny")

    def test_hook_edit_with_a_bad_payload_exits_0(self):
        result = self.run_tool(["hook-edit"], "not json")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("systemMessage", json.loads(result.stdout))

    def test_check_exit_codes(self):
        write_files(self.root, {"Thorn/src/c.cxx": "int c; // why\n"})
        result = self.run_tool(["check", os.path.join(self.root, "Thorn")])
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertEqual(result.stdout.split("\n")[0], "Thorn/src/c.cxx:1   // why")
        self.assertEqual(result.stdout.rstrip("\n").split("\n")[-1], "✗ no-comments: 1 comment line")
        result = self.run_tool(["check", os.path.join(self.root, "Thorn/src/a.cxx")])
        self.assertEqual((result.returncode, result.stdout), (0, "✓ no-comments: 0 comment lines\n"))

    def test_check_of_a_missing_path_exits_2(self):
        self.assertEqual(self.run_tool(["check", os.path.join(self.root, "nope")]).returncode, 2)

    def test_check_without_a_path_scans_the_whole_tree(self):
        write_files(self.root, {"Thorn/param.ccl": "# p\n", "Thorn/src/c.cxx": "int c; // why\n", "scripts/x.par": "# no\n"})
        result = self.run_tool(["check"])
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertEqual(result.stdout.split("\n")[:2], ["Thorn/param.ccl:1   # p", "Thorn/src/c.cxx:1   // why"])
        self.assertEqual(result.stdout.rstrip("\n").split("\n")[-1], "✗ no-comments: 2 comment lines")

    @unittest.skipIf(os.geteuid() == 0, "root reads a file without read permission")
    def test_a_file_that_cannot_be_read_names_the_file(self):
        write_files(self.root, {"Thorn/src/c.cxx": "int c;\n"})
        os.chmod(self.path("Thorn/src/c.cxx"), 0)
        result = self.run_tool(["check", os.path.join(self.root, "Thorn")])
        self.assertEqual(result.returncode, 2)
        self.assertRegex(result.stderr, r"^comments check: Thorn/src/c\.cxx: \[Errno 13\] Permission denied")

    def test_hook_stop_with_a_bad_payload_exits_0(self):
        result = self.run_tool(["hook-stop"], "not json")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("systemMessage", json.loads(result.stdout))


def configured(event):
    with open(SETTINGS) as f:
        entries = json.load(f)["hooks"].get(event, [])
    return [
        (entry.get("matcher"), hook["command"])
        for entry in entries
        for hook in entry["hooks"]
        if "scripts/comments" in hook["command"]
    ]


def configured_pretooluse():
    return configured("PreToolUse")


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

    def test_the_command_runs_the_project_s_tool(self):
        [(_, command)] = configured_pretooluse()
        copy_tool(self.root)
        result = self.run_command(command, self.root, edit_payload(self.root))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["hookSpecificOutput"]["permissionDecision"], "deny")

    def test_stop_and_subagent_stop_run_hook_stop(self):
        for event in ("Stop", "SubagentStop"):
            with self.subTest(event=event):
                [(matcher, command)] = configured(event)
                self.assertIsNone(matcher)
                self.assertTrue(command.endswith('python3 "$d" hook-stop'), command)
                self.assertEqual(command.replace("hook-stop", "hook-edit"), configured_pretooluse()[0][1])

    def test_the_stop_command_is_silent_where_the_tool_is_absent(self):
        [(_, command)] = configured("Stop")
        for project in [self.root, None]:
            result = self.run_command(command, project, {"stop_hook_active": False})
            self.assertEqual((result.returncode, result.stdout, result.stderr), (0, "", ""), project)

    def test_the_stop_command_runs_the_project_s_tool(self):
        [(_, command)] = configured("Stop")
        copy_tool(self.root)
        with mock.patch.dict(os.environ, ISOLATED_GIT):
            git(self.root, "init", "-q", "-b", "dev")
            git(self.root, "add", "-A")
            git(self.root, "commit", "-q", "-m", "c")
            write_files(self.root, {"Thorn/param.ccl": "# x\n"})
            result = self.run_command(command, self.root, {"stop_hook_active": False})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["decision"], "block")


CLEAN = "✓ no-comments: no new or touched comments\n"


class CheckSince(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = os.path.realpath(tmp.name)
        patcher = mock.patch.dict(os.environ, ISOLATED_GIT)
        patcher.start()
        self.addCleanup(patcher.stop)
        git(self.root, "init", "-q", "-b", "dev")
        write_files(self.root, STOP_FILES)
        git(self.root, "add", "-A")
        git(self.root, "commit", "-q", "-m", "base")
        self.base = git(self.root, "rev-parse", "HEAD").strip()
        copy_tool(self.root)
        git(self.root, "add", "-A")
        git(self.root, "commit", "-q", "-m", "tool")

    def run_tool(self, *args):
        return subprocess.run(
            [sys.executable, os.path.join(self.root, "scripts", "comments"), *args], capture_output=True, text=True
        )

    def append(self, name, text):
        with open(os.path.join(self.root, name), "a") as f:
            f.write(text)

    def test_old_comments_pass_and_new_ones_fail(self):
        result = self.run_tool("check", "--since", self.base)
        self.assertEqual((result.returncode, result.stdout), (0, CLEAN), result.stderr)
        self.append("Thorn/param.ccl", "# x\n")
        self.append("Thorn/test/t.par", "Thorn::n = 3\n")
        result = self.run_tool("check", "--since", self.base)
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertEqual(result.stdout.split("\n")[0], "Thorn/param.ccl:11   new       # x")
        self.assertEqual(result.stdout.rstrip("\n").split("\n")[-1], "✗ no-comments: 1 new comment line")

    def test_a_touched_comment_fails(self):
        write_files(self.root, {"Thorn/src/a.cxx": STOP_FILES["Thorn/src/a.cxx"].replace("a = 1", "a = 2")})
        result = self.run_tool("check", "--since", self.base)
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertEqual(result.stdout.split("\n")[0], "Thorn/src/a.cxx:2   touched   // old")
        self.assertEqual(result.stdout.rstrip("\n").split("\n")[-1], "✗ no-comments: 1 touched comment")

    def test_a_comment_committed_since_the_base_fails(self):
        self.append("Thorn/test/t.par", "# why\n")
        git(self.root, "commit", "-q", "-am", "add a comment")
        result = self.run_tool("check", "--since", self.base)
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertEqual(result.stdout.split("\n")[0], "Thorn/test/t.par:2   new       # why")
        self.assertEqual(self.run_tool("check", "--since", "HEAD").returncode, 0)

    def test_a_merged_in_comment_passes_and_a_resolution_comment_fails(self):
        git(self.root, "switch", "-q", "-c", "upstream", self.base)
        self.append("Thorn/param.ccl", "# upstream\n")
        git(self.root, "commit", "-q", "-am", "upstream")
        git(self.root, "switch", "-q", "dev")
        git(self.root, "merge", "--no-ff", "-q", "-m", "m", "upstream")
        result = self.run_tool("check", "--since", self.base)
        self.assertEqual((result.returncode, result.stdout), (0, CLEAN), result.stderr)
        self.append("Thorn/test/t.par", "Thorn::n = 3 # resolution\n")
        git(self.root, "commit", "-q", "--amend", "--no-edit", "-a")
        result = self.run_tool("check", "--since", self.base)
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertEqual(result.stdout.split("\n")[0], "Thorn/test/t.par:2   new       # resolution")
        self.assertEqual(result.stdout.rstrip("\n").split("\n")[-1], "✗ no-comments: 1 new comment line")

    def test_a_base_that_moved_on_is_judged_from_the_merge_base(self):
        write_files(self.root, {"Thorn/test/u.par": "# dev deletes this later\nThorn::n = 4\n"})
        git(self.root, "add", "-A")
        git(self.root, "commit", "-q", "-m", "an existing comment")
        git(self.root, "switch", "-q", "-c", "work")
        self.append("Thorn/test/t.par", "Thorn::n = 3 # why\n")
        git(self.root, "commit", "-q", "-am", "work adds a comment")
        git(self.root, "switch", "-q", "dev")
        write_files(
            self.root,
            {"Thorn/test/u.par": "Thorn::n = 4\n", "Thorn/param.ccl": PARAM.replace("# old prose", "# reworded prose")},
        )
        git(self.root, "commit", "-q", "-am", "dev deletes one comment and rewords another")
        git(self.root, "switch", "-q", "work")
        result = self.run_tool("check", "--since", "dev")
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertEqual(result.stdout.split("\n")[0], "Thorn/test/t.par:2   new       # why")
        self.assertEqual(result.stdout.rstrip("\n").split("\n")[-1], "✗ no-comments: 1 new comment line")

    def test_a_file_renamed_and_then_shrunk_is_judged_under_its_old_name(self):
        write_files(self.root, {"Thorn/src/big.cxx": BIG})
        git(self.root, "add", "-A")
        git(self.root, "commit", "-q", "-m", "big")
        base = git(self.root, "rev-parse", "HEAD").strip()
        git(self.root, "mv", "Thorn/src/big.cxx", "Thorn/src/kept.cxx")
        git(self.root, "commit", "-q", "-m", "rename")
        write_files(self.root, {"Thorn/src/kept.cxx": SHRUNK})
        git(self.root, "commit", "-q", "-am", "shrink")
        result = self.run_tool("check", "--since", base)
        self.assertEqual((result.returncode, result.stdout), (0, CLEAN), result.stderr)
        self.append("Thorn/src/kept.cxx", "int z = 0; // added\n")
        result = self.run_tool("check", "--since", base)
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertEqual(result.stdout.split("\n")[0], "Thorn/src/kept.cxx:3   new       // added")
        self.assertEqual(result.stdout.rstrip("\n").split("\n")[-1], "✗ no-comments: 1 new comment line")

    def test_a_rev_with_no_common_ancestor_exits_2(self):
        unrelated = git(self.root, "commit-tree", "HEAD^{tree}", "-m", "unrelated").strip()
        result = self.run_tool("check", "--since", unrelated)
        self.assertEqual(result.returncode, 2)
        self.assertIn("and HEAD have no common ancestor", result.stderr)

    def test_an_unknown_rev_exits_2(self):
        result = self.run_tool("check", "--since", "nonsense")
        self.assertEqual(result.returncode, 2)
        self.assertIn("'nonsense' is not a commit in this repository", result.stderr)

    def test_since_refuses_paths(self):
        result = self.run_tool("check", "--since", self.base, "Thorn")
        self.assertEqual(result.returncode, 2)
        self.assertIn("--since takes no PATH", result.stderr)


if __name__ == "__main__":
    unittest.main()
