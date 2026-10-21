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
from files import PROJECT, Lang
from hooks import added, simulate_edit, trimmed
from report import TRIM_RULE, trim_head
from rules import units
from test_files import ISOLATED_GIT, git

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


class Added(unittest.TestCase):
    def texts(self, before, after):
        return self.texts_from([before], after)

    def texts_from(self, befores, after):
        return [(u.line, u.text) for u in added(befores, after, Lang.CXX)]

    def test_a_new_comment_is_added(self):
        self.assertEqual(self.texts("int a;\n", "int a; // why\n"), [(1, "// why")])

    def test_comments_already_in_the_file_are_not_the_edit_s(self):
        self.assertEqual(self.texts("// one\nint x;\n// two\nint y;\n", "// one\nint x;\nint y;\n"), [])
        self.assertEqual(self.texts("// one\nint x;\n", "{\n  // one\n  int x;\n}\n"), [])

    def test_spacing_and_opener_changes_add_nothing(self):
        self.assertEqual(self.texts("int a;      // x y\n", "int a; // x  y\n"), [])
        self.assertEqual(self.texts("// one\n// two\nint x;\n", "/* one\n * two */\nint x;\n"), [])

    def test_re_wrapping_a_comment_adds_its_new_lines(self):
        before = "// The finest clock group: the finest\n// level, widened down to every coarser\nint x;\n"
        after = "// The finest clock group:\n// the finest level, widened down to every coarser\nint x;\n"
        self.assertEqual(self.texts(before, after),
                         [(1, "// The finest clock group:"), (2, "// the finest level, widened down to every coarser")])

    def test_a_moved_comment_is_not_added(self):
        self.assertEqual(self.texts("// one\n// two\nint x;\nint y;\n", "int x;\n// one\n// two\nint y;\n"), [])
        self.assertEqual(self.texts("int x; // one\n", "// one\nint x;\n"), [])

    def test_a_reworded_or_appended_line_is_added(self):
        before = "// one two\n// three\nint x;\n"
        self.assertEqual(self.texts(before, "// one too\n// three\nint x;\n"), [(1, "// one too")])
        self.assertEqual(self.texts(before, "// one two\n// three\n// four\nint x;\n"), [(3, "// four")])

    def test_deleting_lines_adds_nothing(self):
        before = "// one\n// two\n// three\nint x;\n"
        self.assertEqual(self.texts(before, "// one\n// three\nint x;\n"), [])
        self.assertEqual(self.texts(before, "// two\n// three\nint x;\n"), [])
        self.assertEqual(self.texts(before, "{\n  // one\n  int x;\n}\n"), [])
        self.assertEqual(self.texts("/* one\n   two\n   three */\n", "/* one\n   three */\n"), [])
        code = "// Copy in one go\n// int y = x;\n// y += 1;\nint x;\n"
        self.assertEqual(self.texts(code, "// Copy in one go\nint x;\n"), [])
        self.assertEqual(self.texts_from(["int x;\n", before], "// one\nint x;\n"), [])

    def test_deleting_some_lines_and_rewording_another_adds_the_reworded_one(self):
        before = "// one two\n// three\n// four\nint x;\n"
        self.assertEqual(self.texts(before, "// one two\n// four!\nint x;\n"), [(2, "// four!")])
        self.assertEqual(self.texts(before, "// one\n// four\nint x;\n"), [(1, "// one")])

    def test_code_edits_around_comments_add_nothing(self):
        self.assertEqual(self.texts("// a\n// b\n", "// a\nint y;\n// b\n"), [])
        self.assertEqual(self.texts("// a\nint x;\n// b\n", "// a\n// b\n"), [])
        self.assertEqual(self.texts("// one two\n// three\n// four\nint x;\n", "// four\n// one two\nint x;\n"), [])
        before = "// a\n// c\nint x;\n// a\n// b\nint y;\n"
        self.assertEqual(self.texts(before, "// a\nint x;\n// c\nint y;\n// b\n"), [])

    def test_a_line_beyond_the_old_count_is_added_once(self):
        before = "// a\n// b\nint x;\n"
        self.assertEqual(self.texts(before, "// a\nint y;\n\n// a\n// b\nint x;\n"), [(4, "// a")])
        before = "// a\n// c\nint x;\n// a\n// b\nint y;\n"
        self.assertEqual(self.texts(before, "// a\nint x;\n// c\nint y;\n// b\n// b\n"), [(6, "// b")])

    def test_an_added_citation_does_not_take_a_line_s_match(self):
        before = "// See arXiv:1111.2177\n// for the stencil\nint x;\n"
        self.assertEqual(self.texts(before, "// See arXiv:1111.2177\nint y;\n" + before), [])

    def test_a_comment_in_any_starting_tree_is_not_added(self):
        befores = ["// a\nint x;\n", "int x;\n// b\nint y;\n"]
        self.assertEqual(self.texts_from(befores, "// a\nint x;\n// b\nint y;\n"), [])
        self.assertEqual(self.texts_from(befores, "// a\nint x;\n// b\nint y;\n// c\n"), [(5, "// c")])
        self.assertEqual(self.texts_from([], "int a; // x\n"), [(1, "// x")])

    def test_starting_trees_take_the_maximum_count(self):
        befores = ["int a; // x\nint b; // x\n", "int a; // x\n"]
        self.assertEqual(self.texts_from(befores, "int a; // x\nint b; // x\n"), [])
        self.assertEqual(self.texts_from(befores, "int a; // x\nint b; // x\nint c; // x\n"), [(3, "// x")])

    def test_a_duplicate_of_an_existing_comment_is_added(self):
        self.assertEqual(self.texts("int a; // x\n", "int a; // x\nint b; // x\n"), [(2, "// x")])

    def test_an_allowed_comment_does_not_hide_a_new_violation(self):
        before = "namespace X {\n} // namespace X\n"
        after = "// namespace X\nnamespace X {\n} // namespace X\n"
        self.assertEqual(self.texts(before, after), [(1, "// namespace X")])

    def test_an_allowed_comment_is_not_added(self):
        self.assertEqual(self.texts("int a;\n", "namespace X {\nint a;\n} // namespace X\n"), [])

    def test_a_new_guard_or_message_is_added(self):
        self.assertEqual(self.texts("int a;\n", "#if 0\nint a;\n#endif\n"), [(1, "#if 0")])
        self.assertEqual(self.texts("int a;\n", '#warning "x"\nint a;\n'), [(1, '#warning "x"')])
        self.assertEqual(len(added(["#if 0\nint a;\n#endif\n"], "#if 0\nint a;\n#endif\n#if 0\n#endif\n", Lang.CXX)), 1)

    def test_an_existing_guard_is_not_the_edit_s(self):
        before = "#if 0\nint a;\n#endif\n#warning \"x\"\n"
        self.assertEqual(self.texts(before, "#if 0\nint b;\nint c;\n#endif\n#warning \"x\"\n"), [])
        self.assertEqual(self.texts(before, "#  if   0\nint a;\n#endif\n#warning   \"x\"\n"), [])
        self.assertEqual(self.texts("#if 0\n#if 0\n#endif\n#endif\n", "#if 0\n#endif\n"), [])

    def test_changing_a_guard_adds_one(self):
        self.assertEqual(len(added(["#if X\n#endif\n"], "#if 1\n#endif\n", Lang.CXX)), 1)
        self.assertEqual(len(added(['#warning "x"\n'], '#warning "y"\n', Lang.CXX)), 1)

    def test_a_comment_cannot_excuse_a_directive(self):
        self.assertEqual(self.texts("// # if 0\n", "#if 0\n#endif\n"), [(1, "#if 0")])


class Trimmed(unittest.TestCase):
    def trims(self, before, after, lang=Lang.CXX):
        return [[u.text for u in gone] for gone in trimmed(units(before, lang), units(after, lang))]

    def test_deleting_some_lines_of_a_comment_is_a_trim(self):
        before = "// `tl = -1` syncs every timelevel\n// except the oldest.\nint f(int tl = -1);\n"
        self.assertEqual(self.trims(before, "// `tl = -1` syncs every timelevel\nint f(int tl = -1);\n"),
                         [["// except the oldest."]])

    def test_trims_come_in_file_order_with_their_old_lines(self):
        before = "// a\n// b\nint x;\n// c d e\n// f g h\n// i\nint y;\n"
        found = trimmed(units(before, Lang.CXX), units("// a\nint x;\n// c d e\n// i\nint y;\n", Lang.CXX))
        self.assertEqual([[(u.line, u.text) for u in gone] for gone in found], [[(2, "// b")], [(5, "// f g h")]])

    def test_deleting_moving_or_keeping_a_whole_comment_is_not(self):
        before = "// one\n// two\nint x;\n"
        self.assertEqual(self.trims(before, "int x;\n"), [])
        self.assertEqual(self.trims(before, "int y;\n// one\n// two\nint x;\n"), [])
        self.assertEqual(self.trims(before, "{\n  // one\n  // two\n  int x;\n}\n"), [])
        self.assertEqual(self.trims(before, "int x;\n// one\n// two\nint y;\n"), [])

    def test_splitting_a_comment_or_moving_one_of_its_lines_is_not(self):
        before = "// one\n// two\n// three\nint x;\n"
        self.assertEqual(self.trims(before, "// one\nint y;\n// two\n// three\nint x;\n"), [])
        self.assertEqual(self.trims(before, "// one\n// three\nint x;\n// two\nint y;\n"), [])

    def test_deleting_one_of_two_copies_is_not(self):
        copy = "// The ROCm compiler can't handle\n// operator[], so we use pointers\n"
        self.assertEqual(self.trims(copy + "int x;\n" + copy + "int y;\n", "int x;\n" + copy + "int y;\n"), [])

    def test_a_trim_of_one_of_two_copies_is_found(self):
        copy = "// The ROCm compiler can't handle\n// operator[], so we use pointers\n"
        after = "// The ROCm compiler can't handle\nint x;\n" + copy + "int y;\n"
        self.assertEqual(self.trims(copy + "int x;\n" + copy + "int y;\n", after),
                         [["// operator[], so we use pointers"]])

    def test_a_whole_deletion_takes_a_line_another_comment_shares(self):
        before = "// We sync every direction.\n// We don't sync the\n// oldest.\nint a;\n// We don't restrict the\n// oldest.\nint b;\n"
        self.assertEqual(self.trims(before, "int a;\n// We don't restrict the\n// oldest.\nint b;\n"), [])
        before = "// oldest.\nint a;\n// We don't restrict the\n// oldest.\nint b;\n"
        self.assertEqual(self.trims(before, "// oldest.\nint a;\nint b;\n"), [])

    def test_commented_out_code_counts_like_prose(self):
        before = "// Copy in one go\n// int y = x;\nint x;\n"
        self.assertEqual(self.trims(before, "// Copy in one go\nint x;\n"), [["// int y = x;"]])

    def test_a_citation_can_stay_alone(self):
        before = "// See arXiv:1111.2177\n// for the stencil\nint x;\n"
        self.assertEqual(self.trims(before, "// See arXiv:1111.2177\nint x;\n"), [])

    def test_the_lines_of_a_block_comment(self):
        self.assertEqual(self.trims("/* one\n   two\n   three */\nint x;\n", "/* one\n   three */\nint x;\n"),
                         [["two"]])

    def test_a_trailing_comment_and_the_lines_below_it_are_one_comment(self):
        before = "int a = 1; // one\n           // two\nint b;\n"
        self.assertEqual(self.trims(before, "int a = 1; // one\nint b;\n"), [["// two"]])
        self.assertEqual(self.trims("int a; // one\nint b; // two\n", "int a; // one\nint b;\n"), [])

    def test_hash_languages(self):
        before = "# The grid spacing\n# on the coarsest level\nREAL dx \"dx\"\n{\n} 1\n"
        self.assertEqual(self.trims(before, "# The grid spacing\nREAL dx \"dx\"\n{\n} 1\n", Lang.PARAM),
                         [["# on the coarsest level"]])
        before = "! The flux\n! at the face\nx = 1\n"
        self.assertEqual(self.trims(before, "! The flux\nx = 1\n", Lang.FORTRAN), [["! at the face"]])


class AddedHash(unittest.TestCase):
    def texts(self, before, after, lang):
        return [(u.line, u.text) for u in added([before], after, lang)]

    def test_only_new_hash_comments_are_added(self):
        before = "# header\nINT a \"a\"\n{\n} 1\n"
        self.assertEqual(self.texts(before, before + "# x\n", Lang.PARAM), [(5, "# x")])
        self.assertEqual(self.texts(before, before.replace("1\n", "2\n"), Lang.PARAM), [])

    def test_re_indenting_adds_nothing_and_re_wrapping_adds_the_new_lines(self):
        param = "INT a \"a\"\n{\n} 1\n"
        before = "# one two\n# three\n" + param
        self.assertEqual(self.texts(before, "#   one two\n  #   three\n" + param, Lang.PARAM), [])
        self.assertEqual(self.texts(before, "# one\n# two three\n" + param, Lang.PARAM), [(1, "# one"), (2, "# two three")])
        self.assertEqual(self.texts(before, "# one too\n# three\n" + param, Lang.PARAM), [(1, "# one too")])
        fortran = "! one two\n! three\nx = 1\n"
        self.assertEqual(self.texts(fortran, "  ! one two\n  ! three\nx = 1\n", Lang.FORTRAN), [])
        self.assertEqual(self.texts(fortran, "! one two\nx = 1\n! three\n", Lang.FORTRAN), [])

    def test_deleting_hash_comment_lines_adds_nothing(self):
        param = "INT a \"a\"\n{\n} 1\n"
        before = "# one\n# two\n# three\n" + param
        self.assertEqual(self.texts(before, "# one\n# three\n" + param, Lang.PARAM), [])
        self.assertEqual(self.texts(before, "# one\n# thre\n" + param, Lang.PARAM), [(2, "# thre")])
        fortran = "! one\n! two\n! three\nx = 1\n"
        self.assertEqual(self.texts(fortran, "! two\n! three\nx = 1\n", Lang.FORTRAN), [])
        self.assertEqual(self.texts(fortran, "! three\n! one\nx = 1\n", Lang.FORTRAN), [])


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

    def noted(self, output):
        self.assertTrue(output, "the hook printed nothing")
        decision = json.loads(output)["hookSpecificOutput"]
        self.assertEqual(sorted(decision), ["additionalContext", "hookEventName"])
        self.assertEqual(decision["hookEventName"], "PreToolUse")
        return decision["additionalContext"].split("\n")


class HookEdit(Project):
    def test_denies_an_added_comment(self):
        lines = self.denied(self.edit("Thorn/src/a.cxx", "int a = 1;\n", "int a = 1; // why\n"))
        self.assertEqual(lines[0], "Thorn/src/a.cxx:2   // why")
        self.assertEqual(lines[-1], "✗ no-comments: 1 comment line")

    def test_allows_an_edit_without_a_new_comment(self):
        self.assertEqual(self.edit("Thorn/src/a.cxx", "int a = 1;\n", "int a = 2;\n"), "")
        self.assertEqual(
            self.edit("Thorn/src/a.cxx", "} // namespace Thorn\n", "} // namespace Thorn\nnamespace B {\n} // namespace B\n"),
            "",
        )

    def test_an_edit_that_deletes_a_line_of_a_comment_is_allowed_with_a_note(self):
        write_files(self.root, {"Thorn/src/c.cxx": "// one\n// two\n// three\nint c;\n"})
        self.assertEqual(self.noted(self.edit("Thorn/src/c.cxx", "// two\n", ""))[1], "  // two")
        self.assertEqual(self.noted(self.edit("Thorn/src/c.cxx", "// one\n// two\n", "// one\n"))[1], "  // two")
        self.assertEqual(self.edit("Thorn/src/c.cxx", "// one\n// two\n// three\n", ""), "")
        self.denied(self.edit("Thorn/src/c.cxx", "// two\n", "// too\n"))

    def test_the_note_lists_the_deleted_lines(self):
        text = "// `tl = -1` syncs every timelevel\n// except the oldest\n// and the one before.\nint f(int tl = -1);\n"
        write_files(self.root, {"Thorn/src/c.cxx": text})
        lines = self.noted(self.edit("Thorn/src/c.cxx", "// except the oldest\n// and the one before.\n", ""))
        self.assertEqual(lines, [trim_head("Thorn/src/c.cxx", 1), "  // except the oldest", "  // and the one before."]
                         + TRIM_RULE.split("\n"))

    def test_a_denied_edit_carries_no_note(self):
        write_files(self.root, {"Thorn/src/c.cxx": "// one\n// two\nint c;\n"})
        output = self.edit("Thorn/src/c.cxx", "// two\nint c;\n", "int c; // why\n")
        self.denied(output)
        self.assertNotIn("additionalContext", json.loads(output)["hookSpecificOutput"])

    def test_a_write_that_trims_a_comment_is_noted(self):
        write_files(self.root, {"Thorn/param.ccl": "# The grid spacing\n# on the coarsest level\nREAL dx \"dx\"\n{\n} 1\n"})
        lines = self.noted(self.write("Thorn/param.ccl", "# The grid spacing\nREAL dx \"dx\"\n{\n} 1\n"))
        self.assertEqual(lines[:2], [trim_head("Thorn/param.ccl", 1), "  # on the coarsest level"])

    def test_an_edit_to_the_code_between_comments_is_allowed(self):
        write_files(self.root, {"Thorn/src/c.cxx": "// one\nsync();\n// two\napply();\n"})
        self.assertEqual(self.edit("Thorn/src/c.cxx", "sync();\n", ""), "")
        self.assertEqual(self.edit("Thorn/src/c.cxx", "// two\n", "// two\nprepare();\n"), "")

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
        with mock.patch("rules.lex", side_effect=RuntimeError("lexer crash")):
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
        write_files(self.root, {"Thorn/param.ccl": "# old\nINT n \"n\"\n{\n} 1\n"})
        lines = self.denied(self.edit("Thorn/param.ccl", "} 1\n", "} 1 # why\n"))
        self.assertEqual(lines[0], "Thorn/param.ccl:4   # why")
        self.assertEqual(self.edit("Thorn/param.ccl", "} 1\n", "} 2\n"), "")
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


STOP_FILES = {
    "Thorn/interface.ccl": "IMPLEMENTS: Thorn\n",
    "Thorn/param.ccl": "# old prose\nINT n \"n\"\n{\n  *:* :: \"\"\n} 1\n",
    "Thorn/src/a.cxx": "namespace Thorn {\nint a = 1; // old\n} // namespace Thorn\n",
    "Thorn/src/make.code.defn": "SRCS = a.cxx\n",
    "Thorn/test/t.par": "Thorn::n = 2\n",
    "Thorn/README": "not covered\n",
}


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

    def append(self, name, text):
        with open(os.path.join(self.root, name), "a") as f:
            f.write(text)


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
        self.commit({"Thorn/param.ccl": STOP_FILES["Thorn/param.ccl"].replace("} 1", "} 2")})
        self.assertEqual(self.run_hook(), "")

    def test_a_comment_written_to_the_working_tree_blocks(self):
        self.append("Thorn/param.ccl", "# x\n")
        lines = self.blocked(self.run_hook())
        self.assertEqual(lines[0], "Thorn/param.ccl:6   # x")
        self.assertEqual(lines[-1], "✗ no-comments: 1 comment line")

    def test_stop_hook_active_allows_the_stop(self):
        self.append("Thorn/param.ccl", "# x\n")
        self.assertEqual(self.run_hook({"hook_event_name": "Stop", "stop_hook_active": True}), "")
        self.assertEqual(self.run_hook({"hook_event_name": "SubagentStop", "stop_hook_active": True}), "")
        self.blocked(self.run_hook({"hook_event_name": "SubagentStop", "stop_hook_active": False}))

    def test_a_comment_committed_on_the_branch_blocks(self):
        self.commit({"Thorn/test/t.par": "Thorn::n = 2 # why\n"})
        self.assertEqual(self.blocked(self.run_hook())[0], "Thorn/test/t.par:1   # why")

    def test_every_kind_of_file_is_checked(self):
        self.append("Thorn/src/a.cxx", "#if 0\n#endif\n")
        self.append("Thorn/src/make.code.defn", "# m\n")
        write_files(self.root, {"Thorn/src/new.cxx": "int n; // n\n", "Thorn/schedule.ccl": "/* s */\n"})
        lines = self.blocked(self.run_hook())
        self.assertEqual(lines[:4], ["Thorn/schedule.ccl:1         /* s */", "Thorn/src/a.cxx:4            #if 0",
                                     "Thorn/src/make.code.defn:2   # m", "Thorn/src/new.cxx:1          // n"])

    def test_comments_that_dev_already_has_are_not_reported(self):
        self.append("Thorn/src/a.cxx", "int b = 2;\n")
        self.commit({"Thorn/param.ccl": STOP_FILES["Thorn/param.ccl"] + "INT m \"m\"\n{\n} 1\n"})
        self.assertEqual(self.run_hook(), "")

    def test_a_renamed_file_keeps_its_old_comments(self):
        git(self.root, "mv", "Thorn/param.ccl", "Thorn/test/test.ccl")
        self.assertEqual(self.run_hook(), "")
        self.append("Thorn/test/test.ccl", "# new\n")
        self.assertEqual(self.blocked(self.run_hook())[0], "Thorn/test/test.ccl:6   # new")

    def test_comments_merged_in_from_another_branch_are_not_reported(self):
        self.merge_upstream({
            "Thorn/param.ccl": STOP_FILES["Thorn/param.ccl"] + "# upstream\n",
            "Thorn/schedule.ccl": "# upstream schedule\n",
        })
        self.assertEqual(self.run_hook(), "")

    def test_the_merge_resolution_itself_is_judged(self):
        self.merge_upstream({"Thorn/param.ccl": STOP_FILES["Thorn/param.ccl"] + "# upstream\n"})
        with open(os.path.join(self.root, "Thorn/param.ccl")) as f:
            merged = f.read()
        write_files(self.root, {"Thorn/param.ccl": merged.replace("} 1\n", "} 1 # resolution\n")})
        git(self.root, "commit", "-q", "--amend", "--no-edit", "-a")
        lines = self.blocked(self.run_hook())
        self.assertEqual(lines[0], "Thorn/param.ccl:5   # resolution")
        self.assertEqual(lines[-1], "✗ no-comments: 1 comment line")

    def test_a_merge_on_the_branch_s_own_history_is_walked_first_parent(self):
        git(self.root, "switch", "-q", "-c", "side")
        self.commit({"Thorn/test/t.par": "Thorn::n = 2 # side\n"})
        git(self.root, "switch", "-q", "work")
        self.commit({"Thorn/param.ccl": STOP_FILES["Thorn/param.ccl"].replace("} 1\n", "} 1 # work\n")})
        self.merge("side")
        lines = self.blocked(self.run_hook())
        self.assertEqual(lines[0], "Thorn/param.ccl:5   # work")
        self.assertEqual(lines[-1], "✗ no-comments: 1 comment line")

    def test_a_merged_in_comment_is_found_under_the_file_s_old_name(self):
        git(self.root, "switch", "-q", "-c", "upstream", "dev")
        self.commit({"Thorn/param.ccl": STOP_FILES["Thorn/param.ccl"] + "# upstream\n"})
        git(self.root, "switch", "-q", "work")
        git(self.root, "mv", "Thorn/param.ccl", "Thorn/test/test.ccl")
        git(self.root, "commit", "-q", "-m", "rename")
        self.merge("upstream")
        with open(os.path.join(self.root, "Thorn/test/test.ccl")) as f:
            self.assertIn("# upstream\n", f.read())
        self.assertEqual(self.run_hook(), "")

    def test_a_comment_moved_to_another_file_is_not_reported(self):
        self.commit({
            "Thorn/param.ccl": STOP_FILES["Thorn/param.ccl"].replace("# old prose\n", ""),
            "Thorn/schedule.ccl": "# old prose\n",
        })
        self.assertEqual(self.run_hook(), "")

    def test_a_copy_is_reported_where_it_was_written(self):
        self.append("Thorn/interface.ccl", "# old prose\n")
        self.commit({"Thorn/param.ccl": STOP_FILES["Thorn/param.ccl"].replace("} 1", "} 2")})
        lines = self.blocked(self.run_hook())
        self.assertEqual(lines[0], "Thorn/interface.ccl:2   # old prose")
        self.assertEqual(lines[-1], "✗ no-comments: 1 comment line")

    def test_a_renamed_and_rewritten_file_keeps_its_old_comments(self):
        git(self.root, "mv", "Thorn/param.ccl", "Thorn/test/test.ccl")
        write_files(self.root, {"Thorn/test/test.ccl": "# old prose\nTEST t\n{\n  NPROCS 2\n}\n"})
        self.assertEqual(self.run_hook(), "")

    def test_a_file_merged_in_and_then_renamed_keeps_its_comments(self):
        self.merge_upstream({"Thorn/schedule.ccl": "# upstream schedule\n"})
        git(self.root, "mv", "Thorn/schedule.ccl", "Thorn/test/test.ccl")
        git(self.root, "commit", "-q", "-m", "rename")
        self.assertEqual(self.run_hook(), "")

    def test_a_renamed_thorn_keeps_its_comments(self):
        git(self.root, "mv", "Thorn", "Renamed")
        self.assertEqual(self.run_hook(), "")
        self.append("Renamed/param.ccl", "# new\n")
        self.assertEqual(self.blocked(self.run_hook())[0], "Renamed/param.ccl:6   # new")

    def test_files_that_are_not_covered_or_are_gone_are_skipped(self):
        self.append("Thorn/README", "# not covered\n")
        write_files(self.root, {"scripts/x.par": "# not a thorn\n", "Other/param.ccl": "# no interface.ccl\n"})
        os.remove(os.path.join(self.root, "Thorn/test/t.par"))
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
        self.assertEqual(self.blocked(self.run_hook())[0], "Thorn/test/t.par:2   # x")

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
        before = None
        if os.path.isfile(path):
            with open(path) as f:
                before = f.read()
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

    def test_a_comment_moved_by_two_edits_is_allowed(self):
        self.assertEqual(self.edit("Thorn/param.ccl", "# old prose\n", ""), "")
        self.assertEqual(self.edit("Thorn/param.ccl", "} 1\n", "} 1\n# old prose\n"), "")
        self.assertEqual(self.stop(), "")

    def test_a_comment_moved_to_another_file_by_two_edits_is_allowed(self):
        self.assertEqual(self.edit("Thorn/param.ccl", "# old prose\n", ""), "")
        self.assertEqual(self.write("Thorn/schedule.ccl", "# old prose\n"), "")
        self.assertEqual(self.stop(), "")

    def test_a_copy_is_denied(self):
        lines = self.denied(self.write("Thorn/schedule.ccl", "# old prose\n"))
        self.assertEqual(lines[0], "Thorn/schedule.ccl:1   # old prose")
        self.assertEqual(lines[-1], "✗ no-comments: 1 comment line")

    def test_a_comment_moved_away_and_then_put_back_is_a_copy(self):
        write_files(self.root, {
            "Thorn/param.ccl": STOP_FILES["Thorn/param.ccl"].replace("# old prose\n", ""),
            "Thorn/schedule.ccl": "# old prose\n",
        })
        self.assertEqual(self.stop(), "")
        lines = self.denied(self.edit("Thorn/param.ccl", "} 1\n", "} 1\n# old prose\n"))
        self.assertEqual(lines[0], "Thorn/param.ccl:5   # old prose")
        self.assertEqual(lines[-1], "✗ no-comments: 1 comment line")

    def test_a_trimmed_comment_is_noted_at_the_edit_and_not_at_the_stop(self):
        git(self.root, "switch", "-q", "dev")
        self.commit({"Thorn/param.ccl": "# The grid spacing\n# on the coarsest level\n" + STOP_FILES["Thorn/param.ccl"]})
        git(self.root, "switch", "-q", "work")
        git(self.root, "merge", "-q", "--ff-only", "dev")
        output = self.edit("Thorn/param.ccl", "# on the coarsest level\n", "")
        note = json.loads(output)["hookSpecificOutput"]["additionalContext"].split("\n")
        self.assertEqual(note[:2], [trim_head("Thorn/param.ccl", 1), "  # on the coarsest level"])
        with open(os.path.join(self.root, "Thorn/param.ccl")) as f:
            self.assertNotIn("coarsest", f.read())
        self.assertEqual(self.stop(), "")

    def test_a_comment_the_branch_deleted_can_be_put_back(self):
        self.commit({"Thorn/param.ccl": STOP_FILES["Thorn/param.ccl"].replace("# old prose\n", "")})
        self.assertEqual(self.edit("Thorn/param.ccl", "} 1\n", "} 1\n# old prose\n"), "")

    def test_the_report_names_only_the_edit_s_own_lines(self):
        write_files(self.root, {"Thorn/schedule.ccl": "# written by a shell command\n"})
        lines = self.denied(self.edit("Thorn/param.ccl", "} 1\n", "} 1\n# written by a shell command\n# new\n"))
        self.assertEqual(lines[:2], ["Thorn/param.ccl:6   # written by a shell command", "Thorn/param.ccl:7   # new"])
        self.assertEqual(lines[-1], "✗ no-comments: 2 comment lines")

    def test_without_the_branch_the_edit_is_judged_alone(self):
        self.assertEqual(self.edit("Thorn/param.ccl", "# old prose\n", ""), "")
        with mock.patch("hooks.merge_base", side_effect=RuntimeError("git crashed")):
            lines = self.denied(self.edit("Thorn/param.ccl", "} 1\n", "} 1\n# old prose\n"))
        self.assertEqual(lines[0], "Thorn/param.ccl:5   # old prose")


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
    return [(entry.get("matcher"), hook["command"]) for entry in entries for hook in entry["hooks"]]


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

    def test_old_comments_pass_and_added_ones_fail(self):
        result = self.run_tool("check", "--since", self.base)
        self.assertEqual((result.returncode, result.stdout), (0, "✓ no-comments: 0 comment lines\n"), result.stderr)
        self.append("Thorn/param.ccl", "# x\n")
        self.append("Thorn/test/t.par", "Thorn::n = 3\n")
        result = self.run_tool("check", "--since", self.base)
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertEqual(result.stdout.split("\n")[0], "Thorn/param.ccl:6   # x")
        self.assertEqual(result.stdout.rstrip("\n").split("\n")[-1], "✗ no-comments: 1 comment line")

    def test_a_comment_committed_since_the_base_fails(self):
        self.append("Thorn/test/t.par", "# why\n")
        git(self.root, "commit", "-q", "-am", "add a comment")
        result = self.run_tool("check", "--since", self.base)
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertEqual(result.stdout.split("\n")[0], "Thorn/test/t.par:2   # why")
        self.assertEqual(self.run_tool("check", "--since", "HEAD").returncode, 0)

    def test_a_merged_in_comment_passes_and_a_resolution_comment_fails(self):
        git(self.root, "switch", "-q", "-c", "upstream", self.base)
        self.append("Thorn/param.ccl", "# upstream\n")
        git(self.root, "commit", "-q", "-am", "upstream")
        git(self.root, "switch", "-q", "dev")
        git(self.root, "merge", "--no-ff", "-q", "-m", "m", "upstream")
        result = self.run_tool("check", "--since", self.base)
        self.assertEqual((result.returncode, result.stdout), (0, "✓ no-comments: 0 comment lines\n"), result.stderr)
        self.append("Thorn/test/t.par", "Thorn::n = 3 # resolution\n")
        git(self.root, "commit", "-q", "--amend", "--no-edit", "-a")
        result = self.run_tool("check", "--since", self.base)
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertEqual(result.stdout.split("\n")[0], "Thorn/test/t.par:2   # resolution")
        self.assertEqual(result.stdout.rstrip("\n").split("\n")[-1], "✗ no-comments: 1 comment line")

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
            {
                "Thorn/test/u.par": "Thorn::n = 4\n",
                "Thorn/param.ccl": STOP_FILES["Thorn/param.ccl"].replace("# old prose", "# reworded prose"),
            },
        )
        git(self.root, "commit", "-q", "-am", "dev deletes one comment and rewords another")
        git(self.root, "switch", "-q", "work")
        result = self.run_tool("check", "--since", "dev")
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertEqual(result.stdout.split("\n")[0], "Thorn/test/t.par:2   # why")
        self.assertEqual(result.stdout.rstrip("\n").split("\n")[-1], "✗ no-comments: 1 comment line")

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
