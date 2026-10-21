from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from unittest import mock

import files
from files import (
    EMPTY_TREE,
    Lang,
    changes_since,
    common_ancestor,
    covered,
    lang_of,
    list_tree,
    merge_base,
    merged_tips,
    show,
    thorns_in,
    untracked,
)

ISOLATED_GIT = {"GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"}


def write_files(root, contents):
    for name, text in contents.items():
        path = os.path.join(root, name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(text)


def git(root, *args):
    env = dict(os.environ, **ISOLATED_GIT)
    return subprocess.run(
        ["git", "-C", root, "-c", "user.name=Test", "-c", "user.email=test@example.org", *args],
        check=True, capture_output=True, text=True, env=env,
    ).stdout


class LangOf(unittest.TestCase):
    def test_every_covered_kind(self):
        for name, lang in [
            ("T/src/a.cxx", Lang.CXX),
            ("T/src/sub/b.hxx", Lang.CXX),
            ("T/src/c.h", Lang.CXX),
            ("T/src/d.F90", Lang.FORTRAN),
            ("T/src/e.f90", Lang.FORTRAN),
            ("T/src/make.code.defn", Lang.MAKE),
            ("T/src/sub/make.code.defn", Lang.MAKE),
            ("T/interface.ccl", Lang.INTERFACE),
            ("T/param.ccl", Lang.PARAM),
            ("T/schedule.ccl", Lang.SCHEDULE),
            ("T/configuration.ccl", Lang.CONFIGURATION),
            ("T/test/test.ccl", Lang.TEST),
            ("T/test/tests.ccl", Lang.TEST),
            ("T/test/a.par", Lang.PAR),
            ("T/par/b.par", Lang.PAR),
            ("T/test/c.par-DISABLED", Lang.PAR),
        ]:
            with self.subTest(name=name):
                self.assertIs(lang_of(name), lang)

    def test_what_is_not_covered(self):
        for name in [
            "T/a.cxx",
            "T/README",
            "T/src/a.cxx.orig",
            "T/src/x.py",
            "T/make.code.defn",
            "T/src/make.code.deps",
            "T/test/a/out.tsv",
            "T/doc/notes.md",
            "a.ccl",
            "T/test/a.parfile",
        ]:
            with self.subTest(name=name):
                self.assertIsNone(lang_of(name))

    def test_a_thorn_is_a_top_level_directory_with_interface_ccl(self):
        self.assertEqual(thorns_in(["A/interface.ccl", "B/src/interface.ccl", "C/param.ccl", "interface.ccl"]), {"A"})


class Covered(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = os.path.realpath(tmp.name)
        write_files(self.root, {"Thorn/interface.ccl": "", "Other/param.ccl": "", "Thorn/src/a.cxx": ""})

    def test_only_files_of_a_thorn_are_covered(self):
        self.assertIs(covered("Thorn/src/a.cxx", self.root), Lang.CXX)
        self.assertIs(covered("Thorn/interface.ccl", self.root), Lang.INTERFACE)
        self.assertIsNone(covered("Other/param.ccl", self.root))
        self.assertIsNone(covered("scripts/x.par", self.root))

    def test_list_tree_without_git_walks_the_directory(self):
        with mock.patch("files.is_work_tree", return_value=False):
            self.assertEqual(
                [(name, lang) for name, _, lang in list_tree(self.root)],
                [("Thorn/interface.ccl", Lang.INTERFACE), ("Thorn/src/a.cxx", Lang.CXX)],
            )


class Git(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = os.path.realpath(tmp.name)
        patcher = mock.patch.dict(os.environ, ISOLATED_GIT)
        patcher.start()
        self.addCleanup(patcher.stop)
        git(self.root, "init", "-q", "-b", "dev")

    def commit(self, contents, message="c"):
        write_files(self.root, contents)
        git(self.root, "add", "-A")
        git(self.root, "commit", "-q", "-m", message)
        return git(self.root, "rev-parse", "HEAD").strip()

    def test_no_commits_compares_against_the_empty_tree(self):
        self.assertEqual(merge_base(self.root), EMPTY_TREE)
        self.assertEqual(show(self.root, EMPTY_TREE, "x"), "")

    def test_the_base_is_the_newest_commit_shared_with_dev_or_origin_dev(self):
        first = self.commit({"T/interface.ccl": ""})
        git(self.root, "switch", "-q", "-c", "work")
        self.assertEqual(merge_base(self.root), first)
        git(self.root, "switch", "-q", "dev")
        second = self.commit({"T/param.ccl": ""})
        git(self.root, "update-ref", "refs/remotes/origin/dev", second)
        git(self.root, "reset", "-q", "--hard", first)
        git(self.root, "switch", "-q", "work")
        git(self.root, "merge", "-q", "--ff-only", second)
        third = self.commit({"T/schedule.ccl": ""})
        self.assertEqual(merge_base(self.root), second)
        self.assertNotEqual(merge_base(self.root), third)

    def test_without_dev_the_base_is_head(self):
        head = self.commit({"T/interface.ccl": ""})
        git(self.root, "branch", "-q", "-m", "dev", "main")
        self.assertEqual(merge_base(self.root), head)

    def test_common_ancestor_is_where_head_left_rev_or_none_without_shared_history(self):
        first = self.commit({"T/a.ccl": ""})
        git(self.root, "switch", "-q", "-c", "work")
        second = self.commit({"T/b.ccl": ""})
        git(self.root, "switch", "-q", "dev")
        third = self.commit({"T/c.ccl": ""})
        git(self.root, "switch", "-q", "work")
        self.assertEqual(common_ancestor(self.root, first), first)
        self.assertEqual(common_ancestor(self.root, third), first)
        self.assertEqual(common_ancestor(self.root, second), second)
        unrelated = git(self.root, "commit-tree", "HEAD^{tree}", "-m", "unrelated").strip()
        self.assertIsNone(common_ancestor(self.root, unrelated))

    def test_changes_since_maps_each_changed_file_to_its_name_in_rev(self):
        base = self.commit({"T/a.ccl": "a\n", "T/b.ccl": "b\n", "T/c.ccl": "c\n" * 20, "T/gone.ccl": "g\n"})
        git(self.root, "switch", "-q", "-c", "work")
        self.commit({"T/a.ccl": "a2\n"})
        write_files(self.root, {"T/b.ccl": "b2\n", "T/new.ccl": "n\n"})
        git(self.root, "mv", "T/c.ccl", "T/moved.ccl")
        git(self.root, "add", "T/new.ccl")
        git(self.root, "rm", "-q", "T/gone.ccl")
        write_files(self.root, {"T/untracked.ccl": "u\n"})
        self.assertEqual(changes_since(self.root, base), {
            "T/a.ccl": "T/a.ccl",
            "T/b.ccl": "T/b.ccl",
            "T/moved.ccl": "T/c.ccl",
            "T/new.ccl": None,
            "T/untracked.ccl": None,
        })
        self.assertEqual(untracked(self.root), ["T/untracked.ccl"])
        self.assertEqual(show(self.root, base, "T/c.ccl"), "c\n" * 20)
        self.assertEqual(show(self.root, base, "T/nope.ccl"), "")
        self.assertEqual(show(self.root, base, None), "")

    def test_merged_tips_are_the_other_parents_of_each_first_parent_merge(self):
        self.assertEqual(merged_tips(self.root, EMPTY_TREE), [])
        base = self.commit({"T/a.ccl": "a\n"})
        git(self.root, "switch", "-q", "-c", "work")
        self.commit({"T/b.ccl": "b\n"})
        self.assertEqual(merged_tips(self.root, base), [])
        git(self.root, "switch", "-q", "-c", "x", base)
        self.commit({"T/x.ccl": "x\n"})
        git(self.root, "switch", "-q", "-c", "side", base)
        self.commit({"T/s.ccl": "s\n"})
        git(self.root, "merge", "--no-ff", "-q", "-m", "m", "x")
        side = git(self.root, "rev-parse", "HEAD").strip()
        git(self.root, "switch", "-q", "-c", "y", base)
        y = self.commit({"T/y.ccl": "y\n"})
        git(self.root, "switch", "-q", "work")
        git(self.root, "merge", "--no-ff", "-q", "-m", "m", "side", "y")
        self.assertEqual(merged_tips(self.root, base), [side, y])
        self.assertEqual(merged_tips(self.root, "HEAD"), [])

    def test_a_deleted_file_is_not_a_change_to_judge(self):
        base = self.commit({"T/a.ccl": "a\n", "T/b.ccl": "b\n"})
        os.remove(os.path.join(self.root, "T/a.ccl"))
        self.assertEqual(changes_since(self.root, base), {})

    def test_staged_files_differ_from_the_empty_tree(self):
        write_files(self.root, {"T/a.ccl": "a\n", "T/b.ccl": "b\n"})
        git(self.root, "add", "T/a.ccl")
        self.assertEqual(changes_since(self.root, EMPTY_TREE), {"T/a.ccl": None, "T/b.ccl": None})

    def test_list_tree_lists_tracked_and_untracked_files_but_not_ignored_ones(self):
        self.commit({"T/interface.ccl": "", "T/src/a.cxx": "", ".gitignore": "*.par\n"})
        write_files(self.root, {"T/src/b.cxx": "", "T/test/x.par": ""})
        self.assertEqual([name for name, _, _ in list_tree(self.root)], ["T/interface.ccl", "T/src/a.cxx", "T/src/b.cxx"])


class Parallel(unittest.TestCase):
    def test_keeps_the_order(self):
        self.assertEqual(files.parallel(lambda x: x * x, range(50)), [x * x for x in range(50)])
        self.assertEqual(files.parallel(lambda x: x, []), [])


if __name__ == "__main__":
    unittest.main()
