from __future__ import annotations

import os
import subprocess
import tempfile
import unittest

from rules import Finding, findings, new_findings, render, since_findings, summary
from files import Lang

ISOLATED_GIT = {"GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"}
GUARD = '#ifndef CARPETX_INTERNAL\n#error "CarpetX internal header; include <carpetx.hxx> instead"\n#endif\n'


def reported(name, text, lang=Lang.CXX):
    return [(f.rule, f.text) for f in findings(name, text, lang)]


class Includes(unittest.TestCase):
    def test_paths_into_carpetx_src_are_reported_outside_carpetx(self):
        for line in [
            '#include "../../CarpetX/src/driver.hxx"',
            '#include "../../../CarpetX/CarpetX/src/schedule.hxx"',
            '#include "CarpetX/CarpetX/src/subcycling.hxx"',
            "#include <CarpetX/CarpetX/src/driver.hxx>",
            '  #  include   "../../CarpetX/src/timer.hxx"',
        ]:
            with self.subTest(line=line):
                self.assertEqual(reported("ODESolvers/src/solve.hxx", line + "\n"), [("include", line.strip())])

    def test_bare_names_and_other_thorns_are_fine(self):
        text = '#include <carpetx.hxx>\n#include <driver.hxx>\n#include "../../Loop/src/loop.hxx"\n#include "solve.hxx"\n'
        self.assertEqual(reported("ODESolvers/src/solve.hxx", text), [])

    def test_mentions_outside_an_include_are_not_includes(self):
        text = 'const char *p = "CarpetX/src/driver.hxx";\n'
        self.assertEqual(reported("T/src/a.cxx", text), [])

    def test_only_c_and_cxx_files_hold_includes(self):
        self.assertEqual(reported("T/src/a.F90", '#include "../../CarpetX/src/driver.hxx"\n', Lang.FORTRAN), [])
        self.assertEqual(reported("T/src/make.code.defn", '#include "../../CarpetX/src/driver.hxx"\n', Lang.MAKE), [])
        self.assertEqual(reported("T/interface.ccl", '#include "../../CarpetX/src/driver.hxx"\n', Lang.INTERFACE), [])

    def test_carpetx_own_sources_may_include_siblings_by_any_path(self):
        self.assertEqual(reported("CarpetX/src/driver.cxx", '#include "../src/driver.hxx"\n'), [])

    def test_line_numbers(self):
        text = 'int a;\n#include "../../CarpetX/src/driver.hxx"\n'
        self.assertEqual(findings("T/src/a.cxx", text, Lang.CXX), [Finding(2, "include", '#include "../../CarpetX/src/driver.hxx"')])


class Markers(unittest.TestCase):
    def test_the_marker_is_reported_outside_carpetx_src(self):
        for name, lang in [("T/src/a.cxx", Lang.CXX), ("T/src/make.code.defn", Lang.MAKE), ("CarpetX/interface.ccl", Lang.INTERFACE)]:
            text = "CPPFLAGS += -DCARPETX_INTERNAL\n" if lang is Lang.MAKE else "#define CARPETX_INTERNAL 1\n"
            with self.subTest(name=name):
                expected = [("marker", text.strip())] if lang in (Lang.CXX, Lang.MAKE) else []
                self.assertEqual(reported(name, text, lang), expected)

    def test_a_longer_name_is_not_the_marker(self):
        self.assertEqual(reported("T/src/a.cxx", "#define CARPETX_INTERNAL_X 1\n"), [])

    def test_carpetx_make_code_defn_defines_it(self):
        self.assertEqual(reported("CarpetX/src/make.code.defn", "SRCS = a.cxx\n\nCPPFLAGS += -DCARPETX_INTERNAL\n", Lang.MAKE), [])

    def test_a_private_header_guards_on_it(self):
        self.assertEqual(reported("CarpetX/src/driver.hxx", GUARD), [])
        self.assertEqual(reported("CarpetX/src/driver.hxx", "  # ifndef  CARPETX_INTERNAL  \n#endif\n"), [])

    def test_any_other_use_inside_carpetx_src_is_reported(self):
        for text in ["#define CARPETX_INTERNAL\n", "#ifdef CARPETX_INTERNAL\n", "#if !defined(CARPETX_INTERNAL)\n", "#ifndef CARPETX_INTERNAL // x\n"]:
            with self.subTest(text=text):
                self.assertEqual(reported("CarpetX/src/driver.cxx", text), [("marker", text.strip())])

    def test_the_public_headers_never_name_it(self):
        self.assertEqual(reported("CarpetX/src/include/carpetx.hxx", GUARD), [("marker", "#ifndef CARPETX_INTERNAL")])

    def test_another_thorns_make_code_defn_cannot_set_it(self):
        self.assertEqual(reported("T/src/make.code.defn", "CPPFLAGS += -DCARPETX_INTERNAL\n", Lang.MAKE), [("marker", "CPPFLAGS += -DCARPETX_INTERNAL")])


class Validity(unittest.TestCase):
    KERNELS = [
        "error_if_invalid(groupdata, vi, tl, make_valid_int(), msg);",
        "warn_if_invalid(groupdata, vi, tl, make_valid_int(), msg);",
        "poison_invalid_gf(active_levels, gi, vi, tl);",
        "check_valid_gf (active_levels, gi, vi, tl, msg);",
        "poison_invalid_ga(gi, vi, tl);",
        "check_valid_ga(gi, vi, tl, msg);",
    ]
    SETTERS = [
        "groupdata.timelevel(0).why_valid(0).set_all(valid_t(), why);",
        "why.set_int(false, reason);",
        "why.set_outer(true, reason);",
        "why.set_ghosts(true, reason);",
        "timelevel->valid_.at(vi).set_invalid(destroys, not_filled);",
        "p->set_valid (fills, filled);",
    ]

    def test_kernels_and_member_setters_are_reported_in_carpetx_src(self):
        for line in self.KERNELS + self.SETTERS:
            for name in ["CarpetX/src/sync_restrict.cxx", "CarpetX/src/api.cxx", "CarpetX/src/driver.hxx"]:
                with self.subTest(name=name, line=line):
                    self.assertEqual(reported(name, "  " + line + "\n"), [("validity", line)])

    def test_the_validity_module_defines_them(self):
        for line in self.KERNELS + self.SETTERS:
            for name in ["CarpetX/src/valid.cxx", "CarpetX/src/valid.hxx"]:
                with self.subTest(name=name, line=line):
                    self.assertEqual(reported(name, line + "\n"), [])

    def test_another_thorn_may_use_the_names(self):
        for line in self.KERNELS + self.SETTERS:
            with self.subTest(line=line):
                self.assertEqual(reported("ODESolvers/src/solve.cxx", line + "\n"), [])

    def test_the_public_headers_are_outside_the_private_tree(self):
        self.assertEqual(reported("CarpetX/src/include/carpetx.hxx", self.SETTERS[0] + "\n"), [])

    def test_a_setter_is_only_a_member_call(self):
        for text in [
            "set_all(flags, why);\n",
            "void set_all(const valid_t &val);\n",
            "const auto set_valid = make_valid_all();\n",
            "x.set_all_flags(true);\n",
            "x.set(which, val, why);\n",
            "x.reset_all(true);\n",
        ]:
            with self.subTest(text=text):
                self.assertEqual(reported("CarpetX/src/driver.cxx", text), [])

    def test_a_longer_name_is_not_a_kernel(self):
        for text in ["check_valid_gfs(a);\n", "my_error_if_invalid(a);\n", "error_if_invalid_x(a);\n", "int error_if_invalid;\n"]:
            with self.subTest(text=text):
                self.assertEqual(reported("CarpetX/src/driver.cxx", text), [])

    def test_only_c_and_cxx_files_are_checked(self):
        self.assertEqual(reported("CarpetX/src/make.code.defn", "SRCS = check_valid_gf(a)\n", Lang.MAKE), [])

    def test_line_numbers(self):
        text = "int a;\n\nposion;\nwhy.set_all(valid_t(), r);\n"
        self.assertEqual(findings("CarpetX/src/driver.cxx", text, Lang.CXX), [Finding(4, "validity", "why.set_all(valid_t(), r);")])


class Directions(unittest.TestCase):
    FORWARDED = (
        "int SyncGroupsByDirI(const active_levels_t &active_levels,\n"
        "                     const cGH *restrict cctkGH, int numgroups,\n"
        "                     const int *groups, const int *directions);\n"
    )
    FLESH_OVERLOAD = (
        "int SyncGroupsOnActiveLevels(const cGH *restrict cctkGH, int numgroups,\n"
        "                             const int *groups, const int *directions);\n"
    )
    FLESH_DEFINITION = (
        "int SyncGroupsOnActiveLevels(const cGH *restrict const cctkGH,\n"
        "                             const int numgroups, const int *const groups,\n"
        "                             const int *const directions) {\n"
        "  assert(active_levels);\n"
        "  const std::vector<gi_t> gis(groups, groups + numgroups);\n"
        "  return SyncGroups(*active_levels, cctkGH, gis);\n"
        "}\n"
    )

    def test_directions_reported(self):
        for name in ["CarpetX/src/schedule.hxx", "CarpetX/src/sync_restrict.cxx", "CarpetX/src/sync_restrict_internal.hxx"]:
            with self.subTest(name=name):
                self.assertEqual(
                    reported(name, self.FORWARDED),
                    [("directions", "const int *groups, const int *directions);")],
                )
        text = "  return SyncGroups(*active_levels, cctkGH, numgroups, groups, directions);\n"
        self.assertEqual(reported("CarpetX/src/schedule.cxx", text), [("directions", text.strip())])

    def test_directions_allowed_in_flesh_overload(self):
        self.assertEqual(reported("CarpetX/src/schedule.hxx", self.FLESH_OVERLOAD), [])
        self.assertEqual(reported("CarpetX/src/schedule.cxx", self.FLESH_DEFINITION), [])
        self.assertEqual(reported("CarpetX/src/schedule.hxx", self.FLESH_OVERLOAD + "\n" + self.FORWARDED), [("directions", "const int *groups, const int *directions);")])

    def test_other_files_are_not_the_sync_layer(self):
        for name in ["CarpetX/src/io_meta.cxx", "CarpetX/src/driver.hxx", "ODESolvers/src/solve.cxx"]:
            with self.subTest(name=name):
                self.assertEqual(reported(name, self.FORWARDED), [])

    def test_a_longer_name_is_not_the_parameter(self):
        self.assertEqual(reported("CarpetX/src/sync_restrict.cxx", "std::vector<int> output_directions;\nint all_directions_done;\n"), [])

    def test_line_numbers(self):
        text = "int a;\n" + self.FORWARDED
        self.assertEqual(findings("CarpetX/src/schedule.hxx", text, Lang.CXX), [Finding(4, "directions", "const int *groups, const int *directions);")])


class NewFindings(unittest.TestCase):
    def test_only_lines_absent_before_are_new(self):
        a = Finding(1, "include", '#include "../../CarpetX/src/driver.hxx"')
        b = Finding(2, "include", '#include "../../CarpetX/src/schedule.hxx"')
        moved = Finding(9, "include", a.text)
        self.assertEqual(new_findings([a], [moved, b]), [b])
        self.assertEqual(new_findings([a, b], [b]), [])
        self.assertEqual(new_findings([], [a]), [a])

    def test_a_duplicated_line_counts(self):
        a = Finding(1, "include", '#include "../../CarpetX/src/driver.hxx"')
        self.assertEqual(new_findings([a], [a, Finding(2, "include", a.text)]), [Finding(2, "include", a.text)])


def git(root, *args):
    env = dict(os.environ, **ISOLATED_GIT)
    return subprocess.run(
        ["git", "-C", root, "-c", "user.name=Test", "-c", "user.email=test@example.org", *args],
        check=True, capture_output=True, text=True, env=env,
    ).stdout


def write(root, name, text):
    path = os.path.join(root, name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)


class Since(unittest.TestCase):
    def test_reports_what_the_tree_added_since_the_base(self):
        with tempfile.TemporaryDirectory() as root:
            git(root, "init", "-q")
            write(root, "T/interface.ccl", "IMPLEMENTS: T\n")
            write(root, "T/src/a.cxx", '#include "../../CarpetX/src/driver.hxx"\n')
            write(root, "CarpetX/interface.ccl", "IMPLEMENTS: Driver\n")
            write(root, "CarpetX/src/io.hxx", "int a;\n")
            git(root, "add", "-A")
            git(root, "commit", "-q", "-m", "base")
            base = git(root, "rev-parse", "HEAD").strip()
            write(root, "T/src/a.cxx", '#include "../../CarpetX/src/driver.hxx"\n#include "../../CarpetX/src/schedule.hxx"\n')
            write(root, "T/src/b.cxx", "#define CARPETX_INTERNAL\n")
            write(root, "CarpetX/src/io.hxx", GUARD + "int a;\n")
            write(root, "notes.md", '#include "../../CarpetX/src/driver.hxx"\n')
            self.assertEqual(
                since_findings(root, base),
                [
                    ("T/src/a.cxx", Finding(2, "include", '#include "../../CarpetX/src/schedule.hxx"')),
                    ("T/src/b.cxx", Finding(1, "marker", "#define CARPETX_INTERNAL")),
                ],
            )

    def test_a_clean_tree_has_nothing_to_report(self):
        with tempfile.TemporaryDirectory() as root:
            git(root, "init", "-q")
            write(root, "T/interface.ccl", "IMPLEMENTS: T\n")
            write(root, "T/src/a.cxx", '#include "../../CarpetX/src/driver.hxx"\n')
            git(root, "add", "-A")
            git(root, "commit", "-q", "-m", "base")
            base = git(root, "rev-parse", "HEAD").strip()
            write(root, "T/src/a.cxx", '#include "../../CarpetX/src/driver.hxx"\nint a;\n')
            self.assertEqual(since_findings(root, base), [])


class Report(unittest.TestCase):
    def test_clean(self):
        self.assertEqual(
            render([]),
            "✓ boundary: no private CarpetX includes, no stray CARPETX_INTERNAL, "
            "no validity call outside valid.cxx, no directions in the sync layer",
        )

    def test_counts(self):
        a = ("T/src/a.cxx", Finding(3, "include", '#include "../../CarpetX/src/driver.hxx"'))
        b = ("T/src/b.cxx", Finding(1, "marker", "#define CARPETX_INTERNAL"))
        c = ("CarpetX/src/driver.cxx", Finding(7, "validity", "why.set_all(valid_t(), r);"))
        d = ("CarpetX/src/schedule.hxx", Finding(2, "directions", "const int *groups, const int *directions);"))
        self.assertEqual(summary([a]), "✗ boundary: 1 private include")
        self.assertEqual(summary([a, a, b]), "✗ boundary: 2 private includes, 1 stray marker")
        self.assertEqual(summary([c]), "✗ boundary: 1 validity call outside valid.cxx")
        self.assertEqual(summary([b, c, c]), "✗ boundary: 1 stray marker, 2 validity calls outside valid.cxx")
        self.assertEqual(summary([d]), "✗ boundary: 1 directions parameter in the sync layer")
        self.assertEqual(summary([c, d, d]), "✗ boundary: 1 validity call outside valid.cxx, 2 directions parameters in the sync layer")

    def test_the_report_shape(self):
        a = ("T/src/a.cxx", Finding(3, "include", '#include "../../CarpetX/src/driver.hxx"'))
        b = ("T/src/b.cxx", Finding(1, "marker", "#define CARPETX_INTERNAL"))
        c = ("CarpetX/src/driver.cxx", Finding(7, "validity", "why.set_all(valid_t(), r);"))
        lines = render([b, c, a]).split("\n")
        self.assertEqual(lines[0], "CarpetX/src/driver.cxx:7   validity   why.set_all(valid_t(), r);")
        self.assertEqual(lines[1], 'T/src/a.cxx:3              include    #include "../../CarpetX/src/driver.hxx"')
        self.assertEqual(lines[2], "T/src/b.cxx:1              marker     #define CARPETX_INTERNAL")
        self.assertEqual(
            lines[-1],
            "✗ boundary: 1 private include, 1 stray marker, 1 validity call outside valid.cxx",
        )


if __name__ == "__main__":
    unittest.main()
