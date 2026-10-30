from __future__ import annotations

import io
import json
import os
import re
import shutil
import subprocess
import tempfile
import unittest
from collections import namedtuple
from unittest import mock

import diagram
import rules
from rules import HIERARCHY_LAYERS, LAYERS, MAKE_DEFN, RULE, Finding, below, findings, layer_of, layer_path, new_findings, render, since_findings, summary
from files import PROJECT, Lang
import hooks

TOOL = os.path.dirname(os.path.abspath(__file__))
SETTINGS = os.path.join(PROJECT, ".claude", "settings.json")
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
        text = 'const char *p = "CarpetX/src/hierarchy/driver.hxx";\n'
        self.assertEqual(reported("T/src/a.cxx", text), [])

    def test_only_c_and_cxx_files_hold_includes(self):
        self.assertEqual(reported("T/src/a.F90", '#include "../../CarpetX/src/driver.hxx"\n', Lang.FORTRAN), [])
        self.assertEqual(reported("T/src/make.code.defn", '#include "../../CarpetX/src/driver.hxx"\n', Lang.MAKE), [])
        self.assertEqual(reported("T/interface.ccl", '#include "../../CarpetX/src/driver.hxx"\n', Lang.INTERFACE), [])

    def test_carpetx_own_sources_are_left_to_the_layer_rule(self):
        self.assertEqual(reported("CarpetX/src/hierarchy/hierarchy.cxx", '#include "../src/driver.hxx"\n'), [("layer", '#include "../src/driver.hxx"')])

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

    EXPORT = "export EXTRA_DEFINES := $(sort $(EXTRA_DEFINES) CARPETX_INTERNAL)"

    def test_carpetx_make_code_defn_defines_it(self):
        self.assertEqual(reported("CarpetX/src/make.code.defn", "SRCS =\n\n" + self.EXPORT + "\n", Lang.MAKE), [])

    def test_a_layer_make_code_defn_does_not(self):
        for name in ["CarpetX/src/comm/make.code.defn", "CarpetX/src/hierarchy/make.code.defn"]:
            with self.subTest(name=name):
                self.assertEqual(reported(name, "SRCS = a.cxx\n" + self.EXPORT + "\n", Lang.MAKE), [("marker", self.EXPORT)])

    def test_a_private_header_guards_on_it(self):
        self.assertEqual(reported("CarpetX/src/hierarchy/driver.hxx", GUARD), [])
        self.assertEqual(reported("CarpetX/src/hierarchy/driver.hxx", "  # ifndef  CARPETX_INTERNAL  \n#endif\n"), [])

    def test_any_other_use_inside_carpetx_src_is_reported(self):
        for text in ["#define CARPETX_INTERNAL\n", "#ifdef CARPETX_INTERNAL\n", "#if !defined(CARPETX_INTERNAL)\n", "#ifndef CARPETX_INTERNAL // x\n"]:
            with self.subTest(text=text):
                self.assertEqual(reported("CarpetX/src/hierarchy/hierarchy.cxx", text), [("marker", text.strip())])

    def test_the_public_headers_never_name_it(self):
        self.assertEqual(reported("CarpetX/src/include/carpetx.hxx", GUARD), [("marker", "#ifndef CARPETX_INTERNAL")])

    def test_another_thorns_make_code_defn_cannot_set_it(self):
        self.assertEqual(reported("T/src/make.code.defn", "CPPFLAGS += -DCARPETX_INTERNAL\n", Lang.MAKE), [("marker", "CPPFLAGS += -DCARPETX_INTERNAL")])


class Validity(unittest.TestCase):
    KERNELS = [
        "error_if_invalid(groupdata, vi, tl, make_valid_int(), msg);",
        "warn_if_invalid(groupdata, vi, tl, make_valid_int(), msg);",
        "poison_invalid_gf(window, gi, vi, tl);",
        "check_valid_gf (window, gi, vi, tl, msg);",
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
            for name in ["CarpetX/src/comm/sync_restrict.cxx", "CarpetX/src/api/api.cxx", "CarpetX/src/hierarchy/driver.hxx", "CarpetX/src/comm/valid.cxx"]:
                with self.subTest(name=name, line=line):
                    self.assertEqual(reported(name, "  " + line + "\n"), [("validity", line)])

    def test_the_validity_module_defines_them(self):
        for line in self.KERNELS + self.SETTERS:
            for name in ["CarpetX/src/hierarchy/valid.cxx", "CarpetX/src/hierarchy/valid.hxx"]:
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
                self.assertEqual(reported("CarpetX/src/hierarchy/hierarchy.cxx", text), [])

    def test_a_longer_name_is_not_a_kernel(self):
        for text in ["check_valid_gfs(a);\n", "my_error_if_invalid(a);\n", "error_if_invalid_x(a);\n", "int error_if_invalid;\n"]:
            with self.subTest(text=text):
                self.assertEqual(reported("CarpetX/src/hierarchy/hierarchy.cxx", text), [])

    def test_only_c_and_cxx_files_are_checked(self):
        self.assertEqual(reported("CarpetX/src/make.code.defn", "SRCS = check_valid_gf(a)\n", Lang.MAKE), [])

    def test_line_numbers(self):
        text = "int a;\n\nposion;\nwhy.set_all(valid_t(), r);\n"
        self.assertEqual(findings("CarpetX/src/hierarchy/hierarchy.cxx", text, Lang.CXX), [Finding(4, "validity", "why.set_all(valid_t(), r);")])


class Directions(unittest.TestCase):
    FORWARDED = (
        "int SyncGroupsByDirI(const window_t window,\n"
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
        "  const std::vector<gi_t> gis(groups, groups + numgroups);\n"
        "  return SyncGroups(scheduled_window(sync_caller(gis).c_str()), cctkGH, gis);\n"
        "}\n"
    )

    def test_directions_reported(self):
        for name in ["CarpetX/src/comm/comm.hxx", "CarpetX/src/comm/sync_restrict.cxx"]:
            with self.subTest(name=name):
                self.assertEqual(
                    reported(name, self.FORWARDED),
                    [("directions", "const int *groups, const int *directions);")],
                )
        text = "  return SyncGroups(scheduled_window(sync_caller(gis).c_str()), cctkGH, gis, directions);\n"
        self.assertEqual(reported("CarpetX/src/comm/sync_restrict.cxx", text), [("directions", text.strip())])

    def test_directions_allowed_in_flesh_overload(self):
        self.assertEqual(reported("CarpetX/src/comm/comm.hxx", self.FLESH_OVERLOAD), [])
        self.assertEqual(reported("CarpetX/src/comm/sync_restrict.cxx", self.FLESH_DEFINITION), [])
        self.assertEqual(reported("CarpetX/src/comm/comm.hxx", self.FLESH_OVERLOAD + "\n" + self.FORWARDED), [("directions", "const int *groups, const int *directions);")])

    def test_other_files_are_not_the_sync_layer(self):
        for name in ["CarpetX/src/schedule/schedule.hxx", "CarpetX/src/schedule/evolve.cxx", "CarpetX/src/io/io_meta.cxx", "CarpetX/src/hierarchy/driver.hxx", "ODESolvers/src/solve.cxx"]:
            with self.subTest(name=name):
                self.assertEqual(reported(name, self.FORWARDED), [])

    def test_a_longer_name_is_not_the_parameter(self):
        self.assertEqual(reported("CarpetX/src/comm/sync_restrict.cxx", "std::vector<int> output_directions;\nint all_directions_done;\n"), [])

    def test_line_numbers(self):
        text = "int a;\n" + self.FORWARDED
        self.assertEqual(findings("CarpetX/src/comm/comm.hxx", text, Lang.CXX), [Finding(4, "directions", "const int *groups, const int *directions);")])


class VarIndex(unittest.TestCase):
    ADDITIONS = [
        "cctkGH->data[group.firstvarindex + vi][tl.index()] = nullptr;",
        "CCTK_FullVarName(groupdata.record.firstvarindex+vi),",
        "const int varindex = CCTK_FirstVarIndexI(gi.index()) + vi;",
        "CCTK_FullVarName(CCTK_FirstVarIndexI(wr.gi.index()) + wr.vi));",
        "varname = CCTK_VarName(CCTK_FirstVarIndexI (gi) + vi);",
        "return var_t{group.firstvarindex + vi.index()};",
    ]
    RECORD_VAR = (
        "    var_t var(const vi_t vi) const {\n"
        "      assert(variables().contains(vi));\n"
        "      return var_t{firstvarindex + vi.index()};\n"
        "    }\n"
    )

    def test_varindex_reported(self):
        for line in self.ADDITIONS:
            for name in ["CarpetX/src/hierarchy/valid.cxx", "CarpetX/src/io/io_silo.cxx", "CarpetX/src/hierarchy/driver.hxx", "CarpetX/src/include/carpetx.hxx"]:
                with self.subTest(name=name, line=line):
                    records = [("records", line)] if layer_of(name) in rules.RECORDS_LAYERS and "CCTK_FirstVarIndexI" in line else []
                    self.assertEqual(reported(name, "  " + line + "\n"), [("varindex", line)] + records)

    def test_varindex_allowed_in_record_var(self):
        self.assertEqual(reported("CarpetX/src/hierarchy/driver.hxx", self.RECORD_VAR), [])
        self.assertEqual(
            reported("CarpetX/src/schedule/evolve.cxx", self.RECORD_VAR),
            [("varindex", "return var_t{firstvarindex + vi.index()};")],
        )

    def test_other_arithmetic_is_not_an_addition(self):
        for text in [
            "firstvarindex = CCTK_FirstVarIndexI(gi.index());\n",
            "return variable_t{gi_t{gi}, vi_t{var.index() - CCTK_FirstVarIndexI(gi)}};\n",
            "int firstvarindex;\n",
            "const int n = firstvarindex_count + 1;\n",
            "const int n = myfirstvarindex + 1;\n",
            "varname = CCTK_FullVarName(driver().record(gi).var(vi_t{vi}).index());\n",
        ]:
            with self.subTest(text=text):
                self.assertEqual(reported("CarpetX/src/hierarchy/valid.cxx", text), [])

    def test_other_thorns_are_not_checked(self):
        self.assertEqual(reported("PDESolvers/src/solve.cxx", "  " + self.ADDITIONS[2] + "\n"), [])

    def test_only_c_and_cxx_files_are_checked(self):
        self.assertEqual(reported("CarpetX/src/make.code.defn", "X = firstvarindex + 1\n", Lang.MAKE), [])

    def test_line_numbers(self):
        text = "int a;\n" + self.ADDITIONS[0] + "\n"
        self.assertEqual(findings("CarpetX/src/schedule/evolve.cxx", text, Lang.CXX), [Finding(2, "varindex", self.ADDITIONS[0])])


class Window(unittest.TestCase):
    USES = [
        "struct active_levels_t;",
        "int SyncGroupsByDirI(active_levels_t window, const cGH *restrict cctkGH,",
        "active_levels_t scheduled_window(const char *caller,",
        "const traversal_scope_t scope(active_levels_t(0, level + 1));",
    ]

    def test_window_reported(self):
        for line in self.USES:
            for name in ["CarpetX/src/schedule/schedule.hxx", "CarpetX/src/comm/sync_restrict.cxx", "CarpetX/src/include/carpetx.hxx"]:
                with self.subTest(name=name, line=line):
                    self.assertEqual(reported(name, "  " + line + "\n"), [("window", line)])

    def test_the_window_type_and_the_scheduler_state_are_fine(self):
        for name, text in [
            ("CarpetX/src/traversal/traversal.cxx", "std::optional<window_t> active_levels;\n"),
            ("CarpetX/src/traversal/traversal.cxx", "active_levels = window_t::levels(0, 1);\n"),
            ("CarpetX/src/schedule/evolve.cxx", "struct active_levels_type;\n"),
            ("CarpetX/src/schedule/evolve.cxx", "my_active_levels_t x;\n"),
        ]:
            with self.subTest(name=name, text=text):
                self.assertEqual(reported(name, text), [])

    def test_other_thorns_are_not_checked(self):
        self.assertEqual(reported("ODESolvers/src/solve.cxx", self.USES[0] + "\n"), [])

    def test_only_c_and_cxx_files_are_checked(self):
        self.assertEqual(reported("CarpetX/src/make.code.defn", "X = active_levels_t\n", Lang.MAKE), [])

    def test_line_numbers(self):
        text = "int a;\n" + self.USES[0] + "\n"
        self.assertEqual(findings("CarpetX/src/hierarchy/valid.hxx", text, Lang.CXX), [Finding(2, "window", self.USES[0])])


class Traversal(unittest.TestCase):
    USES = [
        "active_levels = std::nullopt;",
        "assert(!active_levels);",
        "loop_serially(*active_levels, f);",
        "if (active_levels) return;",
        "driver().tail_bin = bin;",
        "if (!tail_bin) return;",
        "recovery_traversal = true;",
        "assert(!recovery_traversal);",
        "last_routine = attribute;",
        "const auto *f = last_routine;",
    ]

    def test_traversal_reported(self):
        for line in self.USES:
            for name in ["CarpetX/src/schedule/evolve.cxx", "CarpetX/src/api/api.cxx", "CarpetX/src/hierarchy/driver.hxx", "CarpetX/src/hierarchy/traversal.cxx"]:
                with self.subTest(name=name, line=line):
                    self.assertEqual(reported(name, "  " + line + "\n"), [("traversal", line)])

    def test_the_traversal_unit_owns_the_slot(self):
        for line in self.USES:
            for name in ["CarpetX/src/traversal/traversal.cxx", "CarpetX/src/traversal/traversal.hxx"]:
                with self.subTest(name=name, line=line):
                    door = [("door", line)] if "driver()" in line else []
                    self.assertEqual(reported(name, line + "\n"), door)

    def test_another_thorn_and_the_public_headers_may_use_the_name(self):
        for line in self.USES:
            for name in ["ODESolvers/src/solve.cxx", "CarpetX/src/include/carpetx.hxx"]:
                with self.subTest(name=name, line=line):
                    self.assertEqual(reported(name, line + "\n"), [])

    def test_a_longer_name_is_not_the_slot(self):
        self.assertEqual(reported("CarpetX/src/schedule/evolve.cxx", "active_levels_t x;\n"), [("window", "active_levels_t x;")])
        for text in [
            "my_active_levels = 1;\n",
            "int active_levels_count;\n",
            "SyncGroupsOnActiveLevels(cctkGH, n, gs, ds);\n",
            "scheduled_tail_bin();\n",
            "tail_bin_scope_t scope(bin);\n",
            "in_recovery_traversal();\n",
            "recovery_scope_t scope;\n",
            "scheduled_routine();\n",
            "routine_scope_t routine_scope(attribute);\n",
        ]:
            with self.subTest(text=text):
                self.assertEqual(reported("CarpetX/src/schedule/evolve.cxx", text), [])

    def test_only_c_and_cxx_files_are_checked(self):
        self.assertEqual(reported("CarpetX/src/make.code.defn", "X = active_levels\n", Lang.MAKE), [])

    def test_line_numbers(self):
        text = "int a;\n\n" + self.USES[0] + "\n"
        self.assertEqual(findings("CarpetX/src/schedule/evolve.cxx", text, Lang.CXX), [Finding(3, "traversal", self.USES[0])])

    def test_the_recovered_iteration_is_a_slot(self):
        for line in [
            "int recover_iteration = -1;",
            "recover_iteration = codec->read_parameters(recover_dir, recover_file);",
            "return recover_iteration >= 0;",
            "if (cctkGH->cctk_iteration == recover_iteration) {",
        ]:
            with self.subTest(line=line):
                self.assertEqual(reported("CarpetX/src/io/io.cxx", "  " + line + "\n"), [("traversal", line)])
                self.assertEqual(reported("CarpetX/src/traversal/traversal.cxx", line + "\n"), [])

    def test_the_recovered_iteration_readers_are_not_the_slot(self):
        for text in [
            "record_recovered_iteration(codec->read_parameters(recover_dir, recover_file));\n",
            "return recovered_iteration() >= 0;\n",
            "assert(cctkGH->cctk_iteration == recovered_iteration());\n",
        ]:
            with self.subTest(text=text):
                self.assertEqual(reported("CarpetX/src/io/io.cxx", text), [])


class Index(unittest.TestCase):
    BARE = [
        "void EnsureRKBuffers(int patch, int level, gi_t gi);",
        "const GroupRecord &record(const int gi, const std::source_location site) const;",
        "void Restrict(const cGH *cctkGH, int min_level, int max_level,",
        "bool window_is_restricted(int min_level, int max_level);",
        "valid_t valid(group_t group, tl_t tl, int vi);",
        "void prolongate_ghosts(window_t window, std::span<const gi_t> groups, int tl = -1);",
        "int min_patch, max_patch;",
    ]

    def test_index_reported(self):
        for line in self.BARE:
            for name in ["CarpetX/src/comm/subcycling.hxx", "CarpetX/src/hierarchy/driver.hxx", "CarpetX/src/include/carpetx.hxx"]:
                with self.subTest(name=name, line=line):
                    self.assertEqual(reported(name, "  " + line + "\n"), [("index", line)])

    def test_index_allowed_in_the_handle_constructors_and_window_bounds(self):
        text = (
            "struct level_t {\n"
            "  constexpr level_t(const patch_t patch, const int level)\n"
            "      : patch_(patch), level_(level) {}\n"
            "};\n"
            "struct window_t {\n"
            "  static window_t\n"
            "  levels(int min_level, int max_level,\n"
            "         std::source_location site = std::source_location::current());\n"
            "  window_t\n"
            "  narrow(int min_level, int max_level,\n"
            "         std::source_location site = std::source_location::current()) const;\n"
            "private:\n"
            "  constexpr window_t(const int min_level, const int max_level,\n"
            "                     const int min_patch, const int max_patch)\n"
            "      : min_level_(min_level) {}\n"
            "  int min_level_, max_level_;\n"
            "};\n"
        )
        self.assertEqual(reported("CarpetX/src/include/carpetx.hxx", text), [])

    def test_index_allowed_in_flesh_and_amrex_signatures(self):
        self.assertEqual(reported("CarpetX/src/schedule/schedule.hxx", "int QueryGroupStorageB(const cGH *cctkGH, int gi, const char *groupname);\n"), [])
        text = (
            "  virtual void ErrorEst(int level, amrex::TagBoxArray &tags, amrex::Real time,\n"
            "                        int ngrow) override;\n"
            "  virtual void ClearLevel(int level) override;\n"
        )
        self.assertEqual(reported("CarpetX/src/hierarchy/driver.hxx", text), [])

    def test_index_allowed_only_where_the_signature_is_fixed(self):
        for name in ["CarpetX/src/hierarchy/driver.hxx", "CarpetX/src/comm/comm.hxx"]:
            with self.subTest(name=name):
                self.assertEqual(
                    reported(name, "int QueryGroupStorageB(const cGH *cctkGH, int gi, const char *groupname);\n"),
                    [("index", "int QueryGroupStorageB(const cGH *cctkGH, int gi, const char *groupname);")],
                )
        self.assertEqual(reported("CarpetX/src/comm/comm.hxx", "int SyncGroupsOnActiveLevels(const cGH *cctkGH, int gi);\n"), [])
        self.assertEqual(
            reported("CarpetX/src/schedule/schedule.hxx", "int SyncGroupsOnActiveLevels(const cGH *cctkGH, int gi);\n"),
            [("index", "int SyncGroupsOnActiveLevels(const cGH *cctkGH, int gi);")],
        )
        self.assertEqual(
            reported("CarpetX/src/schedule/schedule.hxx", "window_t narrow(int min_level, int max_level);\n"),
            [("index", "window_t narrow(int min_level, int max_level);")],
        )
        self.assertEqual(
            reported("CarpetX/src/include/carpetx.hxx", "int num_levels(int patch);\n"),
            [("index", "int num_levels(int patch);")],
        )

    def test_index_only_in_carpetx_headers_and_the_bound_sources(self):
        for name in ["CarpetX/src/schedule/evolve.cxx", "CarpetX/src/ops/reduction.cxx", "CarpetX/src/hierarchy/hierarchy.cxx", "ODESolvers/src/solve.hxx", "TestCarpetXAPI/src/api_test.cxx"]:
            with self.subTest(name=name):
                self.assertEqual(reported(name, self.BARE[0] + "\n"), [])

    COUNTED = [
        "for (int gi = 0; gi < numgroups; ++gi) {",
        "const int level = leveldata.level.index();",
        "for (int vi = 0; vi < record.numvars; ++vi)",
        "std::string make_varname(int gi, int vi, level_t level) {",
        "num_components(int patch, int level) const {",
    ]

    def test_an_io_source_is_judged_like_a_header(self):
        self.assertEqual(rules.INDEX_SOURCE_LAYERS, ("io",))
        for line in self.COUNTED:
            for name in ["CarpetX/src/io/io_tsv.cxx", "CarpetX/src/io/io.cxx", "CarpetX/src/io/io_silo_checkpoint.cxx"]:
                with self.subTest(name=name, line=line):
                    self.assertEqual(reported(name, "  " + line + "\n"), [("index", line)])
            with self.subTest(name="ops", line=line):
                self.assertEqual(reported("CarpetX/src/ops/reduction.cxx", "  " + line + "\n"), [])

    def test_the_handles_pass_in_an_io_source(self):
        text = (
            "for (const GroupRecord &record : selection.records(group_kind::grid_function)) {\n"
            "  for (const vi_t vi : record.variables())\n"
            "    names.at(vi.index()) = CCTK_FullVarName(record.var(vi).index());\n"
            "}\n"
            "const level_t level = leveldata.level;\n"
            "for (int lev = 0; lev < max_levels; ++lev) {\n"
            "}\n"
        )
        self.assertEqual(reported("CarpetX/src/io/io.cxx", text), [])

    FIXED_DEFINITIONS = [
        ("CarpetX/src/schedule/storage.cxx", "int QueryGroupStorageB(const cGH *cctkGH, int gi, const char *groupname) {\n"),
        ("CarpetX/src/schedule/storage.cxx", "int GroupStorageIncrease(const cGH *cctkGH, int gi, const int *tls) {\n"),
        ("CarpetX/src/schedule/storage.cxx", "int GroupStorageDecrease(const cGH *cctkGH, int gi, const int *tls) {\n"),
        ("CarpetX/src/flesh/startup.cxx", "const int *ArrayGroupSizeB(const cGH *cctkGH, int dir, int gi,\n                           const char *groupname) {\n"),
        ("CarpetX/src/flesh/startup.cxx", "int GroupDynamicData(const cGH *cctkGH, int gi, cGroupDynamicData *data);\n"),
        ("CarpetX/src/comm/sync_restrict.cxx", "int SyncGroupsOnActiveLevels(const cGH *restrict const cctkGH,\n                             const int gi) {\n"),
        ("CarpetX/src/regrid/regrid.cxx", "void CactusAmrCore::ErrorEst(const int level, amrex::TagBoxArray &tags,\n                             const amrex::Real time, const int ngrow) {\n"),
        ("CarpetX/src/regrid/regrid.cxx", "void CactusAmrCore::MakeNewLevelFromScratch(\n    const int level, const amrex::Real time, const amrex::BoxArray &ba,\n    const amrex::DistributionMapping &dm) {\n"),
        ("CarpetX/src/regrid/regrid.cxx", "void CactusAmrCore::MakeNewLevelFromCoarse(\n    const int level, const amrex::Real time, const amrex::BoxArray &ba,\n    const amrex::DistributionMapping &dm) {\n"),
        ("CarpetX/src/regrid/regrid.cxx", "void CactusAmrCore::RemakeLevel(const int level, const amrex::Real time,\n                                const amrex::BoxArray &ba,\n                                const amrex::DistributionMapping &dm) {\n"),
        ("CarpetX/src/regrid/regrid.cxx", "void CactusAmrCore::ClearLevel(const int level) {\n"),
    ]

    @staticmethod
    def bare_lines(text):
        return [("index", line.strip()) for line in text.splitlines() if rules.INDEX_PARAM.search(line)]

    def test_every_fixed_source_name_has_a_definition_case(self):
        named = {(name, call) for name, calls in rules.INDEX_FIXED.items() if name.endswith(".cxx") for call in calls}
        cased = {(name, call) for name, text in self.FIXED_DEFINITIONS for call in rules.INDEX_FIXED[name] if re.search(rf"\b{call}\s*\(", text)}
        self.assertEqual(cased, named)
        for name, text in self.FIXED_DEFINITIONS:
            with self.subTest(name=name, text=text):
                self.assertTrue(self.bare_lines(text), "a case without a bare index int tests no exemption")

    def test_a_flesh_or_amrex_fixed_definition_passes_once_its_layer_is_bound(self):
        with mock.patch.object(rules, "INDEX_SOURCE_LAYERS", LAYERS):
            for name, text in self.FIXED_DEFINITIONS:
                with self.subTest(name=name, text=text):
                    self.assertEqual(reported(name, text), [])
                    self.assertEqual(reported(name, "  const int level = leveldata.level.index();\n"), [("index", "const int level = leveldata.level.index();")])
        for name, text in self.FIXED_DEFINITIONS:
            with self.subTest(name=name, text=text, bound=False):
                self.assertEqual(reported(name, text), [])

    def test_the_exemption_ends_with_the_fixed_signature(self):
        text = "int QueryGroupStorageB(const cGH *cctkGH, int gi, const char *groupname) {\n  for (int level = 0; level < 2; ++level)\n"
        with mock.patch.object(rules, "INDEX_SOURCE_LAYERS", LAYERS):
            self.assertEqual(reported("CarpetX/src/schedule/storage.cxx", text), [("index", "for (int level = 0; level < 2; ++level)")])

    def test_a_fixed_definition_passes_only_in_its_own_file(self):
        with mock.patch.object(rules, "INDEX_SOURCE_LAYERS", LAYERS):
            for name, text in self.FIXED_DEFINITIONS:
                for other in ["CarpetX/src/io/io.cxx", "CarpetX/src/api/api.cxx", "CarpetX/src/hierarchy/hierarchy.cxx"]:
                    with self.subTest(name=name, other=other):
                        self.assertEqual(reported(other, text), self.bare_lines(text))

    def test_a_source_layer_joins_by_its_tuple_entry(self):
        line = "for (int patch = 0; patch < num_patches; ++patch)"
        self.assertEqual(reported("CarpetX/src/schedule/evolve.cxx", line + "\n"), [])
        with mock.patch.object(rules, "INDEX_SOURCE_LAYERS", ("io", "schedule")):
            self.assertEqual(reported("CarpetX/src/schedule/evolve.cxx", line + "\n"), [("index", line)])
            self.assertEqual(reported("CarpetX/src/api/api.cxx", line + "\n"), [])

    def test_other_ints_are_not_indices(self):
        for text in [
            "int levels;\n",
            "int level_count, gi_rhs, tls;\n",
            "void f(int patches, int min_levels, int vi0);\n",
            "int index() const;\n",
            "int min_level_, max_level_;\n",
            "void f(level_t level, gi_t gi, tl_t tl, vi_t vi, patch_t patch);\n",
        ]:
            with self.subTest(text=text):
                self.assertEqual(reported("CarpetX/src/hierarchy/driver.hxx", text), [])

    def test_line_numbers(self):
        text = "int a;\n\n" + self.BARE[0] + "\n"
        self.assertEqual(findings("CarpetX/src/comm/subcycling.hxx", text, Lang.CXX), [Finding(3, "index", self.BARE[0])])


class Records(unittest.TestCase):
    READS = [
        "const int numgroups = CCTK_NumGroups();",
        "if (CCTK_GroupTypeI(gi) != CCTK_GF) continue;",
        "const int ierr = CCTK_GroupData(gi.index(), &cgroup);",
        "const int dim = CCTK_GroupDimI (gi);",
        "const int numvars = CCTK_NumVarsInGroupI(gi);",
        "const int first = CCTK_FirstVarIndexI(gi);",
        "enabled.at(CCTK_GroupIndexFromVarI(index)) = true;",
    ]

    def test_each_group_table_call_is_reported_in_io(self):
        self.assertEqual(rules.RECORDS_LAYERS, ("io",))
        self.assertEqual(len(self.READS), len(rules.RECORDS_CALLS))
        for line in self.READS:
            for name in ["CarpetX/src/io/io_tsv.cxx", "CarpetX/src/io/io_meta.cxx", "CarpetX/src/io/selection.hxx"]:
                with self.subTest(name=name, line=line):
                    self.assertEqual(reported(name, "  " + line + "\n"), [("records", line)])

    def test_a_counted_group_loop_in_io_is_reported_under_index_and_records(self):
        line = "for (int gi = 0; gi < CCTK_NumGroups(); ++gi) {}"
        self.assertEqual(reported("CarpetX/src/io/io_tsv.cxx", line + "\n"), [("index", line), ("records", line)])

    def test_the_layers_that_build_and_receive_the_table_read_it(self):
        for line in self.READS:
            for name in ["CarpetX/src/hierarchy/hierarchy.cxx", "CarpetX/src/hierarchy/driver.hxx", "CarpetX/src/flesh/startup.cxx"]:
                with self.subTest(name=name, line=line):
                    self.assertEqual(reported(name, line + "\n"), [])
        self.assertNotIn("hierarchy", rules.RECORDS_LAYERS)
        self.assertNotIn("flesh", rules.RECORDS_LAYERS)

    def test_a_layer_not_yet_bound_is_not_judged(self):
        for line in self.READS:
            for name in ["CarpetX/src/api/api.cxx", "CarpetX/src/schedule/storage.cxx", "CarpetX/src/comm/sync_restrict.cxx", "CarpetX/src/ops/reduction.cxx"]:
                with self.subTest(name=name, line=line):
                    self.assertEqual(reported(name, line + "\n"), [])

    def test_a_layer_joins_by_its_tuple_entry(self):
        with mock.patch.object(rules, "RECORDS_LAYERS", ("io", "api")):
            self.assertEqual(reported("CarpetX/src/api/api.cxx", self.READS[0] + "\n"), [("records", self.READS[0])])
            self.assertEqual(reported("CarpetX/src/schedule/storage.cxx", self.READS[0] + "\n"), [])

    def test_name_lookups_and_names_are_not_table_reads(self):
        for text in [
            "const gi_t gi{CCTK_GroupIndex(groupname)};\n",
            "const int var = CCTK_VarIndex(name.c_str());\n",
            "const gi_t gi{CCTK_GroupIndexFromVar(name.c_str())};\n",
            "const char *name = CCTK_FullGroupName(record.groupindex.index());\n",
            "const char *name = CCTK_FullVarName(record.var(vi).index());\n",
            "const int n = CCTK_NumVars();\n",
            "const GroupRecord &record = driver().record_of(var_t{var});\n",
            "for (const GroupRecord &record : ext.records(group_kind::grid_function))\n",
            "int CCTK_NumGroupsTotal(void);\n",
            "my_CCTK_GroupData(gi, &cgroup);\n",
            "cGroupDynamicData data;\n",
        ]:
            with self.subTest(text=text):
                self.assertEqual(reported("CarpetX/src/io/io_meta.cxx", text), [])

    def test_other_thorns_public_headers_and_make_files_are_not_checked(self):
        self.assertEqual(reported("ODESolvers/src/solve.cxx", self.READS[0] + "\n"), [])
        self.assertEqual(reported("CarpetX/src/include/carpetx.hxx", self.READS[0] + "\n"), [])
        self.assertEqual(reported("CarpetX/src/io/make.code.defn", "X = CCTK_NumGroups()\n", Lang.MAKE), [])

    def test_line_numbers(self):
        text = "int a;\n\n" + self.READS[2] + "\n"
        self.assertEqual(findings("CarpetX/src/io/io_tsv.cxx", text, Lang.CXX), [Finding(3, "records", self.READS[2])])


class Codec(unittest.TestCase):
    CODECS = [
        "CarpetX/src/io/io_openpmd_checkpoint.cxx",
        "CarpetX/src/io/io_openpmd_names.hxx",
        "CarpetX/src/io/io_silo_checkpoint.cxx",
        "CarpetX/src/io/io_silo_names.hxx",
    ]
    HIERARCHY = [
        "for (const auto &patchdata : ghext->patches()) {",
        "const auto &leveldata = driver().level(level);",
        "const auto &leveldata = CarpetX::driver ().level(level);",
        "const GHExt::GroupRecord &group = record;",
        "for (const LevelData &leveldata : patchdata.levels()) {",
        "const GroupData &groupdata = leveldata.group(gi);",
        '#include "hierarchy/driver.hxx"',
        "#include <hierarchy/driver.hxx>",
    ]

    def test_hierarchy_access_is_reported_in_a_codec_file(self):
        for line in self.HIERARCHY:
            for name in self.CODECS:
                with self.subTest(name=name, line=line):
                    door = [("door", line)] if "ghext" in line else []
                    self.assertEqual(reported(name, "  " + line + "\n"), [("codec", line)] + door)

    def test_the_record_and_names_header_are_allowed(self):
        text = (
            '#include "io/checkpoint.hxx"\n'
            '#include "io/io_openpmd_names.hxx"\n'
            "#include <openPMD/openPMD.hpp>\n"
            "const amrex::Geometry &geom = record.levels[p][l].geom;\n"
            "amrex::MultiFab &mfab = *grid.mfab;\n"
            "const std::string meshname = make_meshname(id.group, id.level);\n"
        )
        for name in self.CODECS:
            with self.subTest(name=name):
                self.assertEqual(reported(name, text), [])

    def test_a_longer_name_is_not_the_hierarchy(self):
        text = (
            "const auto &x = my_ghext->patches();\n"
            "my_driver();\n"
            "struct GHExtension;\n"
            "LevelDataView view;\n"
            "const GroupDataset dataset;\n"
            '#include "hierarchy/driver_api.hxx"\n'
        )
        self.assertEqual(reported(self.CODECS[0], text), [])

    def test_the_slice_backend_is_out_of_scope(self):
        line = "for (const auto &patchdata : driver().patches()) {\n"
        for name in ["CarpetX/src/io/io_openpmd.cxx", "CarpetX/src/io/io_silo.cxx", "CarpetX/src/io/checkpoint.cxx", "ODESolvers/src/io_openpmd_checkpoint.cxx"]:
            with self.subTest(name=name):
                self.assertEqual(reported(name, line), [])

    def test_only_c_and_cxx_files_are_checked(self):
        self.assertEqual(reported(self.CODECS[0], "X = ghext->patches()\n", Lang.MAKE), [])

    def test_line_numbers(self):
        text = '#include "io/checkpoint.hxx"\n\n' + self.HIERARCHY[0] + "\n"
        self.assertEqual(
            findings(self.CODECS[0], text, Lang.CXX),
            [Finding(3, "codec", self.HIERARCHY[0]), Finding(3, "door", self.HIERARCHY[0])],
        )


class Door(unittest.TestCase):
    USES = [
        "for (auto &p : ghext->patches())",
        "assert(!ghext);",
        "// ghext is gone once CarpetX_Shutdown has run",
        'yaml << YAML::LocalTag("ghext-2.0.0");',
    ]

    def test_door_reported(self):
        for line in self.USES:
            for name in ["CarpetX/src/schedule/evolve.cxx", "CarpetX/src/io/io.cxx", "CarpetX/src/hierarchy/driver.hxx", "CarpetX/src/schedule/hierarchy.cxx"]:
                with self.subTest(name=name, line=line):
                    self.assertEqual(reported(name, "  " + line + "\n"), [("door", line)])

    def test_hierarchy_cxx_owns_the_pointer(self):
        for line in self.USES:
            with self.subTest(line=line):
                self.assertEqual(reported("CarpetX/src/hierarchy/hierarchy.cxx", line + "\n"), [])

    def test_another_thorn_and_the_public_headers_may_use_the_name(self):
        for line in self.USES:
            for name in ["ODESolvers/src/solve.cxx", "CarpetX/src/include/carpetx.hxx"]:
                with self.subTest(name=name, line=line):
                    self.assertEqual(reported(name, line + "\n"), [])

    def test_a_longer_name_is_not_the_pointer(self):
        for text in [
            "const auto &x = my_ghext->patches();\n",
            "int ghext_handle = -1;\n",
            "struct GHExtension;\n",
            "const GHExt &ext = driver();\n",
        ]:
            for name in ["CarpetX/src/schedule/evolve.cxx", "CarpetX/src/io/io.cxx", "CarpetX/src/hierarchy/driver.hxx"]:
                with self.subTest(name=name, text=text):
                    self.assertEqual(reported(name, text), [])

    def test_a_codec_naming_it_is_reported_under_both_rules(self):
        line = "for (const auto &patchdata : ghext->patches()) {"
        self.assertEqual(reported("CarpetX/src/io/io_silo_checkpoint.cxx", line + "\n"), [("codec", line), ("door", line)])

    def test_only_c_and_cxx_files_are_checked(self):
        self.assertEqual(reported("CarpetX/src/make.code.defn", "X = ghext\n", Lang.MAKE), [])
        self.assertEqual(reported("CarpetX/src/traversal/make.code.defn", "X = driver()\n", Lang.MAKE), [])

    def test_line_numbers(self):
        text = "int a;\n\n" + self.USES[1] + "\n"
        self.assertEqual(findings("CarpetX/src/schedule/evolve.cxx", text, Lang.CXX), [Finding(3, "door", self.USES[1])])

    CALLS = [
        "GHExt &ext = driver();",
        "const auto &leveldata = driver().level(level);",
        "for (const auto &patchdata : CarpetX::driver ().patches()) {",
        "refuse(site, driver( ).num_patches() == 0, msg);",
        "// driver() refuses once the hierarchy is gone",
    ]

    def test_driver_is_reported_below_the_hierarchy(self):
        for line in self.CALLS:
            for name in [
                "CarpetX/src/traversal/traversal.cxx",
                "CarpetX/src/traversal/traversal.hxx",
                "CarpetX/src/util/refuse.cxx",
                "CarpetX/src/util/task_manager.hxx",
            ]:
                with self.subTest(name=name, line=line):
                    self.assertEqual(reported(name, "  " + line + "\n"), [("door", line)])

    def test_the_hierarchy_and_every_layer_above_it_call_driver(self):
        for line in self.CALLS:
            for name in [layer_path(layer, "x.cxx") for layer in HIERARCHY_LAYERS] + [
                "CarpetX/src/hierarchy/valid.cxx",
                "CarpetX/src/io/io.cxx",
                "CarpetX/src/flesh/aliased.cxx",
            ]:
                with self.subTest(name=name, line=line):
                    self.assertEqual(reported(name, line + "\n"), [])

    def test_a_private_file_outside_the_layers_does_not_call_driver(self):
        for name in ["CarpetX/src/internal.hxx", "CarpetX/src/extra/x.cxx", "CarpetX/src/hierarchy/sub/x.cxx"]:
            with self.subTest(name=name):
                self.assertEqual(reported(name, self.CALLS[0] + "\n"), [("door", self.CALLS[0])])

    def test_another_thorn_and_the_public_headers_may_call_driver(self):
        for name in ["ODESolvers/src/solve.cxx", "CarpetX/src/include/carpetx.hxx"]:
            with self.subTest(name=name):
                self.assertEqual(reported(name, self.CALLS[0] + "\n"), [])

    def test_a_longer_name_or_an_argument_is_not_the_door(self):
        for text in [
            "my_driver();\n",
            "driver_handle();\n",
            "drivers().clear();\n",
            "auto *door = &driver;\n",
            "int driver = 0;\n",
            "driver(level);\n",
        ]:
            with self.subTest(text=text):
                self.assertEqual(reported("CarpetX/src/traversal/traversal.cxx", text), [])

    def test_a_line_naming_both_is_one_finding(self):
        line = "assert(ghext.get() == &driver());"
        self.assertEqual(reported("CarpetX/src/traversal/traversal.cxx", line + "\n"), [("door", line)])
        self.assertEqual(reported("CarpetX/src/schedule/evolve.cxx", line + "\n"), [("door", line)])
        self.assertEqual(reported("CarpetX/src/hierarchy/hierarchy.cxx", line + "\n"), [])

    def test_line_numbers_of_a_driver_call(self):
        text = "int a;\n" + self.CALLS[0] + "\n"
        self.assertEqual(findings("CarpetX/src/util/refuse.cxx", text, Lang.CXX), [Finding(2, "door", self.CALLS[0])])


class Layer(unittest.TestCase):
    def test_a_file_includes_its_own_layer_and_the_layers_below_it(self):
        for line in [
            '#include "comm/comm.hxx"',
            '#include "comm/boundaries_impl.hxx"',
            '#include "hierarchy/driver.hxx"',
            '#include "traversal/traversal.hxx"',
            '#include "util/refuse.hxx"',
            '#include "util/typeprops.hh"',
            '  #  include   "hierarchy/loops.hxx"  // the loops',
        ]:
            with self.subTest(line=line):
                self.assertEqual(reported("CarpetX/src/comm/sync_restrict.cxx", line + "\n"), [])

    def test_an_include_from_a_higher_layer_is_reported(self):
        for line in [
            '#include "schedule/schedule.hxx"',
            '#include "io/io.hxx"',
            '#include "ops/interp.hxx"',
            '#include "regrid/regrid.hxx"',
            '#include "flesh/startup.hxx"',
        ]:
            with self.subTest(line=line):
                self.assertEqual(reported("CarpetX/src/comm/sync_restrict.cxx", line + "\n"), [("layer", line)])
        self.assertEqual(reported("CarpetX/src/api/api.hxx", '#include "flesh/startup.hxx"\n'), [("layer", '#include "flesh/startup.hxx"')])

    def test_each_layer_sees_exactly_the_layers_up_to_its_own(self):
        for layer in LAYERS:
            for other in LAYERS:
                line = f'#include "{other}/x.hxx"'
                expected = [] if LAYERS.index(other) <= LAYERS.index(layer) else [("layer", line)]
                for name in [layer_path(layer, "x.cxx"), layer_path(layer, "x.hxx")]:
                    with self.subTest(name=name, line=line):
                        self.assertEqual(reported(name, line + "\n"), expected)

    def test_an_unqualified_private_name_is_reported(self):
        for line in ['#include "driver.hxx"', '#include "schedule.hxx"', '#include "valid.hxx"', '#include "loop.hxx"']:
            for name in ["CarpetX/src/hierarchy/hierarchy.cxx", "CarpetX/src/hierarchy/valid.cxx", "CarpetX/src/comm/fillpatch.hxx", "CarpetX/src/flesh/startup.cxx"]:
                with self.subTest(name=name, line=line):
                    self.assertEqual(reported(name, line + "\n"), [("layer", line)])

    def test_a_relative_or_unknown_path_is_reported(self):
        for line in [
            '#include "../x.hxx"',
            '#include "../hierarchy/driver.hxx"',
            '#include "comm/../schedule/schedule.hxx"',
            '#include "./comm/comm.hxx"',
            '#include "extra/x.hxx"',
            '#include "include/carpetx.hxx"',
            '#include "CarpetX/src/hierarchy/driver.hxx"',
        ]:
            with self.subTest(line=line):
                self.assertEqual(reported("CarpetX/src/flesh/startup.cxx", line + "\n"), [("layer", line)])

    def test_the_root_header_passes_everywhere(self):
        for name in [layer_path(layer, "x.hxx") for layer in LAYERS] + ["CarpetX/src/extra/x.hxx"]:
            with self.subTest(name=name):
                self.assertEqual(reported(name, '#include "internal.hxx"\n'), [])

    def test_a_private_file_outside_the_layers_includes_only_the_root_header(self):
        for name in ["CarpetX/src/internal.hxx", "CarpetX/src/extra/x.cxx", "CarpetX/src/comm/sub/x.cxx"]:
            with self.subTest(name=name):
                self.assertEqual(reported(name, '#include "util/refuse.hxx"\n'), [("layer", '#include "util/refuse.hxx"')])

    def test_angle_includes_are_judged_only_when_they_name_a_layer(self):
        text = "#include <carpetx.hxx>\n#include <loop_device.hxx>\n#include <AMReX.H>\n#include <openPMD/openPMD.hpp>\n#include <yaml-cpp/yaml.h>\n"
        self.assertEqual(reported("CarpetX/src/util/logo.cxx", text), [])
        self.assertEqual(reported("CarpetX/src/comm/fillpatch.cxx", "#include <hierarchy/driver.hxx>\n"), [])
        self.assertEqual(reported("CarpetX/src/comm/fillpatch.cxx", "#include <io/io.hxx>\n"), [("layer", "#include <io/io.hxx>")])

    def test_mentions_outside_an_include_are_not_includes(self):
        text = 'const char *p = "schedule/schedule.hxx";\n// #include "schedule/schedule.hxx"\n#define X "io/io.hxx"\n'
        self.assertEqual(reported("CarpetX/src/comm/sync_restrict.cxx", text), [])

    def test_other_thorns_public_headers_and_make_files_are_not_checked(self):
        line = '#include "schedule/schedule.hxx"\n'
        self.assertEqual(reported("ODESolvers/src/solve.cxx", line), [])
        self.assertEqual(reported("CarpetX/src/include/carpetx.hxx", line), [])
        self.assertEqual(reported("CarpetX/src/comm/make.code.defn", line, Lang.MAKE), [])

    def test_line_numbers(self):
        text = '#include "comm/comm.hxx"\n#include "hierarchy/driver.hxx"\n#include "schedule/schedule.hxx"\n'
        self.assertEqual(findings("CarpetX/src/comm/sync_restrict.cxx", text, Lang.CXX), [Finding(3, "layer", '#include "schedule/schedule.hxx"')])


class Layers(unittest.TestCase):
    def test_a_file_directly_in_a_layer_directory_is_in_that_layer(self):
        for layer in LAYERS:
            with self.subTest(layer=layer):
                self.assertEqual(layer_of(f"CarpetX/src/{layer}/x.cxx"), layer)
                self.assertEqual(layer_of(layer_path(layer, "make.code.defn")), layer)

    def test_other_files_are_in_no_layer(self):
        for name in [
            "CarpetX/src/internal.hxx",
            "CarpetX/src/make.code.defn",
            "CarpetX/src/include/carpetx.hxx",
            "CarpetX/src/extra/x.cxx",
            "CarpetX/src/comm/sub/x.cxx",
            "CarpetX/comm/x.cxx",
            "ODESolvers/src/comm/x.cxx",
        ]:
            with self.subTest(name=name):
                self.assertIsNone(layer_of(name))

    def test_a_path_is_built_only_in_a_layer(self):
        self.assertEqual(layer_path("hierarchy", "driver.hxx"), "CarpetX/src/hierarchy/driver.hxx")
        with self.assertRaises(AssertionError):
            layer_path("loops", "loops.hxx")

    def test_a_layer_sees_itself_and_every_layer_before_it(self):
        self.assertEqual(below("util"), ("util",))
        self.assertEqual(below("comm"), ("util", "traversal", "hierarchy", "comm"))
        self.assertEqual(below("flesh"), LAYERS)
        for layer in LAYERS:
            with self.subTest(layer=layer):
                self.assertEqual(below(layer)[-1], layer)
        with self.assertRaises(AssertionError):
            below("loops")

    def test_the_hierarchy_and_the_layers_above_it_may_call_driver(self):
        self.assertEqual(HIERARCHY_LAYERS, below("flesh")[len(below("traversal")) :])
        self.assertEqual(HIERARCHY_LAYERS[0], "hierarchy")
        self.assertEqual(set(LAYERS) - set(HIERARCHY_LAYERS), {"util", "traversal"})

    def test_every_file_a_rule_names_exists(self):
        named = (
            rules.VALIDITY_FILES
            + rules.TRAVERSAL_FILES
            + (rules.DOOR_FILE, rules.RECORD_VAR_FILE, MAKE_DEFN)
            + rules.SYNC_FILES
            + tuple(rules.INDEX_FIXED)
            + rules.CODEC_FILES
        )
        for name in named:
            with self.subTest(name=name):
                self.assertTrue(os.path.isfile(os.path.join(PROJECT, name)))

    def test_every_fixed_signature_is_in_its_file(self):
        for name, calls in rules.INDEX_FIXED.items():
            with open(os.path.join(PROJECT, name)) as f:
                text = f.read()
            for call in calls:
                with self.subTest(name=name, call=call):
                    self.assertRegex(text, rf"\b{call}\s*\(")

    def test_every_private_source_is_in_a_layer_the_build_lists(self):
        with open(os.path.join(PROJECT, MAKE_DEFN)) as f:
            subdirs = re.search(r"^SUBDIRS\s*=(.*)$", f.read(), re.M).group(1).split()
        self.assertEqual([layer for layer in LAYERS if layer in subdirs], subdirs)
        src = os.path.join(PROJECT, "CarpetX", "src")
        for dirpath, _, filenames in os.walk(src):
            for filename in filenames:
                name = os.path.relpath(os.path.join(dirpath, filename), PROJECT).replace(os.sep, "/")
                if name in (MAKE_DEFN, "CarpetX/src/internal.hxx") or name.startswith("CarpetX/src/include/"):
                    continue
                with self.subTest(name=name):
                    self.assertIn(layer_of(name), subdirs)


MAP = os.path.join(PROJECT, "agent_docs", "carpetx-internals.md")
MAP_HEADER = "| layer | owns | may include | driver() | rule |"
MAP_SEPARATOR = re.compile(r"^\|(\s*:?-+:?\s*\|){5}$")
MAP_LAYER = re.compile(r"(?<![\w/])(" + "|".join(LAYERS) + r")/")
MapRow = namedtuple("MapRow", "layer owns may_include driver rule")


class Map(unittest.TestCase):
    def rows(self, text):
        lines = text.splitlines()
        self.assertEqual(lines.count(MAP_HEADER), 1, f"one table under {MAP_HEADER}")
        start = lines.index(MAP_HEADER) + 1
        self.assertRegex(lines[start], MAP_SEPARATOR)
        rows = []
        for line in lines[start + 1 :]:
            if not line.startswith("|"):
                break
            cells = [cell.strip().replace("`", "") for cell in line.strip().strip("|").split("|")]
            self.assertEqual(len(cells), len(MapRow._fields), line)
            rows.append(MapRow(*cells))
        return rows

    def check(self, rows):
        self.assertEqual([row.layer for row in rows], [f"{layer}/" for layer in LAYERS])
        for i, row in enumerate(rows):
            layer = row.layer.rstrip("/")
            self.assertEqual(below(layer), tuple(r.layer.rstrip("/") for r in rows[: i + 1]))
            previous = rows[i - 1].layer.rstrip("/") if i else layer
            self.assertEqual(MAP_LAYER.findall(row.may_include), [previous], row)
            if i:
                self.assertIn("and below", row.may_include, row)
            self.assertIn(row.driver, ("yes", "no"), row)
            self.assertEqual(row.driver == "yes", layer in HIERARCHY_LAYERS, row)
            named = [rule.strip() for rule in row.rule.split(",")]
            self.assertEqual("index" in named, layer in rules.INDEX_SOURCE_LAYERS, row)
            self.assertEqual("records" in named, layer in rules.RECORDS_LAYERS, row)

    def map_rows(self):
        with open(MAP) as f:
            return self.rows(f.read())

    def test_the_map_s_table_is_the_rule_table(self):
        rows = self.map_rows()
        self.check(rows)
        self.assertEqual(rows[LAYERS.index("io")].rule, "layer, door, codec, index, records")

    def test_a_missing_reordered_or_wrong_row_fails(self):
        rows = self.map_rows()
        swapped = rows[:3] + [rows[4], rows[3]] + rows[5:]
        io, ops, schedule = (LAYERS.index(layer) for layer in ("io", "ops", "schedule"))
        for name, mutated in [
            ("first row missing", rows[1:]),
            ("last row missing", rows[:-1]),
            ("a middle row missing", rows[:4] + rows[5:]),
            ("two rows swapped", swapped),
            ("a row too many", rows + [rows[-1]]),
            ("may include skips a row", rows[:5] + [rows[5]._replace(may_include="comm/ and below")] + rows[6:]),
            ("may include names no row", rows[:1] + [rows[1]._replace(may_include="the layers below")] + rows[2:]),
            ("may include only the row before", rows[:1] + [rows[1]._replace(may_include="util/")] + rows[2:]),
            ("driver() below the hierarchy", rows[:1] + [rows[1]._replace(driver="yes")] + rows[2:]),
            ("no driver() in the hierarchy", rows[:2] + [rows[2]._replace(driver="no")] + rows[3:]),
            ("driver() neither yes nor no", rows[:-1] + [rows[-1]._replace(driver="some")]),
            ("records missing where it binds", rows[:io] + [rows[io]._replace(rule="layer, door, codec, index")] + rows[io + 1 :]),
            ("index missing where it binds", rows[:io] + [rows[io]._replace(rule="layer, door, codec, records")] + rows[io + 1 :]),
            ("index where it binds no source", rows[:ops] + [rows[ops]._replace(rule="layer, door, index")] + rows[ops + 1 :]),
            ("records where it does not bind", rows[:schedule] + [rows[schedule]._replace(rule="layer, door, records")] + rows[schedule + 1 :]),
        ]:
            with self.subTest(name=name), self.assertRaises(AssertionError):
                self.check(mutated)

    def test_the_table_is_found_under_its_header_and_ends_at_the_first_other_line(self):
        rows = [
            f"| `{layer}/` | x | `{LAYERS[i - 1] if i else layer}/`{' and below' if i else ''} | "
            f"{'yes' if layer in HIERARCHY_LAYERS else 'no'} | layer"
            f"{', index' if layer in rules.INDEX_SOURCE_LAYERS else ''}{', records' if layer in rules.RECORDS_LAYERS else ''} |"
            for i, layer in enumerate(LAYERS)
        ]
        table = "\n".join([MAP_HEADER, "|---|---|---|---|---|"] + rows)
        self.check(self.rows("# Map\n\nProse.\n\n" + table + "\n\nMore prose | with a bar.\n| not | a | row |\n"))
        for text in ["no table\n", table + "\n\n" + table + "\n", table.replace("|---|", "| x |", 1)]:
            with self.subTest(text=text[:40]), self.assertRaises(AssertionError):
                self.rows(text)


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
            write(root, "CarpetX/src/io/io.hxx", "int a;\n")
            git(root, "add", "-A")
            git(root, "commit", "-q", "-m", "base")
            base = git(root, "rev-parse", "HEAD").strip()
            write(root, "T/src/a.cxx", '#include "../../CarpetX/src/driver.hxx"\n#include "../../CarpetX/src/schedule.hxx"\n')
            write(root, "T/src/b.cxx", "#define CARPETX_INTERNAL\n")
            write(root, "CarpetX/src/io/io.hxx", GUARD + "int a;\n")
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
            "no validity call outside valid.cxx, no directions in the sync layer, "
            "no firstvarindex addition outside GroupRecord::var, no active_levels_t, "
            "no active_levels outside traversal.cxx, "
            "no bare index int in a CarpetX header or io/ source, no group-table call in io/, "
            "no hierarchy access in a checkpoint codec, "
            "every internal include at or below its own layer, "
            "and the hierarchy has one door that util/ and traversal/ never use",
        )

    def test_counts(self):
        a = ("T/src/a.cxx", Finding(3, "include", '#include "../../CarpetX/src/driver.hxx"'))
        b = ("T/src/b.cxx", Finding(1, "marker", "#define CARPETX_INTERNAL"))
        c = ("CarpetX/src/hierarchy/hierarchy.cxx", Finding(7, "validity", "why.set_all(valid_t(), r);"))
        d = ("CarpetX/src/comm/comm.hxx", Finding(2, "directions", "const int *groups, const int *directions);"))
        e = ("CarpetX/src/hierarchy/valid.cxx", Finding(5, "varindex", "CCTK_FullVarName(groupdata.record.firstvarindex + vi),"))
        self.assertEqual(summary([a]), "✗ boundary: 1 private include")
        self.assertEqual(summary([a, a, b]), "✗ boundary: 2 private includes, 1 stray marker")
        self.assertEqual(summary([c]), "✗ boundary: 1 validity call outside valid.cxx")
        self.assertEqual(summary([b, c, c]), "✗ boundary: 1 stray marker, 2 validity calls outside valid.cxx")
        self.assertEqual(summary([d]), "✗ boundary: 1 directions parameter in the sync layer")
        self.assertEqual(summary([c, d, d]), "✗ boundary: 1 validity call outside valid.cxx, 2 directions parameters in the sync layer")
        self.assertEqual(summary([e]), "✗ boundary: 1 firstvarindex addition outside GroupRecord::var")
        self.assertEqual(summary([d, e, e]), "✗ boundary: 1 directions parameter in the sync layer, 2 firstvarindex additions outside GroupRecord::var")
        w = ("CarpetX/src/schedule/schedule.hxx", Finding(3, "window", "struct active_levels_t;"))
        i = ("CarpetX/src/comm/subcycling.hxx", Finding(4, "index", "void EnsureRKBuffers(int patch, int level, gi_t gi);"))
        self.assertEqual(summary([w]), "✗ boundary: 1 active_levels_t use")
        self.assertEqual(summary([i]), "✗ boundary: 1 bare index int in a CarpetX header or io/ source")
        self.assertEqual(summary([e, w, w, i, i]), "✗ boundary: 1 firstvarindex addition outside GroupRecord::var, 2 active_levels_t uses, 2 bare index ints in a CarpetX header or io/ source")
        t = ("CarpetX/src/schedule/evolve.cxx", Finding(9, "traversal", "active_levels = std::nullopt;"))
        self.assertEqual(summary([t]), "✗ boundary: 1 traversal use outside traversal.cxx")
        self.assertEqual(summary([w, t, t, i]), "✗ boundary: 1 active_levels_t use, 2 traversal uses outside traversal.cxx, 1 bare index int in a CarpetX header or io/ source")
        r = ("CarpetX/src/io/io_tsv.cxx", Finding(6, "records", "for (int gi = 0; gi < CCTK_NumGroups(); ++gi) {}"))
        self.assertEqual(summary([r]), "✗ boundary: 1 group-table call in io/")
        self.assertEqual(summary([r, i, r]), "✗ boundary: 1 bare index int in a CarpetX header or io/ source, 2 group-table calls in io/")
        k = ("CarpetX/src/io/io_openpmd_checkpoint.cxx", Finding(5, "codec", "for (const auto &patchdata : ghext->patches()) {"))
        self.assertEqual(summary([k]), "✗ boundary: 1 hierarchy use in a checkpoint codec")
        self.assertEqual(summary([i, k, k]), "✗ boundary: 1 bare index int in a CarpetX header or io/ source, 2 hierarchy uses in a checkpoint codec")
        self.assertEqual(summary([k, r]), "✗ boundary: 1 group-table call in io/, 1 hierarchy use in a checkpoint codec")
        g = ("CarpetX/src/schedule/evolve.cxx", Finding(12, "door", "for (auto &p : ghext->patches())"))
        self.assertEqual(summary([g]), "✗ boundary: 1 hierarchy use the door does not allow")
        self.assertEqual(summary([k, g, g]), "✗ boundary: 1 hierarchy use in a checkpoint codec, 2 hierarchy uses the door does not allow")
        y = ("CarpetX/src/comm/sync_restrict.cxx", Finding(3, "layer", '#include "schedule/schedule.hxx"'))
        self.assertEqual(summary([y]), "✗ boundary: 1 include outside the layer order")
        self.assertEqual(summary([g, y, y, k]), "✗ boundary: 1 hierarchy use in a checkpoint codec, 2 includes outside the layer order, 1 hierarchy use the door does not allow")

    def test_the_report_shape(self):
        a = ("T/src/a.cxx", Finding(3, "include", '#include "../../CarpetX/src/driver.hxx"'))
        b = ("T/src/b.cxx", Finding(1, "marker", "#define CARPETX_INTERNAL"))
        c = ("CarpetX/src/hierarchy/hierarchy.cxx", Finding(7, "validity", "why.set_all(valid_t(), r);"))
        lines = render([b, c, a]).split("\n")
        self.assertEqual(lines[0], "CarpetX/src/hierarchy/hierarchy.cxx:7   validity   why.set_all(valid_t(), r);")
        self.assertEqual(lines[1], 'T/src/a.cxx:3                           include    #include "../../CarpetX/src/driver.hxx"')
        self.assertEqual(lines[2], "T/src/b.cxx:1                           marker     #define CARPETX_INTERNAL")
        self.assertEqual(
            lines[-1],
            "✗ boundary: 1 private include, 1 stray marker, 1 validity call outside valid.cxx",
        )


SOURCE = "namespace CarpetX {\nvoid Evolve() {\n}\n} // namespace CarpetX\n"
PLANT = ("void Evolve() {\n", "void Evolve() {\n  for (auto &p : ghext->patches())\n")
PLANTED = SOURCE.replace(*PLANT)
DENIAL = "CarpetX/src/schedule/evolve.cxx:3   door       for (auto &p : ghext->patches())"
DOOR_SUMMARY = "✗ boundary: 1 hierarchy use the door does not allow"
LAYER_SUMMARY = "✗ boundary: 1 include outside the layer order"
SYNC_SOURCE = '#include "comm/comm.hxx"\n#include "hierarchy/driver.hxx"\n\nnamespace CarpetX {\nvoid sync() {\n}\n} // namespace CarpetX\n'
TRAVERSAL_SOURCE = '#include "traversal/traversal.hxx"\n#include "util/refuse.hxx"\n\nnamespace CarpetX {\nvoid scope() {\n}\n} // namespace CarpetX\n'
IO_SOURCE = '#include "io/selection.hxx"\n\nnamespace CarpetX {\nvoid OutputTSV(const cGH *cctkGH) {\n}\n} // namespace CarpetX\n'
PROJECT_FILES = {
    "CarpetX/interface.ccl": "IMPLEMENTS: Driver\n",
    "CarpetX/src/schedule/evolve.cxx": SOURCE,
    "CarpetX/src/hierarchy/hierarchy.cxx": SOURCE,
    "CarpetX/src/comm/sync_restrict.cxx": SYNC_SOURCE,
    "CarpetX/src/traversal/traversal.cxx": TRAVERSAL_SOURCE,
    "CarpetX/src/io/io_tsv.cxx": IO_SOURCE,
    "CarpetX/README": "not covered\n",
}
COUNTED_LOOP = "for (int gi = 0; gi < CCTK_NumGroups(); ++gi) {}"
COUNTED_PLANT = ("void OutputTSV(const cGH *cctkGH) {\n", "void OutputTSV(const cGH *cctkGH) {\n  " + COUNTED_LOOP + "\n")
COUNTED_DENIAL = [
    f"CarpetX/src/io/io_tsv.cxx:5   index      {COUNTED_LOOP}",
    f"CarpetX/src/io/io_tsv.cxx:5   records    {COUNTED_LOOP}",
]
COUNTED_SUMMARY = "✗ boundary: 1 bare index int in a CarpetX header or io/ source, 1 group-table call in io/"
UPWARD_PLANT = ('#include "hierarchy/driver.hxx"\n', '#include "hierarchy/driver.hxx"\n#include "schedule/schedule.hxx"\n')
UPWARD_DENIAL = 'CarpetX/src/comm/sync_restrict.cxx:3   layer      #include "schedule/schedule.hxx"'
BARE_PLANT = ("namespace CarpetX {\n", '#include "driver.hxx"\n\nnamespace CarpetX {\n')
DRIVER_PLANT = ("void scope() {\n", "void scope() {\n  const GHExt &ext = driver();\n")
DRIVER_DENIAL = "CarpetX/src/traversal/traversal.cxx:6   door       const GHExt &ext = driver();"


def project(test):
    tmp = tempfile.TemporaryDirectory()
    test.addCleanup(tmp.cleanup)
    root = os.path.realpath(tmp.name)
    for name, text in PROJECT_FILES.items():
        write(root, name, text)
    return root


def edit_payload(root, name, old, new):
    return {"tool_name": "Edit", "tool_input": {"file_path": os.path.join(root, name), "old_string": old, "new_string": new}, "cwd": root}


def denial(test, output):
    test.assertTrue(output, "the hook printed nothing")
    decision = json.loads(output)["hookSpecificOutput"]
    test.assertEqual(decision["hookEventName"], "PreToolUse")
    test.assertEqual(decision["permissionDecision"], "deny")
    return decision["permissionDecisionReason"].split("\n")


def skip_message(test, output):
    message = json.loads(output)
    test.assertEqual(list(message), ["systemMessage"])
    test.assertTrue(message["systemMessage"].startswith("boundary check skipped: "), message)


class Hooks(unittest.TestCase):
    def setUp(self):
        self.root = project(self)

    def run_edit(self, payload):
        out = io.StringIO()
        raw = payload if isinstance(payload, str) else json.dumps(payload)
        self.assertEqual(hooks.hook_edit(raw, out, root=self.root), 0)
        return out.getvalue()

    def edit(self, name, old, new):
        return self.run_edit(edit_payload(self.root, name, old, new))

    def write_tool(self, name, content):
        return self.run_edit({"tool_name": "Write", "tool_input": {"file_path": os.path.join(self.root, name), "content": content}})

    def run_stop(self, payload):
        out = io.StringIO()
        self.assertEqual(hooks.hook_stop(json.dumps(payload), out, root=self.root), 0)
        return out.getvalue()

    def test_a_planted_ghext_is_denied_under_door(self):
        lines = denial(self, self.edit("CarpetX/src/schedule/evolve.cxx", *PLANT))
        self.assertEqual(lines[0], DENIAL)
        self.assertEqual(lines[1:-1], RULE.split("\n"))
        self.assertEqual(lines[-1], DOOR_SUMMARY)

    def test_the_door_file_may_name_it(self):
        self.assertEqual(self.edit("CarpetX/src/hierarchy/hierarchy.cxx", *PLANT), "")

    def test_an_upward_include_is_denied_under_layer(self):
        lines = denial(self, self.edit("CarpetX/src/comm/sync_restrict.cxx", *UPWARD_PLANT))
        self.assertEqual(lines[0], UPWARD_DENIAL)
        self.assertEqual(lines[1:-1], RULE.split("\n"))
        self.assertEqual(lines[-1], LAYER_SUMMARY)

    def test_an_unqualified_private_include_is_denied_under_layer_anywhere(self):
        for name, line in [
            ("CarpetX/src/schedule/evolve.cxx", 1),
            ("CarpetX/src/hierarchy/hierarchy.cxx", 1),
            ("CarpetX/src/comm/sync_restrict.cxx", 4),
            ("CarpetX/src/traversal/traversal.cxx", 4),
        ]:
            with self.subTest(name=name):
                lines = denial(self, self.edit(name, *BARE_PLANT))
                self.assertEqual(lines[0], f'{name}:{line}   layer      #include "driver.hxx"')
                self.assertEqual(lines[-1], LAYER_SUMMARY)

    def test_a_driver_call_below_the_hierarchy_is_denied_under_door(self):
        lines = denial(self, self.edit("CarpetX/src/traversal/traversal.cxx", *DRIVER_PLANT))
        self.assertEqual(lines[0], DRIVER_DENIAL)
        self.assertEqual(lines[-1], DOOR_SUMMARY)

    def test_a_counted_group_loop_in_io_is_denied_under_index_and_records(self):
        lines = denial(self, self.edit("CarpetX/src/io/io_tsv.cxx", *COUNTED_PLANT))
        self.assertEqual(lines[:2], COUNTED_DENIAL)
        self.assertEqual(lines[2:-1], RULE.split("\n"))
        self.assertEqual(lines[-1], COUNTED_SUMMARY)

    def test_a_record_walk_in_io_passes(self):
        walk = "for (const GroupRecord &record : driver().records()) {}"
        self.assertEqual(self.edit("CarpetX/src/io/io_tsv.cxx", COUNTED_PLANT[0], COUNTED_PLANT[0] + "  " + walk + "\n"), "")

    def test_what_the_layers_allow_passes(self):
        self.assertEqual(self.edit("CarpetX/src/schedule/evolve.cxx", "namespace CarpetX {\n", '#include "comm/comm.hxx"\n\nnamespace CarpetX {\n'), "")
        self.assertEqual(self.edit("CarpetX/src/comm/sync_restrict.cxx", "void sync() {\n", "void sync() {\n  const GHExt &ext = driver();\n"), "")
        self.assertEqual(self.edit("CarpetX/src/traversal/traversal.cxx", '#include "util/refuse.hxx"\n', '#include "util/refuse.hxx"\n#include "internal.hxx"\n'), "")

    def test_only_what_the_edit_adds_is_new(self):
        write(self.root, "CarpetX/src/schedule/evolve.cxx", PLANTED)
        self.assertEqual(self.edit("CarpetX/src/schedule/evolve.cxx", "void Evolve()", "void Initialise()"), "")
        lines = denial(self, self.edit("CarpetX/src/schedule/evolve.cxx", "}\n} //", "  assert(!ghext);\n}\n} //"))
        self.assertEqual(lines[0], "CarpetX/src/schedule/evolve.cxx:4   door       assert(!ghext);")

    def test_a_write_and_a_new_file_are_judged_whole(self):
        self.assertEqual(denial(self, self.write_tool("CarpetX/src/schedule/evolve.cxx", PLANTED))[0], DENIAL)
        self.assertEqual(denial(self, self.edit("CarpetX/src/schedule/new.cxx", "", PLANTED))[0], DENIAL.replace("evolve", "new"))
        self.assertEqual(self.write_tool("CarpetX/src/schedule/new.cxx", SOURCE), "")

    def test_an_edit_that_does_not_apply_is_left_to_the_edit_tool(self):
        self.assertEqual(self.edit("CarpetX/src/schedule/evolve.cxx", "int nope;\n", "ghext->patches();\n"), "")

    def test_files_the_check_does_not_read_pass(self):
        self.assertEqual(self.edit("CarpetX/README", "not covered\n", "ghext->patches();\n"), "")
        self.assertEqual(self.write_tool("CarpetX/schedule.ccl", "# ghext\n"), "")
        self.assertEqual(self.write_tool("Other/src/a.cxx", PLANTED), "")
        with tempfile.TemporaryDirectory() as other:
            self.assertEqual(self.run_edit(edit_payload(other, "CarpetX/src/a.cxx", "", PLANTED)), "")

    def test_a_relative_path_is_taken_from_cwd(self):
        payload = edit_payload(self.root, "CarpetX/src/schedule/evolve.cxx", *PLANT)
        payload["tool_input"]["file_path"] = "CarpetX/src/schedule/evolve.cxx"
        self.assertEqual(denial(self, self.run_edit(payload))[0], DENIAL)

    def test_a_bad_payload_is_a_system_message(self):
        for payload in [
            "not json",
            "[]",
            {"tool_name": "Edit", "tool_input": {"file_path": os.path.join(self.root, "CarpetX/src/schedule/evolve.cxx"), "new_string": "x"}},
            {"tool_name": "Teleport", "tool_input": {"file_path": os.path.join(self.root, "CarpetX/src/schedule/evolve.cxx")}},
        ]:
            with self.subTest(payload=payload):
                skip_message(self, self.run_edit(payload))

    def test_the_stop_hook_blocks_once_on_what_the_branch_added(self):
        with mock.patch.dict(os.environ, ISOLATED_GIT):
            git(self.root, "init", "-q", "-b", "dev")
            git(self.root, "add", "-A")
            git(self.root, "commit", "-q", "-m", "base")
            git(self.root, "switch", "-q", "-c", "work")
            self.assertEqual(self.run_stop({"stop_hook_active": False}), "")
            write(self.root, "CarpetX/src/schedule/evolve.cxx", PLANTED)
            git(self.root, "commit", "-q", "-am", "plant")
            decision = json.loads(self.run_stop({"stop_hook_active": False}))
            self.assertEqual(sorted(decision), ["decision", "reason"])
            self.assertEqual(decision["decision"], "block")
            lines = decision["reason"].split("\n")
            self.assertEqual((lines[0], lines[-1]), (DENIAL, DOOR_SUMMARY))
            self.assertEqual(self.run_stop({"stop_hook_active": True}), "")
            write(self.root, "CarpetX/src/schedule/evolve.cxx", SOURCE)
            self.assertEqual(self.run_stop({"stop_hook_active": False}), "")

    def test_a_bad_stop_payload_or_no_repository_is_a_system_message(self):
        out = io.StringIO()
        hooks.hook_stop("[]", out, root=self.root)
        skip_message(self, out.getvalue())
        skip_message(self, self.run_stop({"stop_hook_active": False}))


def configured(event):
    with open(SETTINGS) as f:
        entries = json.load(f)["hooks"].get(event, [])
    return [
        (entry.get("matcher"), hook["command"])
        for entry in entries
        for hook in entry["hooks"]
        if "scripts/boundary" in hook["command"]
    ]


def copy_tools(root):
    for tool in ("boundary", "comments"):
        shutil.copytree(
            os.path.join(os.path.dirname(TOOL), tool),
            os.path.join(root, "scripts", tool),
            ignore=shutil.ignore_patterns("__pycache__"),
        )


@unittest.skipUnless(os.path.isfile(SETTINGS), "needs .claude/settings.json")
class Settings(unittest.TestCase):
    def setUp(self):
        self.root = project(self)

    def run_command(self, command, project_dir, payload):
        env = {k: v for k, v in os.environ.items() if k != "CLAUDE_PROJECT_DIR"}
        if project_dir is not None:
            env["CLAUDE_PROJECT_DIR"] = project_dir
        return subprocess.run(["sh", "-c", command], input=json.dumps(payload), env=env, capture_output=True, text=True)

    def test_three_entries_run_the_boundary_hooks(self):
        [(matcher, edit)] = configured("PreToolUse")
        self.assertEqual(matcher, "Edit|Write")
        self.assertTrue(edit.endswith('python3 "$d" hook-edit'), edit)
        for event in ("Stop", "SubagentStop"):
            with self.subTest(event=event):
                [(matcher, stop)] = configured(event)
                self.assertIsNone(matcher)
                self.assertEqual(stop, edit.replace("hook-edit", "hook-stop"))

    def test_the_commands_are_silent_where_the_tool_is_absent(self):
        for event, payload in [
            ("PreToolUse", edit_payload(self.root, "CarpetX/src/schedule/evolve.cxx", *PLANT)),
            ("Stop", {"stop_hook_active": False}),
        ]:
            [(_, command)] = configured(event)
            for project_dir in [self.root, None]:
                with self.subTest(event=event, project_dir=project_dir):
                    result = self.run_command(command, project_dir, payload)
                    self.assertEqual((result.returncode, result.stdout, result.stderr), (0, "", ""))

    def test_the_edit_command_runs_the_project_s_tool(self):
        [(_, command)] = configured("PreToolUse")
        copy_tools(self.root)
        result = self.run_command(command, self.root, edit_payload(self.root, "CarpetX/src/schedule/evolve.cxx", *PLANT))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(denial(self, result.stdout)[0], DENIAL)
        result = self.run_command(command, self.root, edit_payload(self.root, "CarpetX/src/hierarchy/hierarchy.cxx", *PLANT))
        self.assertEqual((result.returncode, result.stdout), (0, ""), result.stderr)
        result = self.run_command(command, self.root, edit_payload(self.root, "CarpetX/src/comm/sync_restrict.cxx", *UPWARD_PLANT))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(denial(self, result.stdout)[0], UPWARD_DENIAL)

    def test_the_stop_command_runs_the_project_s_tool(self):
        [(_, command)] = configured("Stop")
        copy_tools(self.root)
        with mock.patch.dict(os.environ, ISOLATED_GIT):
            git(self.root, "init", "-q", "-b", "dev")
            git(self.root, "add", "-A")
            git(self.root, "commit", "-q", "-m", "base")
            write(self.root, "CarpetX/src/schedule/evolve.cxx", PLANTED)
            result = self.run_command(command, self.root, {"stop_hook_active": False})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["decision"], "block")



class Diagram(unittest.TestCase):
    def test_the_map_s_stack_is_the_generated_diagram(self):
        with open(os.path.join(PROJECT, diagram.MAP)) as f:
            text = f.read()
        found = diagram.blocks(text)
        self.assertEqual(len(found), 1, f"one mermaid block opening with {diagram.MARK!r}")
        start, end = found[0]
        self.assertEqual("\n".join(text.split("\n")[start:end]) + "\n", diagram.diagram(), "run python3 scripts/boundary diagram --write")
        self.assertEqual(diagram.rewrite(text), text)

    def test_the_diagram_stacks_the_layers_top_down_with_the_door_line(self):
        lines = diagram.diagram().splitlines()
        self.assertEqual(lines[:2], [diagram.MARK, "flowchart TB"])
        self.assertIn("  " + " --> ".join(reversed(LAYERS)), lines)
        door = lines[lines.index('  subgraph door["driver() allowed"]') + 1 : lines.index("  end")]
        self.assertEqual([line.split("[")[0].strip() for line in door], list(reversed(HIERARCHY_LAYERS)))
        nodoor = lines[lines.index('  subgraph nodoor["no driver()"]') + 1 :]
        nodoor = nodoor[: nodoor.index("  end")]
        self.assertEqual([line.split("[")[0].strip() for line in nodoor], list(reversed(rules.UNDER_HIERARCHY)))
        for layer in LAYERS:
            with self.subTest(layer=layer):
                self.assertEqual(f'"{layer}/ no header"' in diagram.diagram(), layer in diagram.NO_HEADER_LAYERS)

    def test_a_layer_without_a_header_is_in_no_header_layers(self):
        self.assertEqual(diagram.headerless(PROJECT), diagram.NO_HEADER_LAYERS)

    def test_a_dashed_edge_is_declared_low_and_defined_high(self):
        for low, high, names in diagram.LINK_TIME:
            self.assertLess(LAYERS.index(low), LAYERS.index(high))
            for name in names:
                with self.subTest(low=low, high=high, name=name):
                    self.assertTrue(diagram.declares_and_defines(PROJECT, low, high, name))
                    self.assertIn(f'  {low} -. "', diagram.diagram())

    def test_rewrite_replaces_only_the_marked_block(self):
        stale = "```mermaid\n" + diagram.MARK + "\nflowchart TB\n  stale\n```\n"
        other = "```mermaid\nflowchart LR\n  a --> b\n```\n"
        text = "# map\n\n" + other + "\nprose\n\n" + stale + "\ntail\n"
        self.assertEqual(diagram.rewrite(text), "# map\n\n" + other + "\nprose\n\n```mermaid\n" + diagram.diagram() + "```\n\ntail\n")
        with self.assertRaises(ValueError):
            diagram.rewrite(other)
        with self.assertRaises(ValueError):
            diagram.rewrite(stale + stale)


if __name__ == "__main__":
    unittest.main()
