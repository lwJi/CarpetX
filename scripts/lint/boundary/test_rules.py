from __future__ import annotations

import io
import json
import os
import re
import subprocess
import tempfile
import unittest
from unittest import mock

import hooks
from files import PROJECT, Lang
from run import check_since

from . import check, doc, rules
from .rules import HIERARCHY_LAYERS, LAYERS, MAKE_DEFN, RULE, Finding, below, findings, layer_of, layer_path, new_findings, render, summary

ISOLATED_GIT = {"GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"}
GUARD = '#ifndef CARPETX_INTERNAL\n#error "CarpetX internal header; include <carpetx.hxx> instead"\n#endif\n'


def reported(name, text, lang=Lang.CXX):
    return [(f.rule, f.text) for f in findings(name, text, lang)]


def since_findings(root, base):
    [(_, found)] = check_since(root, base, [check.CHECK])
    return [(name, f) for name, verdict in found for f in verdict]


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
                self.assertEqual(reported("CarpetX/src/hierarchy/hierarchy.cxx", "  " + text), [])

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
        "  int SyncGroupsByDirI(const window_t window,\n"
        "                       const cGH *restrict cctkGH, int numgroups,\n"
        "                       const int *groups, const int *directions);\n"
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
                self.assertEqual(reported("CarpetX/src/schedule/evolve.cxx", "  " + text), [])

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

    def test_the_integrator_is_a_slot(self):
        for line in [
            "std::optional<rk_config_t> integrator_config;",
            "integrator_config = config;",
            "return integrator_config->nstages();",
            "std::atomic<bool> integrator_sealed{false};",
            "if (integrator_sealed) return;",
            "integrator_sealed.store(true, std::memory_order_relaxed);",
        ]:
            for name in ["CarpetX/src/api/api_rk.cxx", "CarpetX/src/hierarchy/group_tags.cxx", "CarpetX/src/hierarchy/driver.hxx"]:
                with self.subTest(name=name, line=line):
                    self.assertEqual(reported(name, "  " + line + "\n"), [("traversal", line)])
            for name in ["CarpetX/src/traversal/traversal.cxx", "CarpetX/src/traversal/traversal.hxx"]:
                with self.subTest(name=name, line=line):
                    self.assertEqual(reported(name, line + "\n"), [])

    def test_the_integrator_readers_are_not_the_slot(self):
        for text in [
            "record_integrator(config);\n",
            "const std::optional<rk_config_t> &config = integrator();\n",
            "const std::optional<rk_config_t> &integrator = CarpetX::integrator();\n",
            "return integrator && integrator->is_integrated(gi);\n",
            "my_integrator_config = 1;\n",
            "bool integrator_sealed_early;\n",
        ]:
            for name in ["CarpetX/src/api/api_rk.cxx", "CarpetX/src/hierarchy/group_tags.cxx"]:
                with self.subTest(name=name, text=text):
                    self.assertEqual(reported(name, text), [])


class Index(unittest.TestCase):
    BARE = [
        "void EnsureRKBuffers(int patch, int level, gi_t gi);",
        "const GroupRecord &record(const int gi, const std::source_location site) const;",
        "void Restrict(const cGH *cctkGH, int min_level, int max_level,",
        "bool window_is_restricted(int min_level, int max_level);",
        "valid_t valid(group_t group, tl_t tl, int vi);",
        "void prolongate_ghosts(window_t window, std::span<const gi_t> groups, int tl = -1);",
        "int min_patch, max_patch;",
        "CCTK_INT GetDomainSpecification(const CCTK_INT patch, const CCTK_INT size);",
        "void solve(CCTK_INT gi, CCTK_INT tl);",
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

    def test_index_in_every_carpetx_file_and_no_other(self):
        for layer in LAYERS:
            for name in [layer_path(layer, "x.cxx"), layer_path(layer, "x.hxx")]:
                with self.subTest(name=name):
                    self.assertEqual(reported(name, "  " + self.BARE[0] + "\n"), [("index", self.BARE[0])])
        for name in ["CarpetX/src/internal.hxx", "CarpetX/src/include/carpetx_rk.hxx"]:
            with self.subTest(name=name):
                self.assertEqual(reported(name, self.BARE[0] + "\n"), [("index", self.BARE[0])])
        for name in ["ODESolvers/src/solve.hxx", "TestCarpetXAPI/src/api_test.cxx"]:
            with self.subTest(name=name):
                self.assertEqual(reported(name, self.BARE[0] + "\n"), [])
        self.assertEqual(reported("CarpetX/src/io/make.code.defn", "X = int level\n", Lang.MAKE), [])

    COUNTED = [
        "for (int gi = 0; gi < numgroups; ++gi) {",
        "const int level = leveldata.level.index();",
        "for (int vi = 0; vi < record.numvars; ++vi)",
        "std::string make_varname(int gi, int vi, level_t level) {",
        "num_components(int patch, int level) const {",
        "for (int level = 0; level < patchdata.num_levels(); ++level) {",
        "for (int patch = 0; patch < ext.num_patches(); ++patch) {",
        "CCTK_INT level = 0;",
    ]

    def test_a_source_in_every_layer_is_judged_like_a_header(self):
        for line in self.COUNTED:
            for name in ["CarpetX/src/comm/sync_restrict.cxx", "CarpetX/src/ops/linsolve.cxx", "CarpetX/src/ops/reduction.cxx", "CarpetX/src/schedule/evolve.cxx", "CarpetX/src/io/io_tsv.cxx", "CarpetX/src/hierarchy/loops.cxx"]:
                with self.subTest(name=name, line=line):
                    self.assertEqual(reported(name, "  " + line + "\n"), [("index", line)])

    def test_the_handles_pass_in_a_source(self):
        text = (
            "for (const GroupRecord &record : selection.records(group_kind::grid_function)) {\n"
            "  for (const vi_t vi : record.variables())\n"
            "    names.at(vi.index()) = CCTK_FullVarName(record.var(vi).index());\n"
            "}\n"
            "const level_t level = leveldata.level;\n"
            "for (const level_t level : patchdata.level_handles()) {\n"
            "  geoms.at(level.index()) = patchdata.amrcore->Geom(level.index());\n"
            "}\n"
            "for (const patch_t patch : ext.patch_handles())\n"
            "  containers.at(patch.index());\n"
            "for (int lev = 0; lev < max_levels; ++lev) {\n"
            "}\n"
        )
        for name in ["CarpetX/src/io/io.cxx", "CarpetX/src/ops/linsolve.cxx", "CarpetX/src/schedule/evolve.cxx"]:
            with self.subTest(name=name):
                self.assertEqual(reported(name, text), [])

    FIXED_DEFINITIONS = [
        ("CarpetX/src/api/api.cxx", "window_t window_t::levels(const int min_level, const int max_level,\n                          const std::source_location site) {\n"),
        ("CarpetX/src/api/api.cxx", "window_t window_t::narrow(const int min_level, const int max_level,\n                          const std::source_location site) const {\n"),
        ("CarpetX/src/schedule/storage.cxx", "int QueryGroupStorageB(const cGH *cctkGH, int gi, const char *groupname) {\n"),
        ("CarpetX/src/schedule/storage.cxx", "int GroupStorageIncrease(const cGH *cctkGH, int gi, const int *tls) {\n"),
        ("CarpetX/src/schedule/storage.cxx", "int GroupStorageDecrease(const cGH *cctkGH, int gi, const int *tls) {\n"),
        ("CarpetX/src/flesh/startup.cxx", "const int *ArrayGroupSizeB(const cGH *cctkGH, int dir, int gi,\n                           const char *groupname) {\n"),
        ("CarpetX/src/flesh/startup.cxx", "int GroupDynamicData(const cGH *cctkGH, int gi, cGroupDynamicData *data) {\n"),
        ("CarpetX/src/flesh/aliased.cxx", "extern \"C\" CCTK_INT\nCarpetX_GetDomainSpecification(const CCTK_INT patch, const CCTK_INT size,\n                               CCTK_REAL *restrict const spacing) {\n"),
        ("CarpetX/src/flesh/aliased.cxx", "extern \"C\" CCTK_INT CarpetX_GetBoundarySizesAndTypes(\n    const void *const cctkGH_, const CCTK_INT patch, const CCTK_INT size,\n    CCTK_INT *restrict const is_physbnd) {\n"),
        ("CarpetX/src/comm/sync_restrict.cxx", "int SyncGroupsOnActiveLevels(const cGH *restrict const cctkGH,\n                             const int gi) {\n"),
        ("CarpetX/src/hierarchy/group_tags.cxx", "    const int ierr = CCTK_TraverseString(\n        var_set_string,\n        [](const int vi, const char *const optstring, void *argptr) {\n"),
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

    def test_a_flesh_or_amrex_fixed_definition_passes(self):
        for name, text in self.FIXED_DEFINITIONS:
            with self.subTest(name=name, text=text):
                self.assertEqual(reported(name, text), [])
                self.assertEqual(reported(name, "  const int level = leveldata.level.index();\n"), [("index", "const int level = leveldata.level.index();")])

    def test_the_exemption_ends_with_the_fixed_signature(self):
        text = "int QueryGroupStorageB(const cGH *cctkGH, int gi, const char *groupname) {\n  for (int level = 0; level < 2; ++level)\n"
        self.assertEqual(reported("CarpetX/src/schedule/storage.cxx", text), [("index", "for (int level = 0; level < 2; ++level)")])
        text = "window_t window_t::levels(const int min_level, const int max_level) {\n  const int patch = 0;\n"
        self.assertEqual(reported("CarpetX/src/api/api.cxx", text), [("index", "const int patch = 0;")])

    def test_a_fixed_definition_passes_only_in_its_own_file(self):
        for name, text in self.FIXED_DEFINITIONS:
            for other in ["CarpetX/src/io/io.cxx", "CarpetX/src/ops/linsolve.cxx", "CarpetX/src/hierarchy/hierarchy.cxx"]:
                with self.subTest(name=name, other=other):
                    self.assertEqual(reported(other, text), self.bare_lines(text))

    def test_cctk_int_is_an_int(self):
        for line in ["const CCTK_INT patch = patches.at(n);", "CCTK_INT gi;", "void f(const CCTK_INT level);"]:
            for name in ["CarpetX/src/ops/interpolate.cxx", "CarpetX/src/ops/interp.hxx", "CarpetX/src/flesh/startup.cxx"]:
                with self.subTest(name=name, line=line):
                    self.assertEqual(reported(name, "  " + line + "\n"), [("index", line)])

    def test_other_ints_are_not_indices(self):
        for text in [
            "int levels;\n",
            "int level_count, gi_rhs, tls;\n",
            "void f(int patches, int min_levels, int vi0);\n",
            "int index() const;\n",
            "int min_level_, max_level_;\n",
            "void f(level_t level, gi_t gi, tl_t tl, vi_t vi, patch_t patch);\n",
            "const CCTK_INT nvars, const CCTK_INT *restrict const varinds,\n",
            "extern \"C\" void CarpetX_SolvePoisson(const CCTK_INT gi_sol,\n",
            "CCTK_INT num_patches = 1;\n",
            "MY_CCTK_INT level;\n",
        ]:
            with self.subTest(text=text):
                self.assertEqual(reported("CarpetX/src/hierarchy/driver.hxx", text), [])

    def test_line_numbers(self):
        text = "int a;\n\n  " + self.BARE[0] + "\n"
        self.assertEqual(findings("CarpetX/src/comm/subcycling.hxx", text, Lang.CXX), [Finding(3, "index", self.BARE[0])])
        self.assertEqual(findings("CarpetX/src/comm/subcycling.cxx", text, Lang.CXX), [Finding(3, "index", self.BARE[0])])


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
                self.assertEqual(reported("CarpetX/src/io/io_meta.cxx", "  " + text), [])

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

    MUTABLE_CALLS = [
        "GHExt &ext = mutable_driver();",
        "auto &leveldata = mutable_driver().level(leveldata.level);",
        "mutable_driver ( ).record(gi).array().rotate_timelevels();",
        "GHExt &mutable_driver();",
        "// mutable_driver() is the write side",
    ]

    def test_the_hierarchy_calls_mutable_driver(self):
        for line in self.MUTABLE_CALLS:
            for name in [
                "CarpetX/src/hierarchy/hierarchy.cxx",
                "CarpetX/src/hierarchy/valid.cxx",
                "CarpetX/src/hierarchy/operations.hxx",
                layer_path("hierarchy", "x.cxx"),
            ]:
                with self.subTest(name=name, line=line):
                    self.assertEqual(reported(name, "  " + line + "\n"), [])

    def test_mutable_driver_is_reported_outside_the_hierarchy(self):
        for line in self.MUTABLE_CALLS:
            for name in [layer_path(layer, "x.cxx") for layer in LAYERS if layer != "hierarchy"] + [
                "CarpetX/src/comm/subcycling.cxx",
                "CarpetX/src/io/checkpoint.cxx",
                "CarpetX/src/schedule/evolve.cxx",
                "CarpetX/src/api/api.cxx",
                "CarpetX/src/regrid/regrid.cxx",
                "CarpetX/src/hierarchy/sub/x.cxx",
                "CarpetX/src/internal.hxx",
            ]:
                with self.subTest(name=name, line=line):
                    self.assertEqual(reported(name, "  " + line + "\n"), [("door", line)])

    def test_another_thorn_and_the_public_headers_may_call_mutable_driver(self):
        for name in ["ODESolvers/src/solve.cxx", "CarpetX/src/include/carpetx.hxx"]:
            with self.subTest(name=name):
                self.assertEqual(reported(name, self.MUTABLE_CALLS[0] + "\n"), [])

    def test_a_longer_name_or_an_argument_is_not_mutable_driver(self):
        for text in [
            "my_mutable_driver();\n",
            "mutable_driver_handle();\n",
            "mutable_drivers().clear();\n",
            "auto *door = &mutable_driver;\n",
            "mutable_driver(level);\n",
        ]:
            with self.subTest(text=text):
                self.assertEqual(reported("CarpetX/src/comm/subcycling.cxx", text), [])

    def test_a_line_naming_both_doors_is_one_finding(self):
        line = "assert(&driver() == &mutable_driver());"
        self.assertEqual(reported("CarpetX/src/comm/subcycling.cxx", line + "\n"), [("door", line)])
        self.assertEqual(reported("CarpetX/src/hierarchy/hierarchy.cxx", line + "\n"), [])


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


class Prototype(unittest.TestCase):
    PROTOTYPES = [
        "void ShutdownADIOS2();",
        "int InitGH(cGH *cctkGH);",
        "void *SetupGH(tFleshConfig *fc, int convLevel, cGH *cctkGH);",
        "const int *ArrayGroupSizeB(const cGH *cctkGH, int dir, const char *groupname);",
        'extern "C" int CarpetX_Startup();',
        "inline void advance(const GHExt &ext) ;",
        "template <class T> void apply(const T &x);",
        "template  <class T> void apply(const T &x);",
        "bool operator==(const valid_t &a, const valid_t &b);",
        "std::ostream &operator<<(std::ostream &os, valid_t v);",
    ]

    def test_a_column_0_prototype_is_reported(self):
        for line in self.PROTOTYPES:
            for name in [layer_path(layer, "x.cxx") for layer in LAYERS] + ["CarpetX/src/flesh/startup.cxx", "CarpetX/src/extra/x.cxx"]:
                with self.subTest(name=name, line=line):
                    self.assertEqual(reported(name, line + "\n"), [("prototype", line)])

    def test_a_wrapped_prototype_is_reported_on_its_last_line(self):
        text = "int ScheduleTraverseGH(cGH *cctkGH,\n                       const char *where);\n"
        self.assertEqual(findings("CarpetX/src/flesh/startup.cxx", text, Lang.CXX), [Finding(2, "prototype", "const char *where);")])
        text = (
            "int a;\n"
            'extern "C" CCTK_INT CarpetX_InterpGridArrays(\n'
            "    cGH const *const cctkGH, int const N_dims,\n"
            "    void *const output_arrays[]);\n"
        )
        self.assertEqual(findings("CarpetX/src/flesh/startup.cxx", text, Lang.CXX), [Finding(4, "prototype", "void *const output_arrays[]);")])

    def test_what_precedes_it_on_lines_of_no_code_does_not_hide_it(self):
        for before in [
            "\n",
            "\n\n",
            "#ifdef HAVE_CAPABILITY_ADIOS2\n",
            "#include <cctk.h>\n\n",
            "////////////////////////////////////////////////////////////////////////////////\n\n",
            "// the shutdown of the ADIOS2 writer\n",
            "  // an indented comment\n",
        ]:
            with self.subTest(before=before):
                text = "} // namespace\n" + before + "void ShutdownADIOS2();\n"
                self.assertEqual(reported("CarpetX/src/flesh/startup.cxx", text), [("prototype", "void ShutdownADIOS2();")])

    def test_a_static_forward_declaration_or_a_static_assert_is_not(self):
        text = (
            "static void restrict_level(const GHExt::PatchData::LevelData &coarse,\n"
            "                           const GHExt::PatchData::LevelData &fine);\n"
            "static int counter(const cGH *cctkGH);\n"
            "static_assert(record_takes<gi_t>);\n"
            "static_assert(\n"
            "    !loop_parallel_takes<std::function<void(GHExt::PatchData::LevelData &)>>);\n"
            "\n"
            "static_assert(std::is_same_v<decltype(driver()), const GHExt &>);\n"
        )
        for name in ["CarpetX/src/comm/sync_restrict.cxx", "CarpetX/src/flesh/compile_checks.cxx"]:
            with self.subTest(name=name):
                self.assertEqual(reported(name, text), [])

    def test_an_explicit_instantiation_is_not(self):
        text = (
            "template void BoundaryCondition::apply_on_face<NEG, NEG, INT>();\n"
            "template void prolongate<2>(const amrex::MultiFab &src, amrex::MultiFab &dst);\n"
            "extern template void prolongate<3>(const amrex::MultiFab &src,\n"
            "                                   amrex::MultiFab &dst);\n"
        )
        self.assertEqual(reported("CarpetX/src/comm/boundaries_impl_neg_neg_int.cxx", text), [])

    def test_a_definition_is_not(self):
        text = (
            "void ShutdownIO() {\n"
            "  ShutdownADIOS2();\n"
            "  ShutdownOpenPMD();\n"
            "}\n"
            "void ShutdownADIOS2() {}\n"
            "int Barrier(const cGH *restrict cctkGH) { return 0; }\n"
            "const int *ArrayGroupSizeB(const cGH *cctkGH, int dir,\n"
            "                           const char *groupname) {\n"
            "  return nullptr;\n"
            "}\n"
        )
        for name in ["CarpetX/src/io/io.cxx", "CarpetX/src/flesh/startup.cxx"]:
            with self.subTest(name=name):
                self.assertEqual(reported(name, text), [])

    def test_an_indented_call_or_member_is_not(self):
        text = (
            'extern "C" int CarpetX_Shutdown() {\n'
            "  ShutdownIO();\n"
            '  CCTK_VINFO("Shutdown");\n'
            "  destroy_hierarchy();\n"
            "}\n"
            "struct carpetx_adios2_t {\n"
            "  void OutputADIOS2(const cGH *cctkGH);\n"
            "private:\n"
            "  void flush(const cGH *cctkGH);\n"
            "protected:\n"
            "  static void reset();\n"
            "};\n"
        )
        self.assertEqual(reported("CarpetX/src/io/io_adios2.cxx", text), [])

    def test_a_call_or_an_assignment_at_column_0_is_not(self):
        for text in [
            "refuse(site, msg);\n",
            "amrex::Finalize(pamrex);\n",
            "CCTK_FILEVERSION(CarpetX_flesh_startup_cxx);\n",
            "x = f(1);\n",
            "const GHExt &ext = driver();\n",
            "containers.at(patch.index());\n",
            "p->rotate_timelevels(tl);\n",
            "yaml << YAML::LocalTag(tag);\n",
            "return integrator && integrator->is_integrated(gi);\n",
            "return f(x);\n",
            "else g(x);\n",
        ]:
            with self.subTest(text=text):
                self.assertEqual(reported("CarpetX/src/flesh/startup.cxx", text), [])

    def test_a_preprocessor_or_comment_line_is_not_a_declaration(self):
        text = "#define CHECK(x) assert(x);\n// then ShutdownADIOS2();\n#define CALL(f) \\\n  f(x);\n"
        self.assertEqual(reported("CarpetX/src/flesh/startup.cxx", text), [])

    def test_a_header_another_thorn_and_a_make_file_are_not_checked(self):
        line = self.PROTOTYPES[0] + "\n"
        for name in [
            "CarpetX/src/io/io_adios2.hxx",
            "CarpetX/src/io/io.hxx",
            "CarpetX/src/util/typeprops.hh",
            "CarpetX/src/internal.hxx",
            "CarpetX/src/include/carpetx.hxx",
            "ODESolvers/src/solve.cxx",
            "TestCarpetXAPI/src/api_test.cxx",
        ]:
            with self.subTest(name=name):
                self.assertEqual(reported(name, line), [])
        self.assertEqual(reported("CarpetX/src/io/make.code.defn", "X = f();\n", Lang.MAKE), [])

    def test_line_numbers(self):
        text = "int a;\n\n" + self.PROTOTYPES[0] + "\n"
        self.assertEqual(findings("CarpetX/src/flesh/startup.cxx", text, Lang.CXX), [Finding(3, "prototype", self.PROTOTYPES[0])])


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

    def test_a_finding_merged_in_from_another_branch_is_not_reported(self):
        with tempfile.TemporaryDirectory() as root:
            git(root, "init", "-q", "-b", "dev")
            write(root, "T/interface.ccl", "IMPLEMENTS: T\n")
            write(root, "T/src/a.cxx", "int a;\n")
            git(root, "add", "-A")
            git(root, "commit", "-q", "-m", "base")
            base = git(root, "rev-parse", "HEAD").strip()
            git(root, "switch", "-q", "-c", "upstream")
            write(root, "T/src/a.cxx", '#include "../../CarpetX/src/driver.hxx"\nint a;\n')
            git(root, "commit", "-q", "-am", "upstream")
            git(root, "switch", "-q", "dev")
            write(root, "T/src/b.cxx", "int b;\n")
            git(root, "add", "-A")
            git(root, "commit", "-q", "-m", "work")
            git(root, "merge", "--no-ff", "-q", "-m", "m", "upstream")
            self.assertEqual(since_findings(root, base), [])
            write(root, "T/src/b.cxx", "#define CARPETX_INTERNAL\nint b;\n")
            self.assertEqual(since_findings(root, base), [("T/src/b.cxx", Finding(1, "marker", "#define CARPETX_INTERNAL"))])

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
            "no bare index int in CarpetX, no group-table call in io/, "
            "no hierarchy access in a checkpoint codec, "
            "every internal include at or below its own layer, "
            "no prototype in a CarpetX .cxx, "
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
        self.assertEqual(summary([i]), "✗ boundary: 1 bare index int in CarpetX")
        self.assertEqual(summary([e, w, w, i, i]), "✗ boundary: 1 firstvarindex addition outside GroupRecord::var, 2 active_levels_t uses, 2 bare index ints in CarpetX")
        t = ("CarpetX/src/schedule/evolve.cxx", Finding(9, "traversal", "active_levels = std::nullopt;"))
        self.assertEqual(summary([t]), "✗ boundary: 1 traversal use outside traversal.cxx")
        self.assertEqual(summary([w, t, t, i]), "✗ boundary: 1 active_levels_t use, 2 traversal uses outside traversal.cxx, 1 bare index int in CarpetX")
        r = ("CarpetX/src/io/io_tsv.cxx", Finding(6, "records", "for (int gi = 0; gi < CCTK_NumGroups(); ++gi) {}"))
        self.assertEqual(summary([r]), "✗ boundary: 1 group-table call in io/")
        self.assertEqual(summary([r, i, r]), "✗ boundary: 1 bare index int in CarpetX, 2 group-table calls in io/")
        k = ("CarpetX/src/io/io_openpmd_checkpoint.cxx", Finding(5, "codec", "for (const auto &patchdata : ghext->patches()) {"))
        self.assertEqual(summary([k]), "✗ boundary: 1 hierarchy use in a checkpoint codec")
        self.assertEqual(summary([i, k, k]), "✗ boundary: 1 bare index int in CarpetX, 2 hierarchy uses in a checkpoint codec")
        self.assertEqual(summary([k, r]), "✗ boundary: 1 group-table call in io/, 1 hierarchy use in a checkpoint codec")
        g = ("CarpetX/src/schedule/evolve.cxx", Finding(12, "door", "for (auto &p : ghext->patches())"))
        self.assertEqual(summary([g]), "✗ boundary: 1 hierarchy use the door does not allow")
        self.assertEqual(summary([k, g, g]), "✗ boundary: 1 hierarchy use in a checkpoint codec, 2 hierarchy uses the door does not allow")
        y = ("CarpetX/src/comm/sync_restrict.cxx", Finding(3, "layer", '#include "schedule/schedule.hxx"'))
        self.assertEqual(summary([y]), "✗ boundary: 1 include outside the layer order")
        self.assertEqual(summary([g, y, y, k]), "✗ boundary: 1 hierarchy use in a checkpoint codec, 2 includes outside the layer order, 1 hierarchy use the door does not allow")
        p = ("CarpetX/src/flesh/startup.cxx", Finding(36, "prototype", "int InitGH(cGH *cctkGH);"))
        self.assertEqual(summary([p]), "✗ boundary: 1 prototype in a CarpetX .cxx")
        self.assertEqual(summary([g, p, y, p]), "✗ boundary: 1 include outside the layer order, 2 prototypes in a CarpetX .cxx, 1 hierarchy use the door does not allow")

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
COUNTED_SUMMARY = "✗ boundary: 1 bare index int in CarpetX, 1 group-table call in io/"
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
    test.assertTrue(message["systemMessage"].startswith("lint check skipped: "), message)


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
        self.assertEqual(self.write_tool("CarpetX/schedule.ccl", "STORAGE: ghext\n"), "")
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


class Doc(unittest.TestCase):
    def test_the_map_is_what_doc_writes(self):
        with open(os.path.join(PROJECT, doc.MAP)) as f:
            text = f.read()
        self.assertEqual(doc.rewrite(text), text, "run python3 scripts/lint doc --write")

    def test_the_stack_is_the_layers_top_down_with_the_door_line(self):
        lines = doc.stack().splitlines()
        self.assertEqual(lines[:2], [doc.mark("the stack"), "flowchart TB"])
        self.assertIn("  " + " --> ".join(reversed(LAYERS)), lines)
        door = lines[lines.index('  subgraph door["driver() allowed"]') + 1 : lines.index("  end")]
        self.assertEqual([line.split("[")[0].strip() for line in door], list(reversed(HIERARCHY_LAYERS)))
        nodoor = lines[lines.index('  subgraph nodoor["no driver()"]') + 1 :]
        nodoor = nodoor[: nodoor.index("  end")]
        self.assertEqual([line.split("[")[0].strip() for line in nodoor], list(reversed(rules.UNDER_HIERARCHY)))

    def test_a_stack_node_names_the_rules_at_home_in_its_layer(self):
        self.assertEqual(doc.node("io", (), rules=True), '  io["io/ records, codec"]')
        self.assertEqual(doc.node("hierarchy", (), rules=True), '  hierarchy["hierarchy/ validity, varindex"]')
        self.assertEqual(doc.node("flesh", ("regrid", "flesh"), rules=True), '  flesh["flesh/ no header"]')
        self.assertEqual(doc.node("ops", ("regrid", "flesh"), rules=True), '  ops["ops/"]')
        self.assertEqual(doc.node("io", ()), '  io["io/"]')
        self.assertEqual(doc.rules_of("io"), ["records", "codec"])
        self.assertEqual(doc.rules_of("ops"), [])
        for rule in rules.RULES:
            with self.subTest(rule=rule.name):
                self.assertLessEqual(set(rule.home), set(LAYERS))
        self.assertEqual([rule.name for rule in rules.RULES if rule.home], ["validity", "directions", "varindex", "traversal", "records", "codec"])
        self.assertEqual(rules.layers_of(rules.VALIDITY_FILES), ("hierarchy",))
        self.assertEqual(rules.layers_of(rules.CODEC_FILES + rules.SYNC_FILES), ("io", "comm"))
        with self.assertRaises(AssertionError):
            rules.layers_of((rules.MAKE_DEFN,))

    def test_a_dashed_edge_is_declared_low_and_defined_high(self):
        for low, high, names in doc.LINK_TIME:
            self.assertLess(LAYERS.index(low), LAYERS.index(high))
            for name in names:
                with self.subTest(low=low, high=high, name=name):
                    self.assertTrue(doc.declares_and_defines(PROJECT, low, high, name))
                    self.assertIn(f'  {low} -. "', doc.stack())
        self.assertFalse(doc.declares_and_defines(PROJECT, "hierarchy", "io", "CactusAmrCore"))

    def test_the_includes_are_read_from_the_tree_above_the_base(self):
        with tempfile.TemporaryDirectory() as root:
            src = os.path.join(root, "CarpetX", "src")
            for layer in LAYERS:
                os.makedirs(os.path.join(src, layer))
            files = {
                "flesh/a.cxx": '#include "internal.hxx"\n#include "io/io.hxx"\n#include "comm/comm.hxx"\n#include "hierarchy/driver.hxx"\n#include <ops/interp.hxx>\n',
                "flesh/b.cxx": '#include "io/checkpoint.hxx"\n#include "io/io.hxx"\n#include "util/refuse.hxx"\n',
                "io/x.hxx": '#include "io/io.hxx"\n#include "ops/reduction.hxx"\n#include <mpi.h>\n',
                "comm/comm.hxx": "",
                "comm/notes.txt": '#include "io/io.hxx"\n',
                "hierarchy/driver.hxx": "class Core {\n  Core(int n);\n  void Est(const int lev, Tags &tags) const;\n};\nTags *tags_of(const Core &core);\n",
                "regrid/r.cxx": "Core::Core(int n) : Base(n), lev(0) {}\nvoid Core::Est(const int lev,\n    Tags &tags) const {\n  helper(lev);\n}\nTags *tags_of(const Core &core) {\n  return nullptr;\n}\n",
                "io/y.cxx": "void write(Core &core) {\n  Core &other = core;\n  other.Est(1, tags);\n  std::make_unique<Core>(1);\n}\n",
            }
            for name, text in files.items():
                with open(os.path.join(src, name), "w") as f:
                    f.write(text)
            edges = doc.includes(root)
            self.assertEqual(
                list(edges.items()),
                [
                    (("flesh", "io"), ("checkpoint.hxx", "io.hxx")),
                    (("flesh", "ops"), ("interp.hxx",)),
                    (("flesh", "comm"), ("comm.hxx",)),
                    (("io", "ops"), ("reduction.hxx",)),
                ],
            )
            graph = doc.include_graph(root).splitlines()
            self.assertEqual(graph[:2], [doc.mark("the includes"), "flowchart TB"])
            self.assertEqual(
                [line.split("[")[0].strip() for line in graph[2:9]],
                [layer for layer in reversed(LAYERS) if layer not in doc.BASE_LAYERS],
            )
            self.assertEqual(graph[9:], ['  flesh -- "checkpoint.hxx, io.hxx" --> io', '  flesh -- "interp.hxx" --> ops', '  flesh -- "comm.hxx" --> comm', '  io -- "reduction.hxx" --> ops'])
            self.assertEqual(doc.headerless(root), tuple(layer for layer in LAYERS if layer not in ("hierarchy", "io", "comm")))
            for drawn in (doc.stack(root), doc.include_graph(root)):
                self.assertIn('ops["ops/ no header"]', drawn)
                self.assertIn('comm["comm/', drawn)
                self.assertNotIn('comm["comm/ no header"]', drawn)
            self.assertIn('    io["io/ records, codec"]', doc.stack(root))
            for name in ("Core", "Est", "tags_of"):
                with self.subTest(name=name):
                    self.assertTrue(doc.declares_and_defines(root, "hierarchy", "regrid", name))
                    self.assertFalse(doc.declares_and_defines(root, "hierarchy", "io", name))
            self.assertFalse(doc.declares_and_defines(root, "io", "regrid", "Core"))
            self.assertFalse(doc.declares_and_defines(root, "regrid", "io", "Core"))

    def test_rewrite_replaces_only_the_marked_blocks(self):
        stale_stack = "```mermaid\n" + doc.mark("the stack") + "\nflowchart TB\n  stale\n```\n"
        stale_includes = "```mermaid\n" + doc.mark("the includes") + "\n  stale\n```\n"
        other = "```mermaid\nflowchart LR\n  a --> b\n```\n"
        text = "# map\n\n" + other + "\n" + stale_stack + "\nprose\n\n" + stale_includes + "\ntail\n"
        expected = "# map\n\n" + other + "\n```mermaid\n" + doc.stack() + "```\n\nprose\n\n```mermaid\n" + doc.include_graph() + "```\n\ntail\n"
        self.assertEqual(doc.rewrite(text), expected)
        self.assertEqual(doc.rewrite(expected), expected)
        for name, broken in [("no blocks", other), ("the stack twice", text + stale_stack), ("no includes", text.replace(stale_includes, ""))]:
            with self.subTest(name=name), self.assertRaises(ValueError):
                doc.rewrite(broken)

if __name__ == "__main__":
    unittest.main()
