from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from unittest import mock

from rules import RULE, Finding, findings, new_findings, render, since_findings, summary
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
        for name in ["CarpetX/src/schedule.hxx", "CarpetX/src/sync_restrict.cxx", "CarpetX/src/sync_restrict_internal.hxx"]:
            with self.subTest(name=name):
                self.assertEqual(
                    reported(name, self.FORWARDED),
                    [("directions", "const int *groups, const int *directions);")],
                )
        text = "  return SyncGroups(scheduled_window(sync_caller(gis).c_str()), cctkGH, gis, directions);\n"
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
            for name in ["CarpetX/src/valid.cxx", "CarpetX/src/io_silo.cxx", "CarpetX/src/driver.hxx", "CarpetX/src/include/carpetx.hxx"]:
                with self.subTest(name=name, line=line):
                    self.assertEqual(reported(name, "  " + line + "\n"), [("varindex", line)])

    def test_varindex_allowed_in_record_var(self):
        self.assertEqual(reported("CarpetX/src/driver.hxx", self.RECORD_VAR), [])
        self.assertEqual(
            reported("CarpetX/src/schedule.cxx", self.RECORD_VAR),
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
                self.assertEqual(reported("CarpetX/src/valid.cxx", text), [])

    def test_other_thorns_are_not_checked(self):
        self.assertEqual(reported("PDESolvers/src/solve.cxx", "  " + self.ADDITIONS[2] + "\n"), [])

    def test_only_c_and_cxx_files_are_checked(self):
        self.assertEqual(reported("CarpetX/src/make.code.defn", "X = firstvarindex + 1\n", Lang.MAKE), [])

    def test_line_numbers(self):
        text = "int a;\n" + self.ADDITIONS[0] + "\n"
        self.assertEqual(findings("CarpetX/src/schedule.cxx", text, Lang.CXX), [Finding(2, "varindex", self.ADDITIONS[0])])


class Window(unittest.TestCase):
    USES = [
        "struct active_levels_t;",
        "int SyncGroupsByDirI(active_levels_t window, const cGH *restrict cctkGH,",
        "active_levels_t scheduled_window(const char *caller,",
        "const traversal_scope_t scope(active_levels_t(0, level + 1));",
    ]

    def test_window_reported(self):
        for line in self.USES:
            for name in ["CarpetX/src/schedule.hxx", "CarpetX/src/sync_restrict.cxx", "CarpetX/src/include/carpetx.hxx"]:
                with self.subTest(name=name, line=line):
                    self.assertEqual(reported(name, "  " + line + "\n"), [("window", line)])

    def test_the_window_type_and_the_scheduler_state_are_fine(self):
        for name, text in [
            ("CarpetX/src/traversal.cxx", "std::optional<window_t> active_levels;\n"),
            ("CarpetX/src/traversal.cxx", "active_levels = window_t::levels(0, 1);\n"),
            ("CarpetX/src/schedule.cxx", "struct active_levels_type;\n"),
            ("CarpetX/src/schedule.cxx", "my_active_levels_t x;\n"),
        ]:
            with self.subTest(name=name, text=text):
                self.assertEqual(reported(name, text), [])

    def test_other_thorns_are_not_checked(self):
        self.assertEqual(reported("ODESolvers/src/solve.cxx", self.USES[0] + "\n"), [])

    def test_only_c_and_cxx_files_are_checked(self):
        self.assertEqual(reported("CarpetX/src/make.code.defn", "X = active_levels_t\n", Lang.MAKE), [])

    def test_line_numbers(self):
        text = "int a;\n" + self.USES[0] + "\n"
        self.assertEqual(findings("CarpetX/src/valid.hxx", text, Lang.CXX), [Finding(2, "window", self.USES[0])])


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
            for name in ["CarpetX/src/schedule.cxx", "CarpetX/src/api.cxx", "CarpetX/src/driver.hxx"]:
                with self.subTest(name=name, line=line):
                    self.assertEqual(reported(name, "  " + line + "\n"), [("traversal", line)])

    def test_the_traversal_unit_owns_the_slot(self):
        for line in self.USES:
            for name in ["CarpetX/src/traversal.cxx", "CarpetX/src/traversal.hxx"]:
                with self.subTest(name=name, line=line):
                    self.assertEqual(reported(name, line + "\n"), [])

    def test_another_thorn_and_the_public_headers_may_use_the_name(self):
        for line in self.USES:
            for name in ["ODESolvers/src/solve.cxx", "CarpetX/src/include/carpetx.hxx"]:
                with self.subTest(name=name, line=line):
                    self.assertEqual(reported(name, line + "\n"), [])

    def test_a_longer_name_is_not_the_slot(self):
        self.assertEqual(reported("CarpetX/src/schedule.cxx", "active_levels_t x;\n"), [("window", "active_levels_t x;")])
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
                self.assertEqual(reported("CarpetX/src/schedule.cxx", text), [])

    def test_only_c_and_cxx_files_are_checked(self):
        self.assertEqual(reported("CarpetX/src/make.code.defn", "X = active_levels\n", Lang.MAKE), [])

    def test_line_numbers(self):
        text = "int a;\n\n" + self.USES[0] + "\n"
        self.assertEqual(findings("CarpetX/src/schedule.cxx", text, Lang.CXX), [Finding(3, "traversal", self.USES[0])])


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
            for name in ["CarpetX/src/subcycling.hxx", "CarpetX/src/driver.hxx", "CarpetX/src/include/carpetx.hxx"]:
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
        self.assertEqual(reported("CarpetX/src/schedule.hxx", "int QueryGroupStorageB(const cGH *cctkGH, int gi, const char *groupname);\n"), [])
        text = (
            "  virtual void ErrorEst(int level, amrex::TagBoxArray &tags, amrex::Real time,\n"
            "                        int ngrow) override;\n"
            "  virtual void ClearLevel(int level) override;\n"
        )
        self.assertEqual(reported("CarpetX/src/driver.hxx", text), [])

    def test_index_allowed_only_where_the_signature_is_fixed(self):
        self.assertEqual(
            reported("CarpetX/src/driver.hxx", "int QueryGroupStorageB(const cGH *cctkGH, int gi, const char *groupname);\n"),
            [("index", "int QueryGroupStorageB(const cGH *cctkGH, int gi, const char *groupname);")],
        )
        self.assertEqual(
            reported("CarpetX/src/schedule.hxx", "window_t narrow(int min_level, int max_level);\n"),
            [("index", "window_t narrow(int min_level, int max_level);")],
        )
        self.assertEqual(
            reported("CarpetX/src/include/carpetx.hxx", "int num_levels(int patch);\n"),
            [("index", "int num_levels(int patch);")],
        )

    def test_index_only_in_carpetx_headers(self):
        for name in ["CarpetX/src/schedule.cxx", "ODESolvers/src/solve.hxx", "TestCarpetXAPI/src/api_test.cxx"]:
            with self.subTest(name=name):
                self.assertEqual(reported(name, self.BARE[0] + "\n"), [])

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
                self.assertEqual(reported("CarpetX/src/driver.hxx", text), [])

    def test_line_numbers(self):
        text = "int a;\n\n" + self.BARE[0] + "\n"
        self.assertEqual(findings("CarpetX/src/subcycling.hxx", text, Lang.CXX), [Finding(3, "index", self.BARE[0])])


class Codec(unittest.TestCase):
    CODECS = [
        "CarpetX/src/io_openpmd_checkpoint.cxx",
        "CarpetX/src/io_openpmd_names.hxx",
        "CarpetX/src/io_silo_checkpoint.cxx",
        "CarpetX/src/io_silo_names.hxx",
    ]
    HIERARCHY = [
        "for (const auto &patchdata : ghext->patches()) {",
        "const auto &leveldata = driver().level(level);",
        "const auto &leveldata = CarpetX::driver ().level(level);",
        "const GHExt::GroupRecord &group = record;",
        "for (const LevelData &leveldata : patchdata.levels()) {",
        "const GroupData &groupdata = leveldata.group(gi);",
        '#include "driver.hxx"',
        "#include <driver.hxx>",
    ]

    def test_hierarchy_access_is_reported_in_a_codec_file(self):
        for line in self.HIERARCHY:
            for name in self.CODECS:
                with self.subTest(name=name, line=line):
                    door = [("door", line)] if "ghext" in line else []
                    self.assertEqual(reported(name, "  " + line + "\n"), [("codec", line)] + door)

    def test_the_record_and_names_header_are_allowed(self):
        text = (
            '#include "checkpoint.hxx"\n'
            '#include "io_openpmd_names.hxx"\n'
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
            '#include "driver_api.hxx"\n'
        )
        self.assertEqual(reported(self.CODECS[0], text), [])

    def test_the_slice_backend_is_out_of_scope(self):
        line = "for (const auto &patchdata : driver().patches()) {\n"
        for name in ["CarpetX/src/io_openpmd.cxx", "CarpetX/src/io_silo.cxx", "CarpetX/src/checkpoint.cxx", "ODESolvers/src/io_openpmd_checkpoint.cxx"]:
            with self.subTest(name=name):
                self.assertEqual(reported(name, line), [])

    def test_only_c_and_cxx_files_are_checked(self):
        self.assertEqual(reported(self.CODECS[0], "X = ghext->patches()\n", Lang.MAKE), [])

    def test_line_numbers(self):
        text = '#include "checkpoint.hxx"\n\n' + self.HIERARCHY[0] + "\n"
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
            for name in ["CarpetX/src/schedule.cxx", "CarpetX/src/io.cxx", "CarpetX/src/driver.hxx"]:
                with self.subTest(name=name, line=line):
                    self.assertEqual(reported(name, "  " + line + "\n"), [("door", line)])

    def test_driver_cxx_owns_the_pointer(self):
        for line in self.USES:
            with self.subTest(line=line):
                self.assertEqual(reported("CarpetX/src/driver.cxx", line + "\n"), [])

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
            for name in ["CarpetX/src/schedule.cxx", "CarpetX/src/io.cxx", "CarpetX/src/driver.hxx"]:
                with self.subTest(name=name, text=text):
                    self.assertEqual(reported(name, text), [])

    def test_a_codec_naming_it_is_reported_under_both_rules(self):
        line = "for (const auto &patchdata : ghext->patches()) {"
        self.assertEqual(reported("CarpetX/src/io_silo_checkpoint.cxx", line + "\n"), [("codec", line), ("door", line)])

    def test_only_c_and_cxx_files_are_checked(self):
        self.assertEqual(reported("CarpetX/src/make.code.defn", "X = ghext\n", Lang.MAKE), [])

    def test_line_numbers(self):
        text = "int a;\n\n" + self.USES[1] + "\n"
        self.assertEqual(findings("CarpetX/src/schedule.cxx", text, Lang.CXX), [Finding(3, "door", self.USES[1])])


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
            "no validity call outside valid.cxx, no directions in the sync layer, "
            "no firstvarindex addition outside GroupRecord::var, no active_levels_t, "
            "no active_levels outside traversal.cxx, "
            "no bare index int in a CarpetX header, no hierarchy access in a checkpoint codec, "
            "and the hierarchy has one door",
        )

    def test_counts(self):
        a = ("T/src/a.cxx", Finding(3, "include", '#include "../../CarpetX/src/driver.hxx"'))
        b = ("T/src/b.cxx", Finding(1, "marker", "#define CARPETX_INTERNAL"))
        c = ("CarpetX/src/driver.cxx", Finding(7, "validity", "why.set_all(valid_t(), r);"))
        d = ("CarpetX/src/schedule.hxx", Finding(2, "directions", "const int *groups, const int *directions);"))
        e = ("CarpetX/src/valid.cxx", Finding(5, "varindex", "CCTK_FullVarName(groupdata.record.firstvarindex + vi),"))
        self.assertEqual(summary([a]), "✗ boundary: 1 private include")
        self.assertEqual(summary([a, a, b]), "✗ boundary: 2 private includes, 1 stray marker")
        self.assertEqual(summary([c]), "✗ boundary: 1 validity call outside valid.cxx")
        self.assertEqual(summary([b, c, c]), "✗ boundary: 1 stray marker, 2 validity calls outside valid.cxx")
        self.assertEqual(summary([d]), "✗ boundary: 1 directions parameter in the sync layer")
        self.assertEqual(summary([c, d, d]), "✗ boundary: 1 validity call outside valid.cxx, 2 directions parameters in the sync layer")
        self.assertEqual(summary([e]), "✗ boundary: 1 firstvarindex addition outside GroupRecord::var")
        self.assertEqual(summary([d, e, e]), "✗ boundary: 1 directions parameter in the sync layer, 2 firstvarindex additions outside GroupRecord::var")
        w = ("CarpetX/src/schedule.hxx", Finding(3, "window", "struct active_levels_t;"))
        i = ("CarpetX/src/subcycling.hxx", Finding(4, "index", "void EnsureRKBuffers(int patch, int level, gi_t gi);"))
        self.assertEqual(summary([w]), "✗ boundary: 1 active_levels_t use")
        self.assertEqual(summary([i]), "✗ boundary: 1 bare index int in a CarpetX header")
        self.assertEqual(summary([e, w, w, i, i]), "✗ boundary: 1 firstvarindex addition outside GroupRecord::var, 2 active_levels_t uses, 2 bare index ints in a CarpetX header")
        t = ("CarpetX/src/schedule.cxx", Finding(9, "traversal", "active_levels = std::nullopt;"))
        self.assertEqual(summary([t]), "✗ boundary: 1 traversal use outside traversal.cxx")
        self.assertEqual(summary([w, t, t, i]), "✗ boundary: 1 active_levels_t use, 2 traversal uses outside traversal.cxx, 1 bare index int in a CarpetX header")
        k = ("CarpetX/src/io_openpmd_checkpoint.cxx", Finding(5, "codec", "for (const auto &patchdata : ghext->patches()) {"))
        self.assertEqual(summary([k]), "✗ boundary: 1 hierarchy use in a checkpoint codec")
        self.assertEqual(summary([i, k, k]), "✗ boundary: 1 bare index int in a CarpetX header, 2 hierarchy uses in a checkpoint codec")
        g = ("CarpetX/src/schedule.cxx", Finding(12, "door", "for (auto &p : ghext->patches())"))
        self.assertEqual(summary([g]), "✗ boundary: 1 raw hierarchy name outside driver.cxx")
        self.assertEqual(summary([k, g, g]), "✗ boundary: 1 hierarchy use in a checkpoint codec, 2 raw hierarchy names outside driver.cxx")

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


SOURCE = "namespace CarpetX {\nvoid Evolve() {\n}\n} // namespace CarpetX\n"
PLANT = ("void Evolve() {\n", "void Evolve() {\n  for (auto &p : ghext->patches())\n")
PLANTED = SOURCE.replace(*PLANT)
DENIAL = "CarpetX/src/schedule.cxx:3   door       for (auto &p : ghext->patches())"
DOOR_SUMMARY = "✗ boundary: 1 raw hierarchy name outside driver.cxx"
PROJECT_FILES = {
    "CarpetX/interface.ccl": "IMPLEMENTS: Driver\n",
    "CarpetX/src/schedule.cxx": SOURCE,
    "CarpetX/src/driver.cxx": SOURCE,
    "CarpetX/README": "not covered\n",
}


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
        lines = denial(self, self.edit("CarpetX/src/schedule.cxx", *PLANT))
        self.assertEqual(lines[0], DENIAL)
        self.assertEqual(lines[1:-1], RULE.split("\n"))
        self.assertEqual(lines[-1], DOOR_SUMMARY)

    def test_the_door_file_may_name_it(self):
        self.assertEqual(self.edit("CarpetX/src/driver.cxx", *PLANT), "")

    def test_only_what_the_edit_adds_is_new(self):
        write(self.root, "CarpetX/src/schedule.cxx", PLANTED)
        self.assertEqual(self.edit("CarpetX/src/schedule.cxx", "void Evolve()", "void Initialise()"), "")
        lines = denial(self, self.edit("CarpetX/src/schedule.cxx", "}\n} //", "  assert(!ghext);\n}\n} //"))
        self.assertEqual(lines[0], "CarpetX/src/schedule.cxx:4   door       assert(!ghext);")

    def test_a_write_and_a_new_file_are_judged_whole(self):
        self.assertEqual(denial(self, self.write_tool("CarpetX/src/schedule.cxx", PLANTED))[0], DENIAL)
        self.assertEqual(denial(self, self.edit("CarpetX/src/new.cxx", "", PLANTED))[0], DENIAL.replace("schedule", "new"))
        self.assertEqual(self.write_tool("CarpetX/src/new.cxx", SOURCE), "")

    def test_an_edit_that_does_not_apply_is_left_to_the_edit_tool(self):
        self.assertEqual(self.edit("CarpetX/src/schedule.cxx", "int nope;\n", "ghext->patches();\n"), "")

    def test_files_the_check_does_not_read_pass(self):
        self.assertEqual(self.edit("CarpetX/README", "not covered\n", "ghext->patches();\n"), "")
        self.assertEqual(self.write_tool("CarpetX/schedule.ccl", "# ghext\n"), "")
        self.assertEqual(self.write_tool("Other/src/a.cxx", PLANTED), "")
        with tempfile.TemporaryDirectory() as other:
            self.assertEqual(self.run_edit(edit_payload(other, "CarpetX/src/a.cxx", "", PLANTED)), "")

    def test_a_relative_path_is_taken_from_cwd(self):
        payload = edit_payload(self.root, "CarpetX/src/schedule.cxx", *PLANT)
        payload["tool_input"]["file_path"] = "CarpetX/src/schedule.cxx"
        self.assertEqual(denial(self, self.run_edit(payload))[0], DENIAL)

    def test_a_bad_payload_is_a_system_message(self):
        for payload in [
            "not json",
            "[]",
            {"tool_name": "Edit", "tool_input": {"file_path": os.path.join(self.root, "CarpetX/src/schedule.cxx"), "new_string": "x"}},
            {"tool_name": "Teleport", "tool_input": {"file_path": os.path.join(self.root, "CarpetX/src/schedule.cxx")}},
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
            write(self.root, "CarpetX/src/schedule.cxx", PLANTED)
            git(self.root, "commit", "-q", "-am", "plant")
            decision = json.loads(self.run_stop({"stop_hook_active": False}))
            self.assertEqual(sorted(decision), ["decision", "reason"])
            self.assertEqual(decision["decision"], "block")
            lines = decision["reason"].split("\n")
            self.assertEqual((lines[0], lines[-1]), (DENIAL, DOOR_SUMMARY))
            self.assertEqual(self.run_stop({"stop_hook_active": True}), "")
            write(self.root, "CarpetX/src/schedule.cxx", SOURCE)
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
            ("PreToolUse", edit_payload(self.root, "CarpetX/src/schedule.cxx", *PLANT)),
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
        result = self.run_command(command, self.root, edit_payload(self.root, "CarpetX/src/schedule.cxx", *PLANT))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(denial(self, result.stdout)[0], DENIAL)
        result = self.run_command(command, self.root, edit_payload(self.root, "CarpetX/src/driver.cxx", *PLANT))
        self.assertEqual((result.returncode, result.stdout), (0, ""), result.stderr)

    def test_the_stop_command_runs_the_project_s_tool(self):
        [(_, command)] = configured("Stop")
        copy_tools(self.root)
        with mock.patch.dict(os.environ, ISOLATED_GIT):
            git(self.root, "init", "-q", "-b", "dev")
            git(self.root, "add", "-A")
            git(self.root, "commit", "-q", "-m", "base")
            write(self.root, "CarpetX/src/schedule.cxx", PLANTED)
            result = self.run_command(command, self.root, {"stop_hook_active": False})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["decision"], "block")


if __name__ == "__main__":
    unittest.main()
