from __future__ import annotations

import os
import tempfile
import unittest
from unittest import mock

from boundary import rules
from boundary.symbols import (
    Reach,
    Stray,
    SymbolsError,
    declared_name,
    layer_objects,
    names,
    parse_definitions,
    parse_imports,
    parse_mach_o_weak,
    reaches,
    render,
    strays,
    strong_definitions,
    summary,
    weakened,
)

SHUTDOWN = "__ZN7CarpetX10ShutdownIOEv"
DRIVER = "__ZN7CarpetX6driverEv"
TIMER = "__ZN7CarpetX5TimerD2Ev"
PARENT = "__ZNK7CarpetX7level_t6parentENSt3__115source_locationE"
LINUX_DEFINED = (
    "0000000000003428 T _ZN7CarpetX10ShutdownIOEv\n"
    "0000000000000000 W _ZN7CarpetX5TimerD2Ev\n"
    "0000000000000010 t _ZN7CarpetXL6helperEv\n"
    "0000000000000000 D _ZN7CarpetX11ghext_countE\n"
    "0000000000000000 T CarpetX_Startup\n"
    "                 U _ZN7CarpetX6driverEv\n"
)
SIZES = "__ZN7CarpetX24per_level_max_grid_sizesEv"
DEFAULTED = "__ZN7CarpetX15size_or_defaultEii"
CURRENT_MODE = "_ZN7CarpetX12current_modeEPK4_cGH"
GLOBAL_MODE = "_ZN7CarpetX14in_global_modeEPK4_cGH"
ALL_LEVELS = "_ZN7CarpetX10all_levelsEv"
PRINT_SYMMETRY = "_ZN7CarpetXlsERSoNS_10symmetry_tE"
TO_STRING = "__ZNK7CarpetX7valid_tcvNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEEv"
REALLOCATE = "__ZNSt5dequeIN7CarpetX13write_scope_tESaIS1_EE17_M_reallocate_mapEmb"
MACOS_WEAK = (
    "0000000000002f4c (__TEXT,__text) weak external __ZN7CarpetX5TimerD2Ev\n"
    "0000000000000000 (__TEXT,__text) external __ZN7CarpetX24per_level_max_grid_sizesEv\n"
    "000000000000034c (__TEXT,__text) non-external __ZN7CarpetXL6helperEv\n"
    "00000000000033a0 (__TEXT,__text) weak private external __ZNK7CarpetX5GHExt5levelENS_7level_tE\n"
    "0000000000004000 (__TEXT,__text) weak external _CCTK_Weak\n"
)
MACOS_UNDEFINED = "__ZN7CarpetX6driverEv\n_CCTK_VInfo\n__ZN7CarpetX10ShutdownIOEv\n"
LINUX_UNDEFINED = "                 U _ZN7CarpetX6driverEv\n                 U _CCTK_VInfo\n"


class Parsing(unittest.TestCase):
    def test_definitions_keep_the_kind_and_only_carpetx_names(self):
        self.assertEqual(
            parse_definitions(LINUX_DEFINED),
            {
                ("T", "_ZN7CarpetX10ShutdownIOEv"),
                ("W", "_ZN7CarpetX5TimerD2Ev"),
                ("t", "_ZN7CarpetXL6helperEv"),
                ("D", "_ZN7CarpetX11ghext_countE"),
                ("U", "_ZN7CarpetX6driverEv"),
            },
        )

    def test_imports_in_both_nm_formats(self):
        self.assertEqual(parse_imports(MACOS_UNDEFINED), {"__ZN7CarpetX6driverEv", "__ZN7CarpetX10ShutdownIOEv"})
        self.assertEqual(parse_imports(LINUX_UNDEFINED), {"_ZN7CarpetX6driverEv"})
        self.assertEqual(parse_imports(""), set())


class Definitions(unittest.TestCase):
    LAYERS = {"io/io.cxx.o": "io", "flesh/startup.cxx.o": "flesh", "hierarchy/model.cxx.o": "hierarchy"}

    def test_a_strong_symbol_defined_once_has_a_home(self):
        definitions = {"io/io.cxx.o": {("T", SHUTDOWN)}, "hierarchy/model.cxx.o": {("T", DRIVER)}}
        self.assertEqual(strong_definitions(definitions, self.LAYERS), {SHUTDOWN: "io", DRIVER: "hierarchy"})

    def test_a_weak_or_local_or_undefined_symbol_has_none(self):
        definitions = {"io/io.cxx.o": {("W", TIMER), ("t", "_ZN7CarpetXL6helperEv"), ("U", DRIVER), ("V", PARENT)}}
        self.assertEqual(strong_definitions(definitions, self.LAYERS), {})

    def test_a_symbol_defined_in_two_objects_has_none(self):
        definitions = {"io/io.cxx.o": {("T", TIMER)}, "flesh/startup.cxx.o": {("T", TIMER), ("T", SHUTDOWN)}}
        self.assertEqual(strong_definitions(definitions, self.LAYERS), {SHUTDOWN: "flesh"})


class Reaches(unittest.TestCase):
    LAYERS = {"flesh/startup.cxx.o": "flesh", "io/io.cxx.o": "io", "hierarchy/model.cxx.o": "hierarchy"}
    DEFINED = {SHUTDOWN: "io", DRIVER: "hierarchy", PARENT: "api"}

    def test_an_import_from_a_lower_layer_or_the_same_one_is_not_a_reach(self):
        imports = {"flesh/startup.cxx.o": {SHUTDOWN, DRIVER}, "io/io.cxx.o": {DRIVER, SHUTDOWN}}
        self.assertEqual(reaches(self.LAYERS, self.DEFINED, imports, {}), [])

    def test_an_import_from_a_higher_layer_is(self):
        imports = {"hierarchy/model.cxx.o": {SHUTDOWN, PARENT}, "io/io.cxx.o": {PARENT}}
        self.assertEqual(
            reaches(self.LAYERS, self.DEFINED, imports, {}),
            [
                Reach("hierarchy/model.cxx.o", "hierarchy", "io", SHUTDOWN),
                Reach("hierarchy/model.cxx.o", "hierarchy", "api", PARENT),
                Reach("io/io.cxx.o", "io", "api", PARENT),
            ],
        )

    def test_an_import_from_a_lower_layer_the_object_does_not_read_is(self):
        layers = {"ops/interpolate.cxx.o": "ops", "api/api.cxx.o": "api", "regrid/regrid.cxx.o": "regrid"}
        defined = {SHUTDOWN: "io", DRIVER: "hierarchy", PARENT: "comm", TIMER: "regrid"}
        imports = {"ops/interpolate.cxx.o": {PARENT, DRIVER}, "api/api.cxx.o": {SHUTDOWN, TIMER, PARENT}, "regrid/regrid.cxx.o": {PARENT, SHUTDOWN}}
        self.assertEqual(
            reaches(layers, defined, imports, {"schedule": ("hierarchy",), "io": ("ops",)}),
            [
                Reach("api/api.cxx.o", "api", "io", SHUTDOWN),
                Reach("api/api.cxx.o", "api", "regrid", TIMER),
                Reach("ops/interpolate.cxx.o", "ops", "comm", PARENT),
                Reach("regrid/regrid.cxx.o", "regrid", "io", SHUTDOWN),
            ],
        )

    def test_an_import_from_a_layer_a_header_carries_in_is_not_a_reach(self):
        layers = {"regrid/regrid.cxx.o": "regrid", "api/api.cxx.o": "api"}
        defined = {TIMER: "util", SHUTDOWN: "io"}
        imports = {"regrid/regrid.cxx.o": {TIMER}, "api/api.cxx.o": {SHUTDOWN}}
        unread = [Reach("api/api.cxx.o", "api", "io", SHUTDOWN), Reach("regrid/regrid.cxx.o", "regrid", "util", TIMER)]
        self.assertEqual(reaches(layers, defined, imports, {}), unread)
        self.assertEqual(reaches(layers, defined, imports, {"hierarchy": ("util",)}), unread[:1])
        self.assertEqual(reaches(layers, defined, imports, {"hierarchy": ("util",), "schedule": ("io",)}), [])

    def test_an_import_nothing_defines_strongly_is_ignored(self):
        imports = {"hierarchy/model.cxx.o": {TIMER, "_CCTK_VInfo"}}
        self.assertEqual(reaches(self.LAYERS, self.DEFINED, imports, {}), [])


class Objects(unittest.TestCase):
    def test_objects_under_a_layer_directory_and_nothing_else(self):
        with tempfile.TemporaryDirectory() as objdir:
            for rel in ["io/io.cxx.o", "io/io_meta.cxx.o", "flesh/startup.cxx.o", "driver.cxx.o", "io/notes.txt", "extra/x.cxx.o"]:
                os.makedirs(os.path.dirname(os.path.join(objdir, rel)) or objdir, exist_ok=True)
                open(os.path.join(objdir, rel), "w").close()
            self.assertEqual(
                layer_objects(objdir, keep=lambda obj: True),
                {"io/io.cxx.o": "io", "io/io_meta.cxx.o": "io", "flesh/startup.cxx.o": "flesh"},
            )
            self.assertEqual(layer_objects(objdir, keep=lambda obj: obj.startswith("io/")), {"io/io.cxx.o": "io", "io/io_meta.cxx.o": "io"})

    def test_an_object_whose_source_left_the_tree_is_skipped(self):
        from boundary.symbols import in_tree

        self.assertTrue(in_tree("io/io.cxx.o"))
        self.assertFalse(in_tree("io/gone.cxx.o"))

    def test_no_objects_is_an_error(self):
        with tempfile.TemporaryDirectory() as objdir:
            os.makedirs(os.path.join(objdir, "io"))
            with self.assertRaises(SymbolsError):
                layer_objects(objdir, keep=lambda obj: True)


class Weak(unittest.TestCase):
    def test_a_mach_o_weak_definition_is_read_from_the_darwin_format(self):
        self.assertEqual(parse_mach_o_weak(MACOS_WEAK), {"__ZN7CarpetX5TimerD2Ev", "__ZNK7CarpetX5GHExt5levelENS_7level_tE"})
        self.assertEqual(parse_mach_o_weak(LINUX_DEFINED), set())

    def test_a_weak_definition_is_not_strong_whatever_letter_nm_gave_it(self):
        found = {("T", "__ZN7CarpetX5TimerD2Ev"), ("T", SIZES), ("t", "__ZN7CarpetXL6helperEv")}
        self.assertEqual(weakened(found, {"__ZN7CarpetX5TimerD2Ev"}), {("W", "__ZN7CarpetX5TimerD2Ev"), ("T", SIZES), ("t", "__ZN7CarpetXL6helperEv")})
        self.assertEqual(weakened(found, set()), found)
        layers = {"config/grid_sizes.cxx.o": "config"}
        self.assertEqual(strong_definitions({"config/grid_sizes.cxx.o": weakened(found, {"__ZN7CarpetX5TimerD2Ev"})}, layers), {SIZES: "config"})


class DeclaredName(unittest.TestCase):
    def test_a_function_or_object_of_the_namespace_is_its_own_name(self):
        for symbol, name in [
            (SIZES, "per_level_max_grid_sizes"),
            (SIZES[1:], "per_level_max_grid_sizes"),
            (CURRENT_MODE, "current_mode"),
            ("_ZN7CarpetX11ghext_countE", "ghext_count"),
            ("_ZN7CarpetXL6helperEv", "helper"),
            ("_ZN7CarpetX13components_ofIiEEvT_", "components_of"),
            ("_ZGVZN7CarpetX6driverEvE4once", "driver"),
        ]:
            with self.subTest(symbol=symbol):
                self.assertEqual(declared_name(symbol), name)

    def test_a_member_is_its_outermost_class(self):
        for symbol, name in [
            ("_ZN7CarpetX5GHExt9PatchDataC1Ev", "GHExt"),
            ("_ZNK7CarpetX5GHExt6recordENS_4gi_tE", "GHExt"),
            ("_ZNR7CarpetX13write_scope_t4mfabENS_7level_tE", "write_scope_t"),
            (TO_STRING, "valid_t"),
            ("_ZTVN7CarpetX13CactusAmrCoreE", "CactusAmrCore"),
            ("_ZTIN7CarpetX13CactusAmrCoreE", "CactusAmrCore"),
            ("_ZThn16_N7CarpetX13CactusAmrCore8ErrorEstEi", "CactusAmrCore"),
        ]:
            with self.subTest(symbol=symbol):
                self.assertEqual(declared_name(symbol), name)

    def test_an_operator_of_the_namespace_is_spelled_as_a_header_spells_it(self):
        self.assertEqual(declared_name(PRINT_SYMMETRY), "operator<<")
        self.assertEqual(declared_name("_ZN7CarpetXeqERKNS_7valid_tES2_"), "operator==")
        self.assertEqual(declared_name("_ZN7CarpetXclEv"), "operator()")
        self.assertEqual(declared_name("_ZN7CarpetXliEy"), "_ZN7CarpetXliEy")
        self.assertTrue(names("std::ostream & operator<<( std::ostream & os , boundary_t b ) ;", "operator<<"))
        self.assertTrue(names("std::ostream &operator << (std::ostream &os, boundary_t b);", "operator<<"))
        self.assertFalse(names("level_t &operator<<=(level_t &level, int by);", "operator<<"))
        self.assertFalse(names("bool operator<(level_t a, level_t b);", "operator<<"))
        self.assertTrue(names("bool operator<(level_t a, level_t b);", "operator<"))
        self.assertFalse(names("bool operator<=(level_t a, level_t b);", "operator<"))
        self.assertTrue(names("int operator()(int x);", "operator()"))

    def test_a_symbol_outside_the_namespace_has_no_name(self):
        for symbol in [REALLOCATE, "_CCTK_VInfo", "CarpetX_Startup", "__ZNSt6vectorIN7CarpetX7valid_tESaIS1_EED1Ev", "_ZN5amrex7CarpetXEv"]:
            with self.subTest(symbol=symbol):
                self.assertIsNone(declared_name(symbol))

    def test_a_name_is_a_whole_word_of_the_text(self):
        self.assertTrue(names("amrex::Vector<amrex::IntVect> per_level_max_grid_sizes();", "per_level_max_grid_sizes"))
        self.assertFalse(names("int all_per_level_max_grid_sizes();", "per_level_max_grid_sizes"))
        self.assertFalse(names("int per_level_max_grid_sizes_x;", "per_level_max_grid_sizes"))


class Strays(unittest.TestCase):
    OBJECT = "config/grid_sizes.cxx.o"
    HEADER = "CarpetX/src/config/grid_sizes.hxx"
    LAYERS = {OBJECT: "config", "hierarchy/modes.cxx.o": "hierarchy", "hierarchy/handles.cxx.o": "hierarchy", "schedule/evolve.cxx.o": "schedule"}
    HEADERS = {HEADER: "namespace CarpetX { amrex::Vector<amrex::IntVect> per_level_max_grid_sizes(); }"}
    PUBLIC = "namespace CarpetX { window_t all_levels(); std::ostream &operator<<(std::ostream &os, symmetry_t symmetry); struct valid_t {}; }"

    def test_a_strong_definition_the_header_of_its_stem_names_is_not_a_stray(self):
        self.assertEqual(strays(self.LAYERS, {self.OBJECT: {("T", SIZES)}}, self.HEADERS, {}), [])

    def test_a_strong_definition_the_header_of_its_stem_does_not_name_is(self):
        definitions = {self.OBJECT: {("T", SIZES), ("T", DEFAULTED), ("D", "_ZN7CarpetX11ghext_countE")}}
        self.assertEqual(
            strays(self.LAYERS, definitions, self.HEADERS, {}),
            [Stray(self.OBJECT, "_ZN7CarpetX11ghext_countE", self.HEADER), Stray(self.OBJECT, DEFAULTED, self.HEADER)],
        )

    def test_a_weak_or_local_definition_or_one_outside_the_namespace_is_not_judged(self):
        definitions = {self.OBJECT: {("W", DEFAULTED), ("t", DEFAULTED), ("V", "_ZTVN7CarpetX13CactusAmrCoreE"), ("U", CURRENT_MODE), ("T", REALLOCATE), ("T", "CarpetX_Startup")}}
        self.assertEqual(strays(self.LAYERS, definitions, self.HEADERS, {}), [])

    def test_a_source_with_no_header_of_its_stem_defines_nothing_strong(self):
        layers = {"config/readers.cxx.o": "config"}
        self.assertEqual(strays(layers, {"config/readers.cxx.o": {("T", SIZES)}}, self.HEADERS, {}), [Stray("config/readers.cxx.o", SIZES, "CarpetX/src/config/readers.hxx")])

    def test_only_an_object_of_a_listed_layer_is_judged(self):
        definitions = {"schedule/evolve.cxx.o": {("T", DEFAULTED)}, "hierarchy/modes.cxx.o": {("T", CURRENT_MODE)}}
        self.assertEqual(strays(self.LAYERS, definitions, self.HEADERS, {}), [])
        self.assertNotIn("hierarchy", rules.STEMS)

    def test_a_listed_layer_holds_each_source_to_its_own_header_and_a_public_body_to_the_public_headers(self):
        definitions = {
            "hierarchy/modes.cxx.o": {("T", CURRENT_MODE), ("T", GLOBAL_MODE), ("T", ALL_LEVELS)},
            "hierarchy/handles.cxx.o": {("T", ALL_LEVELS), ("T", PRINT_SYMMETRY), ("T", TO_STRING), ("T", GLOBAL_MODE)},
        }
        headers = {**self.HEADERS, "CarpetX/src/hierarchy/modes.hxx": "bool in_global_mode ( const cGH * cctkGH ) ;"}
        public = {"hierarchy/handles.cxx.o": self.PUBLIC}
        with mock.patch.dict(rules.STEMS, {"hierarchy": {}}):
            self.assertEqual(
                strays(self.LAYERS, definitions, headers, public),
                [
                    Stray("hierarchy/handles.cxx.o", GLOBAL_MODE, "CarpetX/src/hierarchy/handles.hxx"),
                    Stray("hierarchy/modes.cxx.o", ALL_LEVELS, "CarpetX/src/hierarchy/modes.hxx"),
                    Stray("hierarchy/modes.cxx.o", CURRENT_MODE, "CarpetX/src/hierarchy/modes.hxx"),
                ],
            )
            self.assertEqual(len(strays(self.LAYERS, definitions, headers, {})), 6)


class Report(unittest.TestCase):
    UP = Reach("config/groups.cxx.o", "config", "comm", "_ZN7CarpetX16get_interpolatorEv")
    OUT = Stray("config/grid_sizes.cxx.o", DEFAULTED, "CarpetX/src/config/grid_sizes.hxx")
    READABLE = {UP.symbol: "CarpetX::get_interpolator(std::string)", OUT.symbol: "CarpetX::size_or_default(int, int)"}
    CLEAN = "✓ symbols: every CarpetX import from the importing object's layer or one it reads"

    def test_clean(self):
        self.assertEqual(render([], [], {}), self.CLEAN + ", and every definition of config/ named in the header of its stem")
        with mock.patch.dict(rules.STEMS, {"hierarchy": {}}):
            self.assertEqual(summary([], []), self.CLEAN + ", and every definition of config/ and hierarchy/ named in the header of its stem")
        with mock.patch.dict(rules.STEMS, clear=True):
            self.assertEqual(summary([], []), self.CLEAN)

    def test_unread(self):
        lines = render([self.UP], [], self.READABLE).split("\n")
        self.assertEqual(lines[0], "config/groups.cxx.o   -> comm        CarpetX::get_interpolator(std::string)")
        self.assertTrue(lines[1].startswith("symbols: an object the build made"))
        self.assertFalse(any(line.startswith("stems: ") for line in lines))
        self.assertEqual(lines[-1], "✗ symbols: 1 import from a layer the importing object's layer does not read")
        self.assertEqual(summary([self.UP, self.UP], []), "✗ symbols: 2 imports from a layer the importing object's layer does not read")

    def test_stray(self):
        lines = render([], [self.OUT], self.READABLE).split("\n")
        self.assertEqual(lines[0], "config/grid_sizes.cxx.o   not in config/grid_sizes.hxx   CarpetX::size_or_default(int, int)")
        self.assertTrue(lines[1].startswith("stems: in a layer that STEMS in scripts/lint/boundary/rules.py lists"))
        self.assertFalse(any(line.startswith("symbols: ") for line in lines))
        self.assertEqual(lines[-1], "✗ symbols: 1 definition no header of the same stem names")
        both = render([self.UP], [self.OUT, self.OUT], self.READABLE).split("\n")
        self.assertEqual(both[-1], "✗ symbols: 1 import from a layer the importing object's layer does not read, 2 definitions no header of the same stem names")
        self.assertLess(both.index(lines[0]), both.index(lines[1]))
        self.assertLess(both.index(next(line for line in both if line.startswith("symbols: "))), both.index(lines[0]))
