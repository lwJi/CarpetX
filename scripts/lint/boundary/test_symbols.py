from __future__ import annotations

import os
import tempfile
import unittest

from boundary.symbols import (
    ACCEPTED,
    Reach,
    SymbolsError,
    accepted,
    layer_objects,
    parse_definitions,
    parse_imports,
    reaches,
    render,
    split,
    strong_definitions,
    summary,
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
    LAYERS = {"io/io.cxx.o": "io", "flesh/startup.cxx.o": "flesh", "hierarchy/hierarchy.cxx.o": "hierarchy"}

    def test_a_strong_symbol_defined_once_has_a_home(self):
        definitions = {"io/io.cxx.o": {("T", SHUTDOWN)}, "hierarchy/hierarchy.cxx.o": {("T", DRIVER)}}
        self.assertEqual(strong_definitions(definitions, self.LAYERS), {SHUTDOWN: "io", DRIVER: "hierarchy"})

    def test_a_weak_or_local_or_undefined_symbol_has_none(self):
        definitions = {"io/io.cxx.o": {("W", TIMER), ("t", "_ZN7CarpetXL6helperEv"), ("U", DRIVER), ("V", PARENT)}}
        self.assertEqual(strong_definitions(definitions, self.LAYERS), {})

    def test_a_symbol_defined_in_two_objects_has_none(self):
        definitions = {"io/io.cxx.o": {("T", TIMER)}, "flesh/startup.cxx.o": {("T", TIMER), ("T", SHUTDOWN)}}
        self.assertEqual(strong_definitions(definitions, self.LAYERS), {SHUTDOWN: "flesh"})


class Reaches(unittest.TestCase):
    LAYERS = {"flesh/startup.cxx.o": "flesh", "io/io.cxx.o": "io", "hierarchy/hierarchy.cxx.o": "hierarchy"}
    DEFINED = {SHUTDOWN: "io", DRIVER: "hierarchy", PARENT: "api"}

    def test_an_import_from_a_lower_layer_or_the_same_one_is_not_a_reach(self):
        imports = {"flesh/startup.cxx.o": {SHUTDOWN, DRIVER}, "io/io.cxx.o": {DRIVER, SHUTDOWN}}
        self.assertEqual(reaches(self.LAYERS, self.DEFINED, imports), [])

    def test_an_import_from_a_higher_layer_is(self):
        imports = {"hierarchy/hierarchy.cxx.o": {SHUTDOWN, PARENT}, "io/io.cxx.o": {PARENT}}
        self.assertEqual(
            reaches(self.LAYERS, self.DEFINED, imports),
            [
                Reach("hierarchy/hierarchy.cxx.o", "hierarchy", "io", SHUTDOWN),
                Reach("hierarchy/hierarchy.cxx.o", "hierarchy", "api", PARENT),
                Reach("io/io.cxx.o", "io", "api", PARENT),
            ],
        )

    def test_an_import_nothing_defines_strongly_is_ignored(self):
        imports = {"hierarchy/hierarchy.cxx.o": {TIMER, "_CCTK_VInfo"}}
        self.assertEqual(reaches(self.LAYERS, self.DEFINED, imports), [])


class Accepted(unittest.TestCase):
    def test_the_listed_bodies_are_accepted_only_from_their_layer(self):
        reach = Reach("io/io.cxx.o", "io", "api", PARENT)
        self.assertTrue(accepted(reach, "CarpetX::level_t::parent(std::source_location) const"))
        self.assertFalse(accepted(Reach("io/io.cxx.o", "io", "schedule", PARENT), "CarpetX::level_t::parent()"))
        self.assertFalse(accepted(reach, "CarpetX::level_tx::parent()"))
        self.assertFalse(accepted(Reach("io/io.cxx.o", "io", "api", SHUTDOWN), "CarpetX::ShutdownIO()"))

    def test_every_entry_names_a_layer(self):
        from boundary.rules import LAYERS

        for layer, prefix in ACCEPTED:
            with self.subTest(prefix=prefix):
                self.assertIn(layer, LAYERS)
                self.assertTrue(prefix.startswith("CarpetX::"))

    def test_split(self):
        up = Reach("hierarchy/hierarchy.cxx.o", "hierarchy", "io", SHUTDOWN)
        ok = Reach("io/io.cxx.o", "io", "api", PARENT)
        readable = {SHUTDOWN: "CarpetX::ShutdownIO()", PARENT: "CarpetX::level_t::parent() const"}
        self.assertEqual(split([up, ok], readable), ([up], [ok]))


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


class Report(unittest.TestCase):
    UP = Reach("hierarchy/group_tags.cxx.o", "hierarchy", "comm", "_ZN7CarpetX16get_interpolatorEv")
    OK = Reach("io/io.cxx.o", "io", "api", PARENT)
    READABLE = {UP.symbol: "CarpetX::get_interpolator(std::string)", PARENT: "CarpetX::level_t::parent() const"}

    def test_clean(self):
        self.assertEqual(render([], [], {}), "✓ symbols: every CarpetX import from the importing object's layer or a lower one")

    def test_accepted_only(self):
        text = render([], [self.OK, self.OK], self.READABLE)
        self.assertEqual(
            text.split("\n"),
            [
                "accepted: io/ -> api/ 2",
                "✓ symbols: every CarpetX import from the importing object's layer or a lower one, 2 accepted imports listed in ACCEPTED",
            ],
        )

    def test_upward(self):
        lines = render([self.UP], [self.OK], self.READABLE).split("\n")
        self.assertEqual(lines[0], "hierarchy/group_tags.cxx.o   -> comm        CarpetX::get_interpolator(std::string)")
        self.assertTrue(lines[1].startswith("symbols: an object the build made"))
        self.assertEqual(lines[-2], "accepted: io/ -> api/ 1")
        self.assertEqual(lines[-1], "✗ symbols: 1 upward import, 1 accepted import listed in ACCEPTED")
        self.assertEqual(summary([self.UP, self.UP], []), "✗ symbols: 2 upward imports")
