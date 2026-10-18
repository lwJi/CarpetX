import unittest

from files import Lang, covered


class Covered(unittest.TestCase):
    def test_cxx_needs_a_src_directory(self):
        self.assertIs(covered("CarpetX/src/driver.cxx"), Lang.CXX)
        self.assertIs(covered("CarpetX/src/typeprops.hh"), Lang.CXX)
        self.assertIs(covered("Loop/src/loopcontrol.h"), Lang.CXX)
        self.assertIs(covered("A/src/sub/dir/x.hpp"), Lang.CXX)
        self.assertIsNone(covered("CarpetX/stencils/x.cxx"))
        self.assertIsNone(covered("src.cxx"))

    def test_every_ccl_file(self):
        for name in ["X/interface.ccl", "X/param.ccl", "X/schedule.ccl",
                     "X/configuration.ccl", "X/test/test.ccl", "TestInterpolate/test/tests.ccl"]:
            self.assertIs(covered(name), Lang.CCL, name)

    def test_par_files_including_disabled_ones(self):
        self.assertIs(covered("X/test/a.par"), Lang.PAR)
        self.assertIs(covered("X/par/a.par"), Lang.PAR)
        self.assertIs(covered("TestProlongate/test/test_eno_o5.par-DISABLED"), Lang.PAR)
        self.assertIs(covered("BenchX/par/benchmark-carpet.par"), Lang.PAR)
        self.assertIsNone(covered("X/test/a.parfile"))

    def test_not_covered(self):
        for name in ["X/src/make.code.defn", "X/README", "scripts/build.sh",
                     "BenchX/bin/run.jl", "scripts/comments/rules.py", "X/test/a/norms/x.tsv"]:
            self.assertIsNone(covered(name), name)


if __name__ == "__main__":
    unittest.main()
