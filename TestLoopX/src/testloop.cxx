#include <cctk.h>
#include <cctk_Arguments.h>
#include <cctk_Parameters.h>
#include <loop.hxx>
#include <loop_device.hxx>
#include <driver.hxx>

namespace TestLoopX {
using namespace Loop;

extern "C" void TestLoopX_Init(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTSX_TestLoopX_Init;
  DECLARE_CCTK_PARAMETERS;

  grid.loop_int_device<0, 0, 0>(
      grid.nghostzones,
      [=] CCTK_DEVICE CCTK_HOST(const PointDesc &p)
          CCTK_ATTRIBUTE_ALWAYS_INLINE {
          testloop_gf(p.I) = 0.0;
      });

  grid.loop_mix_device<0, 1, 1>(
      grid.nghostzones,
      [=] CCTK_DEVICE CCTK_HOST(const PointDesc &p)
          CCTK_ATTRIBUTE_ALWAYS_INLINE { testloop_gf_fx(p.I) = 0.0; });

  grid.loop_mix_device<1, 0, 1>(
      grid.nghostzones,
      [=] CCTK_DEVICE CCTK_HOST(const PointDesc &p)
          CCTK_ATTRIBUTE_ALWAYS_INLINE { testloop_gf_fy(p.I) = 0.0; });

  grid.loop_mix_device<1, 1, 0>(
      grid.nghostzones,
      [=] CCTK_DEVICE CCTK_HOST(const PointDesc &p)
          CCTK_ATTRIBUTE_ALWAYS_INLINE { testloop_gf_fz(p.I) = 0.0; });
}

extern "C" void TestLoopX_Sync(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTSX_TestLoopX_Sync;
}

extern "C" void TestLoopX_OutermostInterior(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTSX_TestLoopX_OutermostInterior;
  DECLARE_CCTK_PARAMETERS;

  const auto symmetries = CarpetX::ghext->patchdata.at(cctk_patch).symmetries;
  const vect<vect<bool, Loop::dim>, 2> is_sym_bnd {
    {
      symmetries[0][0] != CarpetX::symmetry_t::none,
      symmetries[0][1] != CarpetX::symmetry_t::none,
      symmetries[0][2] != CarpetX::symmetry_t::none
    },
    {
      symmetries[1][0] != CarpetX::symmetry_t::none,
      symmetries[1][1] != CarpetX::symmetry_t::none,
      symmetries[1][2] != CarpetX::symmetry_t::none
    }
  };
  grid.loop_outermost_int<0, 0, 0>(
      grid.nghostzones, is_sym_bnd,
      [=] CCTK_DEVICE CCTK_HOST(const PointDesc &p)
          CCTK_ATTRIBUTE_ALWAYS_INLINE {
          testloop_gf(p.I) += 10.0;
      });

  grid.loop_outermost_int_device<0, 0, 0>(
      grid.nghostzones, is_sym_bnd,
      [=] CCTK_DEVICE CCTK_HOST(const PointDesc &p)
          CCTK_ATTRIBUTE_ALWAYS_INLINE {
          testloop_gf(p.I) += 1.0;
      });
}

extern "C" void TestLoopX_Mix(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTSX_TestLoopX_Mix;
  DECLARE_CCTK_PARAMETERS;

  grid.loop_mix_device<0, 1, 1>(
      grid.nghostzones,
      [=] CCTK_DEVICE CCTK_HOST(const PointDesc &p)
          CCTK_ATTRIBUTE_ALWAYS_INLINE { testloop_gf_fx(p.I) += 1.0; });

  grid.loop_mix_device<1, 0, 1>(
      grid.nghostzones,
      [=] CCTK_DEVICE CCTK_HOST(const PointDesc &p)
          CCTK_ATTRIBUTE_ALWAYS_INLINE { testloop_gf_fy(p.I) += 1.1; });

  grid.loop_mix_device<1, 1, 0>(
      grid.nghostzones,
      [=] CCTK_DEVICE CCTK_HOST(const PointDesc &p)
          CCTK_ATTRIBUTE_ALWAYS_INLINE { testloop_gf_fz(p.I) += 1.2; });
}

// Partition test for loop_allmn_device(ord) / loop_outer_n_device(ord) and
// loop_allm1_device on a box that is tiled in all three directions. Each
// primitive adds its own marker, so the final value of a point tells which
// primitives visited it:
//
//   testloop_gf_n1 (VVV): allm1 += 100, allmn(1) += 1, outer_n(1) += 10
//     -> every point must read 101 (allm1 and allmn(1) agree) or 10 (outer_n)
//   testloop_gf_n2 (CCC): allmn(2) += 1, outer_n(2) += 10
//     -> every point must read 1 or 10
//
// 0 means neither primitive visited the point; 11, 20, 102, 111 mean a
// double visit; 100 or 1 alone on n1 means allm1 and allmn(1) disagree.
// The zeroing pass with loop_all_device covers the whole box, so it also
// overwrites CarpetX's poison; TestLoopX_OuterN_Check therefore does the
// point-by-point check that the poison scan cannot do here.
extern "C" void TestLoopX_OuterN(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTSX_TestLoopX_OuterN;
  DECLARE_CCTK_PARAMETERS;

  grid.loop_all_device<0, 0, 0>(
      grid.nghostzones,
      [=] CCTK_DEVICE CCTK_HOST(const PointDesc &p)
          CCTK_ATTRIBUTE_ALWAYS_INLINE { testloop_gf_n1(p.I) = 0.0; });
  grid.loop_all_device<1, 1, 1>(
      grid.nghostzones,
      [=] CCTK_DEVICE CCTK_HOST(const PointDesc &p)
          CCTK_ATTRIBUTE_ALWAYS_INLINE { testloop_gf_n2(p.I) = 0.0; });

  grid.loop_allm1_device<0, 0, 0>(
      grid.nghostzones,
      [=] CCTK_DEVICE CCTK_HOST(const PointDesc &p)
          CCTK_ATTRIBUTE_ALWAYS_INLINE { testloop_gf_n1(p.I) += 100.0; });
  grid.loop_allmn_device<0, 0, 0>(
      grid.nghostzones, 1,
      [=] CCTK_DEVICE CCTK_HOST(const PointDesc &p)
          CCTK_ATTRIBUTE_ALWAYS_INLINE { testloop_gf_n1(p.I) += 1.0; });
  grid.loop_outer_n_device<0, 0, 0>(
      grid.nghostzones, 1,
      [=] CCTK_DEVICE CCTK_HOST(const PointDesc &p)
          CCTK_ATTRIBUTE_ALWAYS_INLINE { testloop_gf_n1(p.I) += 10.0; });

  grid.loop_allmn_device<1, 1, 1>(
      grid.nghostzones, 2,
      [=] CCTK_DEVICE CCTK_HOST(const PointDesc &p)
          CCTK_ATTRIBUTE_ALWAYS_INLINE { testloop_gf_n2(p.I) += 1.0; });
  grid.loop_outer_n_device<1, 1, 1>(
      grid.nghostzones, 2,
      [=] CCTK_DEVICE CCTK_HOST(const PointDesc &p)
          CCTK_ATTRIBUTE_ALWAYS_INLINE { testloop_gf_n2(p.I) += 10.0; });
}

// Whole-box check of the markers set by TestLoopX_OuterN, ghost points
// included. The TSV axis cuts and the norms (interior only) cannot see the
// ghost edges and corners where the tangential-band gap of the old
// loop_outer_n_device lived, so the check is done here and aborts the run.
extern "C" void TestLoopX_OuterN_Check(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTSX_TestLoopX_OuterN_Check;
  DECLARE_CCTK_PARAMETERS;

  int nbad_n1 = 0, nbad_n2 = 0;
  vect<int, dim> first_n1{0, 0, 0}, first_n2{0, 0, 0};
  CCTK_REAL first_val_n1 = 0, first_val_n2 = 0;

  grid.loop_all<0, 0, 0>(grid.nghostzones, [&](const PointDesc &p) {
    const CCTK_REAL v = testloop_gf_n1(p.I);
    if (!(v == 101.0 || v == 10.0)) {
      if (nbad_n1 == 0) {
        first_n1 = p.I;
        first_val_n1 = v;
      }
      ++nbad_n1;
    }
  });
  grid.loop_all<1, 1, 1>(grid.nghostzones, [&](const PointDesc &p) {
    const CCTK_REAL v = testloop_gf_n2(p.I);
    if (!(v == 1.0 || v == 10.0)) {
      if (nbad_n2 == 0) {
        first_n2 = p.I;
        first_val_n2 = v;
      }
      ++nbad_n2;
    }
  });

  if (nbad_n1 != 0 || nbad_n2 != 0) {
    // The scheduled function runs once per tile from an OpenMP task list;
    // let only one tile report and abort
#pragma omp critical(TestLoopX_OuterN_Check)
    CCTK_VERROR("loop_allmn_device / loop_outer_n_device / loop_allm1_device "
                "do not partition the box on tile [%d,%d,%d]-[%d,%d,%d) of "
                "lsh [%d,%d,%d]: testloop_gf_n1 has %d bad points (first at "
                "[%d,%d,%d] = %g, expected 101 or 10), testloop_gf_n2 has %d "
                "bad points (first at [%d,%d,%d] = %g, expected 1 or 10)",
                grid.tmin[0], grid.tmin[1], grid.tmin[2], grid.tmax[0],
                grid.tmax[1], grid.tmax[2], grid.lsh[0], grid.lsh[1],
                grid.lsh[2], nbad_n1, first_n1[0], first_n1[1], first_n1[2],
                double(first_val_n1), nbad_n2, first_n2[0], first_n2[1],
                first_n2[2], double(first_val_n2));
  }
}

} // namespace TestLoopX
