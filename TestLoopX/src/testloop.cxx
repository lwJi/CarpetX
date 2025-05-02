#include <cctk.h>
#include <cctk_Arguments.h>
#include <cctk_Parameters.h>
#include <loop.hxx>
#include <loop_device.hxx>

namespace TestLoopX {
using namespace Loop;

extern "C" void TestLoopX_Init(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTSX_TestLoopX_Init;
  DECLARE_CCTK_PARAMETERS;

  grid.loop_int_device<0, 0, 0>(
      grid.nghostzones,
      [=] CCTK_DEVICE CCTK_HOST(const PointDesc &p)
          CCTK_ATTRIBUTE_ALWAYS_INLINE { testloop_gf(p.I) = 0.0; });

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

  grid.loop_outermost_int<0, 0, 0>(
      grid.nghostzones,
      [=] CCTK_DEVICE CCTK_HOST(const PointDesc &p)
          CCTK_ATTRIBUTE_ALWAYS_INLINE { testloop_gf(p.I) += 10.0; });

  grid.loop_outermost_int_device<0, 0, 0>(
      grid.nghostzones,
      [=] CCTK_DEVICE CCTK_HOST(const PointDesc &p)
          CCTK_ATTRIBUTE_ALWAYS_INLINE { testloop_gf(p.I) += 1.0; });
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

} // namespace TestLoopX
