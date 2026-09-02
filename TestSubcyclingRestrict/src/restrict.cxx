#include "CarpetX/CarpetX/src/driver.hxx"
#include "CarpetX/CarpetX/src/schedule.hxx"

#include <loop.hxx>

#include <cctk.h>
#include <cctk_Arguments.h>
#include <cctk_Parameters.h>

#include <vector>

extern "C" void TestSubcyclingRestrict_Init(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTSX_TestSubcyclingRestrict_Init;
  DECLARE_CCTK_PARAMETERS;

  grid.loop_int<0, 0, 0>(grid.nghostzones, [=](const Loop::PointDesc &pt) {
    canary(pt.I) = 100 + 10 * cctk_iteration + 1 * cctk_level;
  });
}

extern "C" void TestSubcyclingRestrict_Update(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTSX_TestSubcyclingRestrict_Update;
  DECLARE_CCTK_PARAMETERS;

  grid.loop_int<0, 0, 0>(grid.nghostzones, [=](const Loop::PointDesc &pt) {
    canary(pt.I) = 100 + 10 * cctk_iteration + 1 * cctk_level;
  });
}

// Called at POSTSTEP on every pass, aligned or not. RestrictIfAligned decides
// per level pair whether the clocks match; this routine never checks them.
extern "C" void TestSubcyclingRestrict_Restrict(CCTK_ARGUMENTS) {
  static const std::vector<int> groups = {
      CCTK_GroupIndex("TestSubcyclingRestrict::canary")};
  for (int level = CarpetX::ghext->num_levels() - 2; level >= 0; --level)
    CarpetX::RestrictIfAligned(cctkGH, level, groups);
}
