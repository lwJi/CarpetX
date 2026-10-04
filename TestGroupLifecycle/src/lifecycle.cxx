#include <loop.hxx>

#include <cctk.h>
#include <cctk_Arguments.h>

extern "C" void TestGroupLifecycle_Init(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTSX_TestGroupLifecycle_Init;

  grid.loop_all<0, 0, 0>(grid.nghostzones,
                        [=] CCTK_HOST(const Loop::PointDesc &p) {
                          uncheckpointed_state(p.I) = 1;
                          evolved(p.I) = 17;
                        });
}

extern "C" void TestGroupLifecycle_Update(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTSX_TestGroupLifecycle_Update;

  const CCTK_REAL value = 17 + cctk_iteration;
  grid.loop_all<0, 0, 0>(grid.nghostzones,
                        [=] CCTK_HOST(const Loop::PointDesc &p) {
                          evolved(p.I) = value;
                        });
}

extern "C" void TestGroupLifecycle_ReadCurrent(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTSX_TestGroupLifecycle_ReadCurrent;

  // The negative fixture must fail while checking READS, before this body.
  grid.loop_int<0, 0, 0>(grid.nghostzones,
                        [=] CCTK_HOST(const Loop::PointDesc &p) {
                          probe(p.I) = evolved(p.I);
                        });
  CCTK_INFO("TestGroupLifecycle: current-timelevel read completed");
}
