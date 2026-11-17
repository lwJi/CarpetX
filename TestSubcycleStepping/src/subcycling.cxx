#include "cctk.h"
#include "cctk_Arguments.h"
#include "cctk_Parameters.h"
#include "loop.hxx"

#include <carpetx.hxx>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

extern "C"
void TestSubcycleStepping_Init(CCTK_ARGUMENTS)
{
  DECLARE_CCTK_PARAMETERS;
  DECLARE_CCTK_ARGUMENTSX_TestSubcycleStepping_Init;

  CCTK_INFO("Initializing grid functions");
  grid.loop_int<0,0,0>(grid.nghostzones, [=](const Loop::PointDesc &pt) {
    CCTK_REAL canary = 100 + 10 * cctk_iteration + 1 * cctk_level;
    iteration(pt.I) = canary;
    postrestrict(pt.I) = canary;
    prerestrict(pt.I) = canary;
  });

}

extern "C"
void TestSubcycleStepping_Update(CCTK_ARGUMENTS)
{
  DECLARE_CCTK_PARAMETERS;
  DECLARE_CCTK_ARGUMENTSX_TestSubcycleStepping_Update;

  CCTK_VINFO("Updating grid function at iteration %d level %d time %g", cctk_iteration, cctk_level, cctk_time);
  grid.loop_int<0,0,0>(grid.nghostzones, [=] CCTK_HOST(const Loop::PointDesc &pt) {
    CCTK_REAL canary = 100 + 10 * cctk_iteration + 1 * cctk_level;
    iteration(pt.I) = canary;
  });

}

extern "C"
void TestSubcycleStepping_PostRestrict(CCTK_ARGUMENTS)
{
  DECLARE_CCTK_PARAMETERS;
  DECLARE_CCTK_ARGUMENTSX_TestSubcycleStepping_PostRestrict;

  CCTK_VINFO("Stamping postrestrict at iteration %d level %d time %g", cctk_iteration, cctk_level, cctk_time);
  grid.loop_int<0,0,0>(grid.nghostzones, [=] CCTK_HOST(const Loop::PointDesc &pt) {
    CCTK_REAL canary = 100 + 10 * cctk_iteration + 1 * cctk_level;
    postrestrict(pt.I) = canary;
  });

}

extern "C"
void TestSubcycleStepping_PreRestrict(CCTK_ARGUMENTS)
{
  DECLARE_CCTK_PARAMETERS;
  DECLARE_CCTK_ARGUMENTSX_TestSubcycleStepping_PreRestrict;

  CCTK_VINFO("Stamping prerestrict at iteration %d level %d time %g", cctk_iteration, cctk_level, cctk_time);
  grid.loop_int<0,0,0>(grid.nghostzones, [=] CCTK_HOST(const Loop::PointDesc &pt) {
    CCTK_REAL canary = 100 + 10 * cctk_iteration + 1 * cctk_level;
    prerestrict(pt.I) = canary;
  });

}

extern "C"
void TestSubcycleStepping_Sync(CCTK_ARGUMENTS)
{
  // do nothing
}

extern "C"
void TestSubcycleStepping_StampTail(CCTK_ARGUMENTS)
{
  DECLARE_CCTK_ARGUMENTSX_TestSubcycleStepping_StampTail;

  // Everywhere, so that the stamp needs no SYNC and output never shows a
  // poisoned ghost
  grid.loop_all<0,0,0>(grid.nghostzones, [=] CCTK_HOST(const Loop::PointDesc &pt) {
    tail_iteration(pt.I) = cctk_iteration;
    tail_time(pt.I) = cctk_time;
    tail_timefac(pt.I) = cctk_timefac;
  });

}

namespace {
void reset_counts(CCTK_ARGUMENTS, const CCTK_INT counting_on)
{
  DECLARE_CCTK_ARGUMENTSX_TestSubcycleStepping_ResetCounts;

  *count_post_recover = 0;
  *count_poststep = 0;
  *count_analysis = 0;
  *count_prerestrict = 0;
  *count_postrestrict = 0;
  *counting = counting_on;
}
} // namespace

extern "C"
void TestSubcycleStepping_ResetCounts(CCTK_ARGUMENTS)
{
  reset_counts(cctkGH, 0);
}

extern "C"
void TestSubcycleStepping_StartCounting(CCTK_ARGUMENTS)
{
  reset_counts(cctkGH, 1);
}

extern "C"
void TestSubcycleStepping_StopCounting(CCTK_ARGUMENTS)
{
  DECLARE_CCTK_ARGUMENTSX_TestSubcycleStepping_StopCounting;

  *counting = 0;
}

extern "C"
void TestSubcycleStepping_CountPostRecover(CCTK_ARGUMENTS)
{
  DECLARE_CCTK_ARGUMENTSX_TestSubcycleStepping_CountPostRecover;

  if (*counting)
    ++*count_post_recover;
}

extern "C"
void TestSubcycleStepping_CountPostStep(CCTK_ARGUMENTS)
{
  DECLARE_CCTK_ARGUMENTSX_TestSubcycleStepping_CountPostStep;

  if (*counting)
    ++*count_poststep;
}

extern "C"
void TestSubcycleStepping_CountAnalysis(CCTK_ARGUMENTS)
{
  DECLARE_CCTK_ARGUMENTSX_TestSubcycleStepping_CountAnalysis;

  if (*counting)
    ++*count_analysis;
}

extern "C"
void TestSubcycleStepping_CountPreRestrict(CCTK_ARGUMENTS)
{
  DECLARE_CCTK_ARGUMENTSX_TestSubcycleStepping_CountPreRestrict;

  if (*counting)
    ++*count_prerestrict;
}

extern "C"
void TestSubcycleStepping_CountPostRestrict(CCTK_ARGUMENTS)
{
  DECLARE_CCTK_ARGUMENTSX_TestSubcycleStepping_CountPostRestrict;

  if (*counting)
    ++*count_postrestrict;
}

extern "C"
void TestSubcycleStepping_SyncInTail(CCTK_ARGUMENTS)
{
  // do nothing: the SYNC clause on this routine is the test
}

extern "C"
void TestSubcycleStepping_SyncApiInTail(CCTK_ARGUMENTS)
{
  const std::array groups{
      CarpetX::gi_t{CCTK_GroupIndex("TestSubcycleStepping::tail_context")}};
  CarpetX::sync(CarpetX::active_window(), groups);
}

extern "C"
void TestSubcycleStepping_CheckTime(CCTK_ARGUMENTS)
{
  DECLARE_CCTK_PARAMETERS;
  DECLARE_CCTK_ARGUMENTSX_TestSubcycleStepping_CheckTime;

  const CCTK_REAL expected =
      cctk_initial_time + cctk_iteration * CCTK_DELTA_TIME;
  const CCTK_REAL tolerance =
      16 * std::numeric_limits<CCTK_REAL>::epsilon() *
      std::max({CCTK_REAL(1), std::fabs(cctk_initial_time),
                std::fabs(expected)});
  if (!(std::fabs(cctk_time - expected) <= tolerance))
    CCTK_VERROR("At iteration %d level %d, cctk_time is %.17g but "
                "cctk_initial_time + cctk_iteration * CCTK_DELTA_TIME is "
                "%.17g (cctk_initial_time %.17g, cctk_delta_time %.17g, "
                "cctk_timefac %d)",
                cctk_iteration, cctk_level, double(cctk_time),
                double(expected), double(cctk_initial_time),
                double(cctk_delta_time), cctk_timefac);
}
