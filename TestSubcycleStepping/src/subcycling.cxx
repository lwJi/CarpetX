#include "cctk.h"
#include "cctk_Arguments.h"
#include "cctk_Parameters.h"
#include "loop.hxx"

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
