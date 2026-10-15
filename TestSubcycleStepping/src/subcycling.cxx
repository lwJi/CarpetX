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

extern "C"
void TestSubcycleStepping_ResetTailCount(CCTK_ARGUMENTS)
{
  DECLARE_CCTK_ARGUMENTSX_TestSubcycleStepping_ResetTailCount;

  grid.loop_all<0,0,0>(grid.nghostzones, [=] CCTK_HOST(const Loop::PointDesc &pt) {
    tail_count(pt.I) = 0;
  });

}

extern "C"
void TestSubcycleStepping_CountTail(CCTK_ARGUMENTS)
{
  DECLARE_CCTK_ARGUMENTSX_TestSubcycleStepping_CountTail;

  // Everywhere, like the stamp, so that the count needs no SYNC
  grid.loop_all<0,0,0>(grid.nghostzones, [=] CCTK_HOST(const Loop::PointDesc &pt) {
    tail_count(pt.I) += 1;
  });

}

extern "C"
void TestSubcycleStepping_ResetStepCounts(CCTK_ARGUMENTS)
{
  DECLARE_CCTK_ARGUMENTSX_TestSubcycleStepping_ResetStepCounts;

  *poststep_calls = 0;
  *analysis_calls = 0;
  *last_iteration = -1;
}

namespace {
// Count one global-mode call of an end-of-step bin in this iteration
void count_step_call(const CCTK_INT iteration, CCTK_INT *const calls,
                     CCTK_INT *const poststep_calls,
                     CCTK_INT *const analysis_calls,
                     CCTK_INT *const last_iteration)
{
  if (*last_iteration != iteration) {
    *poststep_calls = 0;
    *analysis_calls = 0;
    *last_iteration = iteration;
  }
  ++*calls;
}
} // namespace

extern "C"
void TestSubcycleStepping_CountPostStepCalls(CCTK_ARGUMENTS)
{
  DECLARE_CCTK_ARGUMENTSX_TestSubcycleStepping_CountPostStepCalls;

  count_step_call(cctk_iteration, poststep_calls, poststep_calls,
                  analysis_calls, last_iteration);
}

extern "C"
void TestSubcycleStepping_CountAnalysisCalls(CCTK_ARGUMENTS)
{
  DECLARE_CCTK_ARGUMENTSX_TestSubcycleStepping_CountAnalysisCalls;

  count_step_call(cctk_iteration, analysis_calls, poststep_calls,
                  analysis_calls, last_iteration);
}

extern "C"
void TestSubcycleStepping_WriteProbe(CCTK_ARGUMENTS)
{
  DECLARE_CCTK_ARGUMENTSX_TestSubcycleStepping_WriteProbe;

  grid.loop_int<0,0,0>(grid.nghostzones, [=] CCTK_HOST(const Loop::PointDesc &pt) {
    pending_probe(pt.I) = cctk_iteration;
  });

}

extern "C"
void TestSubcycleStepping_ReadProbe(CCTK_ARGUMENTS)
{
  DECLARE_CCTK_ARGUMENTSX_TestSubcycleStepping_ReadProbe;

  // The driver checks the READS clause before the call, so on a level whose
  // ghosts the SYNC left invalid the run stops before this point
  CCTK_REAL sum = 0;
  grid.loop_all<0,0,0>(grid.nghostzones, [&] CCTK_HOST(const Loop::PointDesc &pt) {
    sum += pending_probe(pt.I);
  });
#pragma omp critical
  CCTK_VINFO("Read pending_probe everywhere at iteration %d level %d: sum %g",
             cctk_iteration, cctk_level, double(sum));

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
