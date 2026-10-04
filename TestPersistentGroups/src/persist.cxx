#include <loop.hxx>

#include <cctk.h>
#include <cctk_Arguments.h>
#include <cctk_Parameters.h>

#ifdef HAVE_CAPABILITY_MPI
#include <mpi.h>
#endif

#include <array>
#include <cassert>
#include <cmath>

namespace TestPersistentGroups {
using Loop::dim;
using vect3 = Arith::vect<int, dim>;

// The tested groups, in the order of the stale_count / stale_step scalars
constexpr int num_tested = 3;
constexpr const char *tested_names[num_tested] = {
    "TestPersistentGroups::persist_default",
    "TestPersistentGroups::persist_noevolve",
    "TestPersistentGroups::persist_history"};

// Stale points found by the current check pass, per level and tested group,
// on this rank. The check runs in local mode, so the count is accumulated
// here (the level is known only there) and reduced over ranks in global mode
// by TestPersistentGroups_Reduce.
constexpr int max_levels = 32;
long long stale_points[max_levels][num_tested];

// CarpetX::max_num_levels, read by name since it is private to the driver
int get_max_num_levels() {
  int type;
  const void *const ptr = CCTK_ParameterGet("max_num_levels", "CarpetX", &type);
  assert(ptr);
  assert(type == PARAMETER_INT);
  const int max_num_levels = *static_cast<const CCTK_INT *>(ptr);
  assert(max_num_levels >= 1 && max_num_levels <= max_levels);
  return max_num_levels;
}

// Number of finest-level steps one step of level `level` spans
int level_step(const int level) {
  const int lmax = get_max_num_levels();
  assert(level >= 0 && level < lmax);
  return 1 << (lmax - 1 - level);
}

// Update period of persist_history, in finest-level steps: twice level 0's
// step, so a multiple of every level's step. Every level therefore takes a
// step that ends exactly at each multiple of the period, all levels hold the
// same (tl0, tl1) pair there, and prolongating it onto points exposed by a
// regrid (which happens only at aligned times) reproduces it exactly.
int history_period() { return 2 * level_step(0); }

// The (tl0, tl1) pair persist_history holds on the interior of a level at
// time s (in finest-level steps). tl0 is the time of the last update; tl1 is
// the update before that on the step that performed the update (and at
// initial, where Init writes tl1 = -P so that the first update differs from
// a seeded copy), and a seeded copy of tl0 on every other step.
std::array<long, 2> expected_history(const long s, const bool updated_now) {
  const long period = history_period();
  assert(s >= 0);
  const long last = s / period * period;
  return {last, updated_now ? last - period : last};
}

// The level's current time, counted in finest-level steps. Under subcycling
// CarpetX sets cctk_time to cctk_delta_time (the coarse step) times the
// level's rational iteration, so the stamp is an exact integer. Valid at
// INITIAL and EVOL only: at POST_RECOVER_VARIABLES cctk_time is the time of
// the last level stepped before the checkpoint and cctk_delta_time is not yet
// set.
long time_stamp(const cGH *restrict const cctkGH) {
  const CCTK_REAL time = cctkGH->cctk_time;
  const CCTK_REAL delta_time = cctkGH->cctk_delta_time;
  return time == 0 ? 0 : std::lrint(time / delta_time * level_step(0));
}

// Stamp the tested groups on the interior with the initial time. The past
// timelevel of persist_history gets the update before the first one (-P),
// everywhere, since no SYNC fills its ghosts.
extern "C" void TestPersistentGroups_Init(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTSX_TestPersistentGroups_Init;

  const long s = time_stamp(cctkGH);
  const std::array<long, 2> history = expected_history(s, true);
  grid.loop_int<0, 0, 0>(grid.nghostzones,
                         [=] CCTK_HOST(const Loop::PointDesc &p) {
                           persist_default(p.I) = s;
                           persist_noevolve(p.I) = s;
                           persist_history(p.I) = history[0];
                         });
  grid.loop_all<0, 0, 0>(grid.nghostzones,
                         [=] CCTK_HOST(const Loop::PointDesc &p) {
                           persist_history_p(p.I) = history[1];
                         });
}

// Stamp the tested groups on the interior with the level's current time;
// persist_history only when the time is a multiple of its period. On the
// other steps its current timelevel keeps the value CycleTimelevels seeded
// from the past one.
extern "C" void TestPersistentGroups_Write(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTSX_TestPersistentGroups_Write;

  const long s = time_stamp(cctkGH);
  const bool update_history = s % history_period() == 0;
  grid.loop_int<0, 0, 0>(grid.nghostzones,
                         [=] CCTK_HOST(const Loop::PointDesc &p) {
                           persist_default(p.I) = s;
                           persist_noevolve(p.I) = s;
                           if (update_history)
                             persist_history(p.I) = s;
                         });
}

// Empty: the READS clause makes CallFunction require the past timelevel of
// persist_history to be valid everywhere during evolution
extern "C" void TestPersistentGroups_CheckPast(CCTK_ARGUMENTS) {}

// Empty: the SYNC in the schedule restores the ghosts after recovery
extern "C" void TestPersistentGroups_Sync(CCTK_ARGUMENTS) {}

extern "C" void TestPersistentGroups_Zero(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTSX_TestPersistentGroups_Zero;

  *stale_default = 0;
  *stale_noevolve = 0;
  *stale_history = 0;
}

extern "C" void TestPersistentGroups_ZeroStep(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTSX_TestPersistentGroups_ZeroStep;

  for (int level = 0; level < max_levels; ++level)
    for (int n = 0; n < num_tested; ++n)
      stale_points[level][n] = 0;
  *stale_step_default = 0;
  *stale_step_noevolve = 0;
  *stale_step_history = 0;
}

// Count the stale points of one tile. persist_history_p is the past
// timelevel of persist_history, or nullptr where it is not checked (see
// TestPersistentGroups_CheckCurrent).
//
// persist_default and persist_noevolve: every point of the tile, interior and
// ghosts, must hold the level's own stamp; a ghost on a refined level may
// instead hold the parent's stamp (a coarse-fine ghost prolongated from the
// parent, which is either aligned in time or one step of this level ahead;
// with a single timelevel there is no time interpolation). Prolongating a
// constant is exact, so any other value is stale: left over from an earlier
// prolongation. The allowed stamps are derived from the level's own interior
// value, not from cctk_time, so that the check also holds where cctk_time is
// not this level's time (after a regrid, after recovery).
//
// persist_history: the interior must hold the (tl0, tl1) pair expected at the
// level's time s (expected_history), which tells a genuine past timelevel
// from a seeded copy of the current one. s is read from persist_default for
// the same reason as above. A ghost of tl0 may hold the level's own tl0, or
// on a refined level any value between the parent's two stamps (a
// coarse-fine ghost blended in time from the parent's tl0 and tl1). Only the
// interior of tl1 is checked: SYNC does not cover the oldest timelevel, so
// after recovery its ghosts are not restored.
template <typename GF>
void check_tile(const cGH *restrict const cctkGH,
                const Loop::GridDescBase &grid, const GF &persist_default,
                const GF &persist_noevolve, const GF &persist_history,
                const GF *const persist_history_p) {
  using std::max, std::min;

  const int cctk_level = cctkGH->cctk_level;

  vect3 all_min, all_max, int_min, int_max;
  grid.domain_boxes<0, 0, 0>(grid.nghostzones, all_min, all_max, int_min,
                             int_max);
  vect3 imin, imax;
  grid.box_all<0, 0, 0>(grid.nghostzones, imin, imax);
  vect3 bnd_min, bnd_max;
  grid.boundary_box<0, 0, 0>(grid.nghostzones, bnd_min, bnd_max);

  // The level's own stamp, read at the tile's first interior point
  const vect3 I_own = max(int_min, vect3(grid.tmin));
  assert(all(I_own < min(int_max, vect3(grid.tmax))));

  const int r_own = level_step(cctk_level);
  const int r_crse = 2 * r_own;

  // Count the points of the tile (excluding the outer boundary) for which
  // ok(I, interior) is false
  const auto count_stale = [&](const int n, const auto &ok) {
    long long count = 0;
    for (int k = imin[2]; k < imax[2]; ++k) {
      for (int j = imin[1]; j < imax[1]; ++j) {
        for (int i = imin[0]; i < imax[0]; ++i) {
          const vect3 I{i, j, k};
          // Outer boundary points belong to the boundary conditions
          if (any(I < bnd_min || I >= bnd_max))
            continue;
          const bool interior = all(I >= int_min && I < int_max);
          count += !ok(I, interior);
        }
      }
    }

#pragma omp atomic
    stale_points[cctk_level][n] += count;
  };

  const auto check_stamp = [&](const int n, const GF &gf) {
    const CCTK_REAL v_own = gf(I_own);
    const long v_own_int = std::lrint(v_own);
    // Parent's stamp: equal to ours if aligned with the parent, otherwise the
    // parent has already taken the step that this level is catching up to
    const CCTK_REAL v_crse =
        v_own_int % r_crse == 0 ? v_own : CCTK_REAL(v_own_int + r_own);
    count_stale(n, [&](const vect3 &I, const bool interior) {
      const CCTK_REAL v = gf(I);
      return interior ? v == v_own
                      : v == v_own || (cctk_level > 0 && v == v_crse);
    });
  };

  const auto check_history = [&](const int n) {
    const long period = history_period();
    const long s = std::lrint(persist_default(I_own));
    const std::array<long, 2> own = expected_history(s, s % period == 0);
    // The parent is at s if aligned, otherwise one step of this level ahead
    const long s_crse = s % r_crse == 0 ? s : s + r_own;
    const std::array<long, 2> crse =
        expected_history(s_crse, s_crse % period == 0);
    const CCTK_REAL crse_min = min(crse[0], crse[1]);
    const CCTK_REAL crse_max = max(crse[0], crse[1]);
    count_stale(n, [&](const vect3 &I, const bool interior) {
      const CCTK_REAL v0 = persist_history(I);
      if (interior)
        return v0 == own[0] &&
               (!persist_history_p || (*persist_history_p)(I) == own[1]);
      return v0 == own[0] ||
             (cctk_level > 0 && v0 >= crse_min && v0 <= crse_max);
    });
  };

  // In the order of tested_names
  check_stamp(0, persist_default);
  check_stamp(1, persist_noevolve);
  check_history(2);
}

extern "C" void TestPersistentGroups_Check(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTSX_TestPersistentGroups_Check;

  check_tile(cctkGH, grid, persist_default, persist_noevolve, persist_history,
             &persist_history_p);
}

// As TestPersistentGroups_Check, without the past timelevel of
// persist_history; used at POSTREGRID when check_past_after_regrid is off
extern "C" void TestPersistentGroups_CheckCurrent(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTSX_TestPersistentGroups_CheckCurrent;

  check_tile(cctkGH, grid, persist_default, persist_noevolve, persist_history,
             decltype(&persist_history)(nullptr));
}

extern "C" void TestPersistentGroups_Reduce(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTSX_TestPersistentGroups_Reduce;

#ifdef HAVE_CAPABILITY_MPI
  MPI_Allreduce(MPI_IN_PLACE, &stale_points[0][0], max_levels * num_tested,
                MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
#endif

  long long step[num_tested] = {};
  for (int level = 0; level < max_levels; ++level) {
    for (int n = 0; n < num_tested; ++n) {
      const long long count = stale_points[level][n];
      if (count > 0 && CCTK_MyProc(cctkGH) == 0)
        CCTK_VWARN(CCTK_WARN_ALERT,
                   "%s: %lld stale points on level %d at iteration %d, "
                   "time %.17g",
                   tested_names[n], count, level, cctk_iteration,
                   double(cctk_time));
      step[n] += count;
    }
  }

  *stale_step_default = step[0];
  *stale_step_noevolve = step[1];
  *stale_step_history = step[2];
  *stale_default += *stale_step_default;
  *stale_noevolve += *stale_step_noevolve;
  *stale_history += *stale_step_history;
}

} // namespace TestPersistentGroups
