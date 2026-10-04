#include <loop.hxx>

#include <cctk.h>
#include <cctk_Arguments.h>
#include <cctk_Parameters.h>

#ifdef HAVE_CAPABILITY_MPI
#include <mpi.h>
#endif

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cmath>

namespace TestPersistentGroups {

// Defined in persist.cxx: one level step in finest-level step units.
int level_step(int level);

namespace {
std::atomic<int> checked{0};
constexpr int checked_seed = 1;
constexpr int checked_update = 2;
constexpr int checked_regrid = 4;
constexpr int checked_recovery = 8;

// Update only every other coarse step. Regrids happen on coarse-step
// boundaries, where both history slots have the same expected values on
// every level. The nonzero offset also detects an incorrectly zeroed slot.
CCTK_REAL history_value(const long stamp) {
  const long period = 2 * level_step(0);
  return 1 + std::max(0L, stamp) / period * period;
}

template <typename Grid, typename GF>
long level_stamp(const Grid &grid, const GF &stamp) {
  Arith::vect<int, Loop::dim> imin, imax;
  grid.template box_int<0, 0, 0>(grid.nghostzones, imin, imax);
  assert(all(imin < imax));
  return std::lrint(stamp(imin));
}

template <typename Grid, typename Current, typename Previous>
void check_history(const Grid &grid, const Current &current,
                   const Previous &previous,
                   const CCTK_REAL expected_current,
                   const CCTK_REAL expected_previous, const bool interior_only,
                   const char *const phase) {
  const auto check = [&](const Loop::PointDesc &p) {
    if (current(p.I) != expected_current || previous(p.I) != expected_previous)
      CCTK_VERROR("Persistent history %s: current=%.17g (expected %.17g), "
                  "previous=%.17g (expected %.17g)",
                  phase, double(current(p.I)), double(expected_current),
                  double(previous(p.I)), double(expected_previous));
  };
  if (interior_only)
    grid.template loop_int<0, 0, 0>(grid.nghostzones, check);
  else
    grid.template loop_all<0, 0, 0>(grid.nghostzones, check);
}
} // namespace

extern "C" void TestPersistentGroups_HistoryZero(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTSX_TestPersistentGroups_HistoryZero;
  checked = 0;
  *history_checks = 0;
}

extern "C" void TestPersistentGroups_HistoryInit(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTSX_TestPersistentGroups_HistoryInit;
  grid.loop_all<0, 0, 0>(grid.nghostzones, [&](const Loop::PointDesc &p) {
    persistent_history(p.I) = history_value(0);
    persistent_history_p(p.I) = history_value(0);
  });
}

extern "C" void TestPersistentGroups_HistoryRead(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTSX_TestPersistentGroups_HistoryRead;
  // persist_default has not been updated yet and still holds this level's
  // previous time. CycleTimelevels must have copied that history into tl=0,
  // while keeping the genuine previous state in tl=1.
  const CCTK_REAL expected = history_value(level_stamp(grid, persist_default));
  check_history(grid, persistent_history, persistent_history_p, expected,
                expected, false, "after cycling, before update");
  checked.fetch_or(checked_seed);
}

extern "C" void TestPersistentGroups_HistoryUpdate(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTSX_TestPersistentGroups_HistoryUpdate;
  const long stamp = level_stamp(grid, persist_default);
  const CCTK_REAL expected = history_value(stamp);
  if (stamp % (2 * level_step(0)) == 0) {
    grid.loop_all<0, 0, 0>(grid.nghostzones, [&](const Loop::PointDesc &p) {
      persistent_history(p.I) = expected;
    });
  }
  // Most steps intentionally leave tl=0 untouched. A real update must not
  // overwrite tl=1, which differs from the new value at every update.
  check_history(grid, persistent_history, persistent_history_p, expected,
                history_value(stamp - level_step(cctk_level)), false,
                "after occasional update");
  if (stamp % (2 * level_step(0)) == 0)
    checked.fetch_or(checked_update);
}

extern "C" void TestPersistentGroups_HistoryRegrid(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTSX_TestPersistentGroups_HistoryRegrid;
  const long stamp = level_stamp(grid, persist_default);
  check_history(grid, persistent_history, persistent_history_p,
                history_value(stamp),
                history_value(stamp - level_step(cctk_level)), false,
                "after regrid");
  // Do not count creation of the initial hierarchy as moving-grid coverage.
  if (stamp > 0)
    checked.fetch_or(checked_regrid);
}

extern "C" void TestPersistentGroups_HistoryRecover(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTSX_TestPersistentGroups_HistoryRecover;
  const long stamp = level_stamp(grid, persist_default);
  const CCTK_REAL expected = history_value(stamp);
  const CCTK_REAL expected_previous =
      history_value(stamp - level_step(cctk_level));
  // Checkpoint files omit ghosts. Check both recovered interiors before
  // writing anything; restore only omitted points, including the old TL
  // that the driver's current-timelevel-only SYNC cannot fill.
  check_history(grid, persistent_history, persistent_history_p, expected,
                expected_previous, true, "after recovery");
  Arith::vect<int, Loop::dim> all_min, all_max, int_min, int_max;
  grid.domain_boxes<0, 0, 0>(grid.nghostzones, all_min, all_max, int_min,
                            int_max);
  grid.loop_all<0, 0, 0>(grid.nghostzones, [&](const Loop::PointDesc &p) {
    if (any(p.I < int_min || p.I >= int_max)) {
      persistent_history(p.I) = expected;
      persistent_history_p(p.I) = expected_previous;
    }
  });
  checked.fetch_or(checked_recovery);
}

extern "C" void TestPersistentGroups_HistorySummary(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTSX_TestPersistentGroups_HistorySummary;
  int result = checked.load();
#ifdef HAVE_CAPABILITY_MPI
  MPI_Allreduce(MPI_IN_PLACE, &result, 1, MPI_INT, MPI_BOR, MPI_COMM_WORLD);
#endif
  *history_checks = result;
}

extern "C" void TestPersistentGroups_HistoryDone(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTSX_TestPersistentGroups_HistoryDone;
  DECLARE_CCTK_PARAMETERS;
  const int required = checked_seed | checked_update | checked_regrid |
                       (history_expect_recovery ? checked_recovery : 0);
  if ((*history_checks & required) != required)
    CCTK_VERROR("Persistent history checks did not cover the required events: "
                "got %d, need %d",
                int(*history_checks), required);
  CCTK_VINFO("Persistent history checks passed (mask %d)", int(*history_checks));
}

} // namespace TestPersistentGroups
