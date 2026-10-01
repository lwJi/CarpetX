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
constexpr int num_tested = 2;
constexpr const char *tested_names[num_tested] = {
    "TestPersistentGroups::persist_default",
    "TestPersistentGroups::persist_noevolve"};

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

// Stamp both tested groups on the interior with the current time, counted in
// finest-level steps. Under subcycling CarpetX sets cctk_time to
// cctk_delta_time (the coarse step) times the level's rational iteration, so
// the stamp is an exact integer.
void write_stamp(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTSX_TestPersistentGroups_Write;

  const CCTK_REAL stamp =
      cctk_time == 0 ? 0
                     : std::lrint(cctk_time / cctk_delta_time * level_step(0));
  grid.loop_int<0, 0, 0>(grid.nghostzones,
                         [=] CCTK_HOST(const Loop::PointDesc &p) {
                           persist_default(p.I) = stamp;
                           persist_noevolve(p.I) = stamp;
                         });
}

extern "C" void TestPersistentGroups_Init(CCTK_ARGUMENTS) {
  write_stamp(cctkGH);
}

extern "C" void TestPersistentGroups_Write(CCTK_ARGUMENTS) {
  write_stamp(cctkGH);
}

extern "C" void TestPersistentGroups_Zero(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTSX_TestPersistentGroups_Zero;

  *stale_default = 0;
  *stale_noevolve = 0;
}

extern "C" void TestPersistentGroups_ZeroStep(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTSX_TestPersistentGroups_ZeroStep;

  for (int level = 0; level < max_levels; ++level)
    for (int n = 0; n < num_tested; ++n)
      stale_points[level][n] = 0;
  *stale_step_default = 0;
  *stale_step_noevolve = 0;
}

// Every point of the tile, interior and ghosts, must hold the level's own
// stamp; a ghost on a refined level may instead hold the parent's stamp (a
// coarse-fine ghost prolongated from the parent, which is either aligned in
// time or one step of this level ahead; with a single timelevel there is no
// time interpolation). Prolongating a constant is exact, so any other value
// is stale: left over from an earlier prolongation. The allowed stamps are
// derived from the level's own interior value, not from cctk_time, so that
// the check also holds where cctk_time is not this level's time (after a
// regrid, after recovery).
extern "C" void TestPersistentGroups_Check(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTSX_TestPersistentGroups_Check;

  using std::max, std::min;

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

  const auto check = [&](const int n, const auto &gf) {
    const CCTK_REAL v_own = gf(I_own);
    const long v_own_int = std::lrint(v_own);
    // Parent's stamp: equal to ours if aligned with the parent, otherwise the
    // parent has already taken the step that this level is catching up to
    const CCTK_REAL v_crse =
        v_own_int % r_crse == 0 ? v_own : CCTK_REAL(v_own_int + r_own);

    long long count = 0;
    for (int k = imin[2]; k < imax[2]; ++k) {
      for (int j = imin[1]; j < imax[1]; ++j) {
        for (int i = imin[0]; i < imax[0]; ++i) {
          const vect3 I{i, j, k};
          // Outer boundary points belong to the boundary conditions
          if (any(I < bnd_min || I >= bnd_max))
            continue;
          const bool interior = all(I >= int_min && I < int_max);
          const CCTK_REAL v = gf(I);
          const bool ok = interior
                              ? v == v_own
                              : v == v_own || (cctk_level > 0 && v == v_crse);
          count += !ok;
        }
      }
    }

#pragma omp atomic
    stale_points[cctk_level][n] += count;
  };

  // In the order of tested_names
  check(0, persist_default);
  check(1, persist_noevolve);
}

extern "C" void TestPersistentGroups_Reduce(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTSX_TestPersistentGroups_Reduce;

#ifdef HAVE_CAPABILITY_MPI
  MPI_Allreduce(MPI_IN_PLACE, &stale_points[0][0], max_levels * num_tested,
                MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
#endif

  long long step[num_tested] = {0, 0};
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
  *stale_default += *stale_step_default;
  *stale_noevolve += *stale_step_noevolve;
}

} // namespace TestPersistentGroups
