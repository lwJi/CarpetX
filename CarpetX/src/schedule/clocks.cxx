#include "schedule/clocks.hxx"
#include "hierarchy/driver.hxx"
#include "hierarchy/loops.hxx"
#include "hierarchy/operations.hxx"
#include "hierarchy/valid.hxx"
#include "comm/comm.hxx"
#include "comm/reflux.hxx"
#include "comm/subcycling.hxx"
#include "io/io.hxx"
#include "traversal/traversal.hxx"
#include <carpetx.hxx>

#include <cctk.h>
#include <cctk_Parameters.h>

#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>

#include <mpi.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

namespace CarpetX {

level_t WidenMinLevel(level_t level, const rat64 target_iteration) {
  while (level.index() > 0 &&
         driver().level(level.parent()).iteration == target_iteration)
    level = level.parent();
  return level;
}

namespace {
window_t ClockGroupBelow(const window_t window) {
  const level_t finest = window.finest();
  const level_t coarsest =
      WidenMinLevel(finest, driver().level(finest).iteration);
  return window.narrow(coarsest.index(), finest.index() + 1);
}

std::vector<window_t> ClockGroups() {
  const GHExt &ext = driver();
  std::vector<window_t> groups;
  for (window_t below = all_levels(); below.level_bounds().max > 0;
       below = below.narrow(0, groups.back().level_bounds().min))
    groups.push_back(ClockGroupBelow(below));
  std::reverse(groups.begin(), groups.end());
  assert((ext.use_subcycling || groups.size() <= 1) &&
         "without subcycling every level shares one clock");
  return groups;
}
} // namespace

window_t FinestClockGroup() {
  const GHExt &ext = driver();
  const window_t group = ClockGroupBelow(all_levels());
  assert((ext.use_subcycling || group.level_bounds().min == 0) &&
         "without subcycling the end-of-step window holds every level");
  return group;
}

namespace {
int LastStepIteration(const level_t level, const int current_iteration) {
  const GHExt &ext = driver();
  const auto &finest = ext.level(all_levels().finest());
  const auto &leveldata = ext.level(level);
  const rat64 steps_since =
      (finest.iteration - (leveldata.iteration - leveldata.delta_iteration)) /
      finest.delta_iteration;
  assert(steps_since.den == 1 && steps_since.num >= 1);
  const int iteration = current_iteration + 1 - int(steps_since.num);
  assert(iteration >= 0 && iteration <= current_iteration);
  return iteration;
}

std::string format_clock(const rat64 &clock) {
  std::ostringstream buf;
  buf << clock.num;
  if (clock.den != 1)
    buf << "/" << clock.den;
  return buf.str();
}
} // namespace

CCTK_REAL get_coarse_mindx() {
  CCTK_REAL mindx = std::numeric_limits<CCTK_REAL>::infinity();
  for (const auto &patchdata : driver().patches()) {
    const amrex::Geometry &geom = patchdata.amrcore->Geom(0);
    const CCTK_REAL *restrict const dx = geom.CellSize();
    for (int d = 0; d < dim; ++d)
      mindx = fmin(mindx, dx[d]);
  }
  return mindx;
}

CCTK_REAL get_finest_mindx() {
  CCTK_REAL mindx = std::numeric_limits<CCTK_REAL>::infinity();
  for (const auto &patchdata : driver().patches()) {
    const amrex::Geometry &geom = patchdata.amrcore->Geom(0);
    const CCTK_REAL *restrict const dx = geom.CellSize();
    CCTK_REAL mindx1 = std::numeric_limits<CCTK_REAL>::infinity();
    for (int d = 0; d < dim; ++d)
      mindx1 = fmin(mindx1, dx[d]);
    mindx1 = ldexp(mindx1, -(patchdata.num_levels() - 1));
    mindx = fmin(mindx, mindx1);
  }
  return mindx;
}

void SetDeltaTime(cGH *restrict const cctkGH) {
  DECLARE_CCTK_PARAMETERS;
  if (CCTK_EQUALS(timestep_choice, "timestep")) {
    cctkGH->cctk_delta_time = timestep;
  } else if (CCTK_EQUALS(timestep_choice, "dtfac")) {
    cctkGH->cctk_delta_time =
        dtfac *
        (driver().use_subcycling ? get_coarse_mindx() : get_finest_mindx());
  } else {
    CCTK_ERROR("Unexpected value for 'CarpetX::timestep_choice'");
    abort();
  }
  using std::isfinite;
  assert(isfinite(cctkGH->cctk_delta_time));
  update_cctkGHs(cctkGH);
}

void SetStepTime(cGH *restrict const cctkGH, const level_t level,
                 const rat64 clock) {
  if (driver().use_subcycling) {
    cctkGH->cctk_timefac = 1 << level.index();
    cctkGH->cctk_time = cctkGH->cctk_delta_time * double(clock);
  } else {
    cctkGH->cctk_timefac = 1;
  }
}

static void RestoreLevelClocks(const cGH *restrict const cctkGH,
                               const checkpoint_record_t &record) {
  const int checkpoint_iteration = cctkGH->cctk_iteration;

  const GHExt &ext = driver();
  const int iteration_ratio = 1 << (ext.num_levels() - 1);
  bool have_last_step_iterations = true;
  for (const auto &patchdata : ext.patches()) {
    for (const auto &leveldata : patchdata.levels()) {
      const level_record_t &recovered = record.level_record(leveldata.level);
      restore_level_clock(leveldata.level,
                          recovered.iteration
                              ? *recovered.iteration
                              : rat64(checkpoint_iteration) / iteration_ratio,
                          recovered.last_step_iteration);
      if (!recovered.last_step_iteration)
        have_last_step_iterations = false;
    }
  }

  const auto clock_groups = ClockGroups();
  const int num_groups = clock_groups.size();

  std::ostringstream clocks_buf;
  for (const auto &leveldata : ext.patch(patch_t{0}).levels())
    clocks_buf << (leveldata.level.index() == 0 ? "" : ", ") << "L"
               << leveldata.level.index() << " "
               << format_clock(leveldata.iteration);
  const std::string clocks = clocks_buf.str();
#pragma omp critical
  CCTK_VINFO("Level clocks (coarse steps): %s -> %d clock group%s",
             clocks.c_str(), num_groups, num_groups == 1 ? "" : "s");

  if (!have_last_step_iterations) {
    for (const window_t &group : clock_groups) {
      const auto [min_level, max_level] = group.level_bounds();
      const int last_step_iteration =
          LastStepIteration(group.finest(), checkpoint_iteration);
      for (const auto &patchdata : ext.patches())
        for (const auto &leveldata : patchdata.levels())
          if (leveldata.level.index() >= min_level &&
              leveldata.level.index() < max_level)
            restore_level_clock(leveldata.level, leveldata.iteration,
                                last_step_iteration);
    }
    if (num_groups >= 2 && CCTK_MyProc(nullptr) == 0)
      CCTK_VWARN(CCTK_WARN_ALERT,
                 "Checkpoint at iteration %d predates per-level last-step "
                 "iterations; derived them from the %d distinct level clocks "
                 "(%s), assuming that no regrid has changed the number of "
                 "levels since the coarsest clock group last stepped.",
                 checkpoint_iteration, num_groups, clocks.c_str());
  }

  for (const window_t &group : clock_groups) {
    const auto [min_level, max_level] = group.level_bounds();
    const int group_last_step_iteration =
        ext.level(group.finest()).last_step_iteration;
    for (const auto &patchdata : ext.patches())
      for (const auto &leveldata : patchdata.levels())
        if (leveldata.level.index() >= min_level &&
            leveldata.level.index() < max_level &&
            leveldata.last_step_iteration != group_last_step_iteration)
          CCTK_VERROR("Checkpoint at iteration %d is inconsistent: patch %d "
                      "level %d has last-step iteration %d, but its "
                      "clock group [%d,%d) has %d",
                      checkpoint_iteration, leveldata.level.patch().index(),
                      leveldata.level.index(), leveldata.last_step_iteration,
                      min_level, max_level, group_last_step_iteration);
  }

  const int finest_last_step_iteration =
      ext.level(all_levels().finest()).last_step_iteration;
  if (finest_last_step_iteration != checkpoint_iteration)
    CCTK_VERROR("Checkpoint at iteration %d is inconsistent: its finest clock "
                "group has last-step iteration %d",
                checkpoint_iteration, finest_last_step_iteration);
}

static void RecoverHierarchy(cGH *restrict const cctkGH) {
  const checkpoint_record_t record = RecoverGridStructure(cctkGH);

  {
    const traversal_scope_t scope(all_levels());

    CCTK_Traverse(cctkGH, "CCTK_BASEGRID");

    const char *recovery_mode = *static_cast<const char *const *>(
        CCTK_ParameterGet("recovery_mode", "Cactus", nullptr));
    if (!CCTK_Equals(recovery_mode, "strict")) {
      CCTK_Traverse(cctkGH, "CCTK_INITIAL");
      CCTK_Traverse(cctkGH, "CCTK_POSTINITIAL");
      CCTK_Traverse(cctkGH, "CCTK_POSTPOSTINITIAL");
    }

    RecoverGH(cctkGH, record);
    CCTK_Traverse(cctkGH, "CCTK_RECOVER_VARIABLES");
  }

  const GHExt &ext = driver();
  for (const auto &patchdata : ext.patches())
    for (const auto &leveldata : patchdata.levels())
      assert(leveldata.iteration == 0 &&
             "recovery runs BASEGRID and RECOVER_VARIABLES at clock 0");

  RestoreLevelClocks(cctkGH, record);

  for (const auto &patchdata : ext.patches())
    mark_initialized(patchdata.patch);

  SetDeltaTime(cctkGH);
}

void TraverseTailBin(cGH *restrict const cctkGH, const char *const bin) {
  const tail_bin_scope_t scope(bin);
  CCTK_Traverse(cctkGH, bin);
}

void FinalizeWindow(const window_t window, cGH *restrict const cctkGH,
                    const bool reflux) {
  DECLARE_CCTK_PARAMETERS;

  assert(FinestClockGroup() == window);
  const GHExt &ext = driver();
  const auto &finest = ext.level(window.finest());
  SetStepTime(cctkGH, finest.level, finest.iteration);
  loop_serially(window, [&](auto &restrict leveldata) {
    assert(leveldata.last_step_iteration == cctkGH->cctk_iteration);
  });

  if (!window_is_restricted(window)) {
    if (verbose)
#pragma omp critical
      CCTK_VINFO("Skipping restriction at iteration %d: only level %d is "
                 "at the current time",
                 cctkGH->cctk_iteration, window.coarsest().index());
  } else if (!restrict_during_sync) {
    CCTK_Traverse(cctkGH, "CarpetX_PreRestrict");
    if (reflux)
      Reflux(window);
    assert(do_restrict);
    const std::vector<gi_t> groups = collect_restrictable_groups();
    if (ext.use_subcycling)
      RestrictNoPoison(window, groups);
    else
      Restrict(window, groups);
    ProlongateAfterRestriction(window);
    CCTK_Traverse(cctkGH, "CCTK_POSTRESTRICT");
  }
}

static void RecomputePerClockGroup(cGH *restrict const cctkGH) {
  const recovery_scope_t scope;

  for (const window_t &group : ClockGroups()) {
    const auto [min_level, max_level] = group.level_bounds();
    const auto &finest = driver().level(group.finest());

    const traversal_scope_t window_scope(group);

    cctkGH->cctk_iteration = finest.last_step_iteration;
    SetStepTime(cctkGH, finest.level, finest.iteration);

#pragma omp critical
    CCTK_VINFO("POST_RECOVER_VARIABLES on levels [%d,%d): iteration %d, "
               "time %g, timefac %d",
               min_level, max_level, cctkGH->cctk_iteration,
               double(cctkGH->cctk_time), cctkGH->cctk_timefac);

    RecoverRKBoundary(group, cctkGH);
    CCTK_Traverse(cctkGH, "CCTK_POST_RECOVER_VARIABLES");
  }

#pragma omp critical
  CCTK_VINFO("Recovery does not rerun restriction, POSTSTEP or ANALYSIS");
}

void RecoverFromCheckpoint(cGH *restrict const cctkGH) {
  DECLARE_CCTK_PARAMETERS;

#pragma omp critical
  CCTK_VINFO("Recovering from checkpoint...");

  RecoverHierarchy(cctkGH);

  RecomputePerClockGroup(cctkGH);

  assert(cctkGH->cctk_iteration == recovered_iteration());

#pragma omp critical
  CCTK_VINFO("Iteration: %d   time: %g   delta_time: %g",
             cctkGH->cctk_iteration, double(cctkGH->cctk_time),
             double(cctkGH->cctk_delta_time));

  const traversal_scope_t scope(FinestClockGroup());

  CCTK_Traverse(cctkGH, "CCTK_CPINITIAL");
  if (out_initial_data)
    CCTK_OutputGH(cctkGH);
}

bool EvolutionIsDone(cGH *restrict const cctkGH) {
  DECLARE_CCTK_PARAMETERS;

  const bool max_iteration_reached = cctkGH->cctk_iteration >= cctk_itlast;

  const bool max_simulation_time_reached =
      cctk_initial_time < cctk_final_time
          ? cctkGH->cctk_time >= cctk_final_time
          : cctkGH->cctk_time <= cctk_final_time;

  const int runtime = CCTK_RunTime();
  const bool max_runtime_reached = runtime >= 60 * max_runtime;

  int we_are_done;
  if (terminate_next || CCTK_TerminationReached(cctkGH))
    we_are_done = true;
  else if (CCTK_Equals(terminate, "never"))
    we_are_done = false;
  else if (CCTK_Equals(terminate, "iteration"))
    we_are_done = max_iteration_reached;
  else if (CCTK_Equals(terminate, "time"))
    we_are_done = max_simulation_time_reached;
  else if (CCTK_Equals(terminate, "runtime"))
    we_are_done = max_runtime_reached;
  else if (CCTK_Equals(terminate, "any"))
    we_are_done = max_iteration_reached || max_simulation_time_reached ||
                  max_runtime_reached;
  else if (CCTK_Equals(terminate, "all"))
    we_are_done = max_iteration_reached && max_simulation_time_reached &&
                  max_runtime_reached;
  else if (CCTK_Equals(terminate, "either"))
    we_are_done = max_iteration_reached || max_simulation_time_reached;
  else if (CCTK_Equals(terminate, "both"))
    we_are_done = max_iteration_reached && max_simulation_time_reached;
  else
    CCTK_ERROR("internal error");

  MPI_Allreduce(MPI_IN_PLACE, &we_are_done, 1, MPI_INT, MPI_LOR,
                MPI_COMM_WORLD);

  return we_are_done;
}

namespace {

struct seed_run_t {
  window_t levels;
  valid_t seed;
};

std::vector<seed_run_t> seed_runs(const window_t window, const gi_t gi,
                                  const vi_t vi) {
  const auto levels = window.level_bounds();
  std::vector<std::optional<valid_t> > seeds(levels.max - levels.min);
  loop_serially(window, [&](auto &restrict leveldata) {
    std::optional<valid_t> &seed =
        seeds.at(leveldata.level.index() - levels.min);
    const valid_t level_seed = leveldata.group(gi).timelevel(tl_t{1}).valid(vi);
    assert(!seed || *seed == level_seed);
    seed = level_seed;
  });
  std::vector<seed_run_t> runs;
  for (int lo = levels.min; lo < levels.max;) {
    const std::optional<valid_t> seed = seeds.at(lo - levels.min);
    int hi = lo + 1;
    while (hi < levels.max && seeds.at(hi - levels.min) == seed)
      ++hi;
    if (seed)
      runs.push_back({window.narrow(lo, hi), *seed});
    lo = hi;
  }
  return runs;
}

} // namespace

void CycleTimelevels(const window_t window, cGH *restrict const cctkGH) {
  DECLARE_CCTK_PARAMETERS;

  static Timer timer("CycleTimelevels");
  Interval interval(timer);

  update_cctkGHs(cctkGH);

  const GHExt &ext = driver();
  const int num_groups = CCTK_NumGroups();
  const bool presync_only = CCTK_EQUALS(presync_mode, "presync-only");
  std::vector<gi_t> presync_groups;
  for (int gi0 = 0; gi0 < num_groups; ++gi0) {
    const gi_t gi{gi0};
    const auto &group = ext.record(gi);
    const int ntls0 = group.num_timelevels();
    if (ntls0 == 0)
      continue;

    if (group.kind == group_kind::grid_function) {
      rotate_timelevels(window, gi);
      if (ntls0 > 1) {
        if (group.do_evolve) {
          invalidate(window, {.gi = gi,
                              .tl = tl_t{0},
                              .where = make_valid_all(),
                              .event = "CycleTimelevels"});
        } else {
          std::vector<write_scope_t> scopes;
          for (const vi_t vi : group.variables())
            for (const seed_run_t &run : seed_runs(window, gi, vi))
              scopes.push_back(open_write(
                  run.levels,
                  {.gi = gi,
                   .tl = tl_t{0},
                   .vi = vi,
                   .fills = run.seed,
                   .keeps = {},
                   .event = "CycleTimelevels (seed from old time level)"}));
          loop_serially(window, [&](auto &restrict leveldata) {
            auto &restrict groupdata = leveldata.group(gi);
            amrex::MultiFab::Copy(
                *groupdata.timelevel(tl_t{0}).mfab,
                *groupdata.timelevel(tl_t{1}).mfab, 0, 0,
                groupdata.record.numvars,
                groupdata.timelevel(tl_t{0}).mfab->nGrowVect());
          });
        }
      }
      if (group.do_evolve) {
        for (const tl_t tl : group.timelevels()) {
          if (ntls0 > 1 && tl == tl_t{0})
            continue;
          if (tl == tl_t{0} && presync_only)
            presync_groups.push_back(gi);
          else
            require(window, {.gi = gi,
                             .tl = tl,
                             .where = make_valid_all(),
                             .event = "CycleTimelevels for the state vector"});
        }
      }
    } else {
      if (ntls0 > 1) {
        rotate_array_timelevels(gi);
        invalidate_array({.gi = gi,
                          .tl = tl_t{0},
                          .where = make_valid_int(),
                          .event = "CycleTimelevels"});
      }
    }
  }
  if (!presync_groups.empty()) {
    SyncGroupsByDirI(window, presync_groups);
  }
}

} // namespace CarpetX
