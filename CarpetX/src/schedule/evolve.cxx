#include "schedule/schedule.hxx"
#include "schedule/clocks.hxx"
#include "hierarchy/driver.hxx"
#include "hierarchy/group_tags.hxx"
#include "hierarchy/loops.hxx"
#include "hierarchy/operations.hxx"
#include "io/io.hxx"
#include <loop.hxx>
#include "traversal/traversal.hxx"
#include <carpetx.hxx>

#include <cctk.h>
#include <cctk_Parameters.h>
#include <cctk_Schedule.h>
#include <cctki_GHExtensions.h>
#include <cctki_ScheduleBindings.h>
#include <cctki_WarnLevel.h>
#include <util_Table.h>

#include <AMReX_MultiFabUtil.H>

#include <sys/time.h>

#include <algorithm>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <vector>

namespace CarpetX {

#ifndef CCTK_HAVE_CGH_LEVEL
#error                                                                         \
    "The Cactus flesh does not support cctk_level in the cGH structure. Update the flesh."
#endif
#ifndef CCTK_HAVE_CGH_PATCH
#error                                                                         \
    "The Cactus flesh does not support cctk_patch in the cGH structure. Update the flesh."
#endif
#ifndef CCTK_HAVE_CGH_COMPONENT
#error                                                                         \
    "The Cactus flesh does not support cctk_component in the cGH structure. Update the flesh."
#endif
#ifndef CCTK_HAVE_CGH_TILE
#error                                                                         \
    "The Cactus flesh does not support cctk_tile_min etc. in the cGH structure. Update the flesh."
#endif

#if defined _OPENMP
#if !defined AMREX_USE_OMP
#error                                                                         \
    "Cactus is configured with OpenMP, but AMReX is configured without OpenMP. This does not work."
#endif
#endif

namespace {
double gettime() {
  timeval tv;
  gettimeofday(&tv, nullptr);
  return tv.tv_sec + tv.tv_usec / 1.0e+6;
}
} // namespace

////////////////////////////////////////////////////////////////////////////////

int Initialise(tFleshConfig *config) {
  DECLARE_CCTK_PARAMETERS;

  static Timer timer("Initialise");
  Interval interval(timer);

  cGH *restrict const cctkGH = CCTK_SetupGH(config, 0);
  CCTKi_AddGH(config, 0, cctkGH);

  // Check presync mode
  if (!CCTK_EQUALS(presync_mode, "mixed-error") &&
      !CCTK_EQUALS(presync_mode, "presync-only"))
    CCTK_ERROR("CarpetX currently requires Cactus::presync_mode = "
               "\"mixed-error\" or \"presync-only\"");

  // Check restrict_during_sync when use_subcycling is on
  if (use_subcycling && restrict_during_sync)
    CCTK_ERROR("CarpetX requires CarpetX::restrict_during_sync = "
               "\"no\" when CarpetX::use_subcycling = \"yes\"");

  // Check presync_mode when use_subcycling is on: in presync-only mode
  // CallFunction and CycleTimelevels sync through SyncGroupsByDirI, which
  // bypasses the subcycling sync (SyncGroupsByDirISubcycling), its time
  // interpolation and its end-of-step check
  if (use_subcycling && CCTK_EQUALS(presync_mode, "presync-only"))
    CCTK_ERROR("CarpetX requires Cactus::presync_mode = \"mixed-error\" when "
               "CarpetX::use_subcycling = \"yes\"");

  // Initialise iteration and time
  cctkGH->cctk_iteration = 0;
  cctkGH->cctk_time = *static_cast<const CCTK_REAL *>(
      CCTK_ParameterGet("cctk_initial_time", "Cactus", nullptr));

  // Initialise schedule
  CCTKi_ScheduleGHInit(cctkGH);

  const GHExt &ext = driver();
  freeze_storage();

  // Initialise all grid extensions
  CCTKi_InitGHExtensions(cctkGH);

  setup_cctkGH(cctkGH);
  enter_global_mode(cctkGH);
  set_global_cctkGH(cctkGH);

  for (const auto &patchdata : ext.patches())
    assert(patchdata.num_levels() == 0);
  {
    const traversal_scope_t scope(window_t::levels(0, 0));
    CCTK_Traverse(cctkGH, "CCTK_WRAGH");
    CCTK_Traverse(cctkGH, "CCTK_PARAMCHECK");
    CCTKi_FinaliseParamWarn();
    warn_inert_flux_tags();
  }

  CCTK_VINFO("Setting initial values for max_grid_size values for all levels");
  for (const auto &patchdata : ext.patches()) {
    amrex::Vector<amrex::IntVect> max_grid_sizes_vec(20);
    for (int level = 0; level < 20; ++level) {
      int size_x = max_grid_sizes_x[level];
      if (size_x == -1)
        size_x = max_grid_size_x;
      int size_y = max_grid_sizes_y[level];
      if (size_y == -1)
        size_y = max_grid_size_y;
      int size_z = max_grid_sizes_z[level];
      if (size_z == -1)
        size_z = max_grid_size_z;
      const amrex::IntVect max_grid_size_vec{size_x, size_y, size_z};
      max_grid_sizes_vec.at(level) = max_grid_size_vec;
    }
    set_max_grid_size(patchdata.patch, max_grid_sizes_vec);
  }

  if (config->recovered) {
    // Recovery runs its own sequence and skips the fresh-start tail below
    RecoverFromCheckpoint(cctkGH);
    return 0;

  } else {
    // Set up initial conditions
#pragma omp critical
    CCTK_VINFO("Setting up initial conditions...");

    {
      static Timer timer("InitialiseRegrid [coarse]");
      Interval interval(timer);

      const CCTK_REAL time = 0; // dummy time
      for (const auto &patchdata : ext.patches())
        make_new_grids(patchdata.patch, time);

      SetDeltaTime(cctkGH);
#pragma omp critical
      CCTK_VINFO("Iteration: %d   time: %g   delta_time: %g",
                 cctkGH->cctk_iteration, double(cctkGH->cctk_time),
                 double(cctkGH->cctk_delta_time));

      const traversal_scope_t scope(window_t::levels(0, 1));
      CCTK_Traverse(cctkGH, "CCTK_BASEGRID");
    }

    if (CCTK_MyProc(nullptr) == 0) {
      for (const auto &patchdata : ext.patches()) {
        const cGH *const patchGH =
            patchdata.level(level_t{patchdata.patch, 0}).get_patch_cctkGH();
        const int *restrict const gsh = patchGH->cctk_gsh;
        const int *restrict const nghostzones = patchGH->cctk_nghostzones;
        CCTK_REAL x0[dim], x1[dim], dx[dim];
        for (int d = 0; d < dim; ++d) {
          dx[d] = patchGH->cctk_delta_space[d];
          x0[d] = patchGH->cctk_origin_space[d] +
                  (2 * nghostzones[d] - 1) * dx[d] / 2;
          x1[d] = x0[d] + (gsh[d] - 2 * nghostzones[d] - 1) * dx[d];
        }
#pragma omp critical
        {
          CCTK_VINFO("Patch %d:", patchdata.patch.index());
          CCTK_VINFO("  Grid extent:");
          CCTK_VINFO("    gsh=[%d,%d,%d]", gsh[0], gsh[1], gsh[2]);
          const auto &bf = patchdata.amrcore->blockingFactor(0);
          CCTK_VINFO("    blocking_factor=[%d,%d,%d]", bf[0], bf[1], bf[2]);
          const auto &mgs = patchdata.amrcore->maxGridSize(0);
          CCTK_VINFO("    max_grid_size=[%d,%d,%d]", mgs[0], mgs[1], mgs[2]);
          const auto mfitinfo = amrex::MFItInfo().EnableTiling();
          const auto &ts = mfitinfo.tilesize;
          CCTK_VINFO("    max_tile_size=[%d,%d,%d]", ts[0], ts[1], ts[2]);
          CCTK_VINFO("  Domain extent:");
          CCTK_VINFO("    xmin=[%.17g,%.17g,%.17g]", double(x0[0]),
                     double(x0[1]), double(x0[2]));
          CCTK_VINFO("    xmax=[%.17g,%.17g,%.17g]", double(x1[0]),
                     double(x1[1]), double(x1[2]));
          CCTK_VINFO("    base dx=[%.17g,%.17g,%.17g]", double(dx[0]),
                     double(dx[1]), double(dx[2]));
        }
      }
    }

    for (const auto &patchdata : ext.patches())
      mark_initialized(patchdata.patch);

    for (;;) {
      const int level = ext.num_levels() - 1;
#pragma omp critical
      CCTK_VINFO("Initializing level %d...", level);

      for (const auto &patchdata : ext.patches())
        assert(patchdata.amrcore->finestLevel() <= level);

      {
        const traversal_scope_t scope(all_levels());
        InputGH(cctkGH);
        CCTK_Traverse(cctkGH, "CCTK_INITIAL");
        CCTK_Traverse(cctkGH, "CCTK_POSTINITIAL");
        CCTK_Traverse(cctkGH, "CCTK_POSTPOSTINITIAL");
      }

      bool did_modify_any_level;
      {
#pragma omp critical
        CCTK_VINFO("Regridding...");
        static Timer timer("InitialiseRegrid [refined]");
        Interval interval(timer);

        for (const auto &patchdata : ext.patches()) {

          const int old_numlevels = patchdata.amrcore->finestLevel() + 1;
          const CCTK_REAL time = 0; // dummy time
          regrid(level_t{patchdata.patch, 0}, time);

          const int new_numlevels = patchdata.amrcore->finestLevel() + 1;
          const int max_numlevels = patchdata.amrcore->maxLevel() + 1;
          assert(new_numlevels >= 0 && new_numlevels <= max_numlevels);
          assert(new_numlevels == old_numlevels ||
                 new_numlevels == old_numlevels + 1);

#pragma omp critical
          {
            const double pts0 = patchdata.level(level_t{patchdata.patch, 0})
                                    .fab->boxArray()
                                    .d_numPts();
            for (const auto &leveldata : patchdata.levels()) {
              const int sz = leveldata.fab->size();
              const double pts = leveldata.fab->boxArray().d_numPts();
              const int level = leveldata.level.index();
              if (level == 0) {
                CCTK_VINFO(
                    "  level %d: %d boxes, %.0f cells (%.4g%%)", level, sz, pts,
                    100 * pts / (ldexp(CCTK_REAL(1), dim * level) * pts0));
              } else {
                const double ptsc = patchdata.level(leveldata.level.parent())
                                        .fab->boxArray()
                                        .d_numPts();
                CCTK_VINFO("  level %d: %d boxes, %.0f cells (%.4g%%, %.0f%%)",
                           level, sz, pts,
                           100 * pts /
                               (ldexp(CCTK_REAL(1), dim * level) * pts0),
                           100 * pts / (ldexp(CCTK_REAL(1), dim) * ptsc));
              }
            }
          } // omp critical
        } // for patchdata

        int min_modified_level = INT_MAX;
        int max_modified_level = 0;
        for (const auto &patchdata : ext.patches()) {
          const std::vector<bool> &modified =
              patchdata.amrcore->modified_levels();
          for (int lev = 0; lev < int(modified.size()); ++lev) {
            if (modified.at(lev)) {
              using std::max, std::min;
              min_modified_level = min(min_modified_level, lev);
              max_modified_level = max(max_modified_level, lev + 1);
            }
          }
        }
        did_modify_any_level = min_modified_level < max_modified_level;

        if (did_modify_any_level) {
          SetDeltaTime(cctkGH);
#pragma omp critical
          CCTK_VINFO("Iteration: %d   time: %g   delta_time: %g",
                     cctkGH->cctk_iteration, double(cctkGH->cctk_time),
                     double(cctkGH->cctk_delta_time));

          const traversal_scope_t scope(
              window_t::levels(min_modified_level, max_modified_level));
          CCTK_Traverse(cctkGH, "CCTK_BASEGRID");
          CCTK_Traverse(cctkGH, "CCTK_POSTREGRID");
        }
      } // Regrid

      if (!did_modify_any_level)
        break;
    } // for level
  }
#pragma omp critical
  CCTK_VINFO("Initialized %d levels", ext.num_levels());

  const window_t window = FinestClockGroup();
  const traversal_scope_t scope(window);
  FinalizeWindow(window, cctkGH, false);

  // Checkpoint, analysis, output. The end-of-step bins are marked as in
  // Evolve; their window here holds every level, so a sync in them is legal
  // also under subcycling.
  TraverseTailBin(cctkGH, "CCTK_POSTSTEP");
  CCTK_Traverse(cctkGH, "CCTK_CPINITIAL");
  TraverseTailBin(cctkGH, "CCTK_ANALYSIS");
  if (out_initial_data) {
    CCTK_OutputGH(cctkGH);
  }

  return 0;
}

int Evolve(tFleshConfig *config) {
  DECLARE_CCTK_PARAMETERS;

  static Timer timer("Evolve");
  Interval interval(timer);

  assert(config);
  cGH *restrict const cctkGH = config->GH[0];
  assert(cctkGH);
  const GHExt &ext = driver();

#pragma omp critical
  CCTK_VINFO("Starting evolution...");

  double total_evolution_time = 0;
  double total_evolution_output_time = 0;
  int total_iterations = 0;
  double total_cell_updates = 0;
  int average_iteration_time_iterations = 0;
  double average_iteration_time = 0;

  std::ofstream performance_file;
  if (out_performance && CCTK_MyProc(NULL) == 0) {
    const int every =
        out_performance_every == -1 ? out_every : out_performance_every;
    if (every > 0) {
      std::ostringstream buf;
      buf << out_dir << "/performance.yaml";
      const std::string filename = buf.str();
      performance_file.open(filename);
      performance_file << "performance:\n" << std::flush;
    }
  }

  while (!EvolutionIsDone(cctkGH)) {

    const double start_time = gettime();

    rat64 iteration = ext.patches().front().levels().front().iteration;
    using std::min;
    for (const auto &patchdata : ext.patches())
      for (const auto &leveldata : patchdata.levels())
        iteration = min(iteration, leveldata.iteration);

    if (regrid_every > 0 && cctkGH->cctk_iteration % regrid_every == 0) {
#pragma omp critical
      CCTK_VINFO("Regridding...");
      static Timer timer("EvolveRegrid");
      Interval interval(timer);

      for (const auto &patchdata : ext.patches()) {

        int min_active_level = -1;
        for (int level = patchdata.num_levels() - 1; level >= 0; --level) {
          if (patchdata.level(level_t{patchdata.patch, level}).iteration !=
              iteration) {
            break;
          }
          min_active_level = level;
        }
        assert(min_active_level != -1);

        const int old_numlevels = patchdata.amrcore->finestLevel() + 1;
        const CCTK_REAL time = 0; // dummy time

        CCTK_VINFO(
            "Setting max_grid_size values for all levels before regridding");
        amrex::Vector<amrex::IntVect> max_grid_sizes_vec(20);
        for (int level = 0; level < 20; ++level) {
          int size_x = max_grid_sizes_x[level];
          if (size_x == -1)
            size_x = max_grid_size_x;
          int size_y = max_grid_sizes_y[level];
          if (size_y == -1)
            size_y = max_grid_size_y;
          int size_z = max_grid_sizes_z[level];
          if (size_z == -1)
            size_z = max_grid_size_z;
          const amrex::IntVect max_grid_size_vec{size_x, size_y, size_z};
          max_grid_sizes_vec.at(level) = max_grid_size_vec;
        }
        set_max_grid_size(patchdata.patch, max_grid_sizes_vec);

        regrid(level_t{patchdata.patch, min_active_level}, time);

        const int new_numlevels = patchdata.amrcore->finestLevel() + 1;
        const int max_numlevels = patchdata.amrcore->maxLevel() + 1;
        assert(new_numlevels >= 0 && new_numlevels <= max_numlevels);

#pragma omp critical
        {
          CCTK_VINFO("  old levels %d, new levels %d", old_numlevels,
                     new_numlevels);
          double pts0 = patchdata.level(level_t{patchdata.patch, 0})
                            .fab->boxArray()
                            .d_numPts();
          for (const auto &leveldata : patchdata.levels()) {
            const int sz = leveldata.fab->size();
            const double pts = leveldata.fab->boxArray().d_numPts();
            const int level = leveldata.level.index();
            if (level == 0) {
              CCTK_VINFO("  level %d: %d boxes, %.0f cells (%.4g%%)", level, sz,
                         pts,
                         100 * pts / (ldexp(CCTK_REAL(1), dim * level) * pts0));
            } else {
              const double ptsc = patchdata.level(leveldata.level.parent())
                                      .fab->boxArray()
                                      .d_numPts();
              CCTK_VINFO("  level %d: %d boxes, %.0f cells (%.4g%%, %.0f%%)",
                         level, sz, pts,
                         100 * pts / (ldexp(CCTK_REAL(1), dim * level) * pts0),
                         100 * pts / (ldexp(CCTK_REAL(1), dim) * ptsc));
            }
          }
        } // omp critical
      } // for patchdata

      int min_modified_level = INT_MAX;
      int max_modified_level = 0;
      for (const auto &patchdata : ext.patches()) {
        const std::vector<bool> &modified =
            patchdata.amrcore->modified_levels();
        for (int lev = 0; lev < int(modified.size()); ++lev) {
          if (modified.at(lev)) {
            using std::max, std::min;
            min_modified_level = min(min_modified_level, lev);
            max_modified_level = max(max_modified_level, lev + 1);
          }
        }
      }
      const bool did_modify_any_level = min_modified_level < max_modified_level;

      if (did_modify_any_level) {
        SetDeltaTime(cctkGH);
#pragma omp critical
        CCTK_VINFO("Iteration: %d   time: %g   delta_time: %g",
                   cctkGH->cctk_iteration, double(cctkGH->cctk_time),
                   double(cctkGH->cctk_delta_time));

        const traversal_scope_t scope(
            window_t::levels(min_modified_level, max_modified_level));
        CCTK_Traverse(cctkGH, "CCTK_BASEGRID");
        CCTK_Traverse(cctkGH, "CCTK_POSTREGRID");
      }
    } // Regrid

    rat64 delta_iteration =
        ext.patches().front().levels().front().delta_iteration;
    for (const auto &patchdata : ext.patches())
      for (const auto &leveldata : patchdata.levels())
        delta_iteration = min(delta_iteration, leveldata.delta_iteration);

    cctkGH->cctk_iteration += 1;

    for (int min_level = 0, max_level; min_level < ext.num_levels();
         min_level = max_level) {
      max_level = min_level;
      while (++max_level < ext.num_levels()) {
        bool level_is_subcycling_level = false;
        for (const auto &patchdata : ext.patches())
          if (max_level < patchdata.num_levels())
            level_is_subcycling_level |=
                patchdata.level(level_t{patchdata.patch, max_level})
                    .is_subcycling_level;
        if (level_is_subcycling_level)
          break;
      }

      rat64 level_iteration = -1;
      rat64 level_delta_iteration = -1;
      for (const auto &patchdata : ext.patches()) {
        if (min_level < patchdata.num_levels()) {
          const auto &leveldata =
              patchdata.level(level_t{patchdata.patch, min_level});
          level_iteration = leveldata.iteration;
          level_delta_iteration = leveldata.delta_iteration;
          break;
        }
      }
      assert(level_iteration != -1);
      assert(level_delta_iteration != -1);
      // Skip those evolved coarse levels while evolving the fine levels to
      // catch up
      if (level_iteration > iteration)
        continue;

      const window_t window = window_t::levels(min_level, max_level);
      {
        const traversal_scope_t scope(window);

        level_iteration += level_delta_iteration;
        advance_clocks(window);
        loop_serially(window, [&](const auto &leveldata) {
          assert(level_iteration == leveldata.iteration);
          assert(level_delta_iteration == leveldata.delta_iteration);
        });

        CycleTimelevels(window, cctkGH);

        if (!ext.use_subcycling)
          cctkGH->cctk_time += cctkGH->cctk_delta_time;
        SetStepTime(cctkGH, window.coarsest(), level_iteration);

        CCTK_Traverse(cctkGH, "CCTK_PRESTEP");
        CCTK_Traverse(cctkGH, "CCTK_EVOL");
      }

      const window_t widened = window_t::levels(
          WidenMinLevel(window.coarsest(), level_iteration).index(), max_level);
      stamp_last_step(widened, cctkGH->cctk_iteration);
    } // for min_level, max_level

    {
      const window_t window = FinestClockGroup();
      const traversal_scope_t scope(window);
      FinalizeWindow(window, cctkGH, ext.use_subcycling);
      TraverseTailBin(cctkGH, "CCTK_POSTSTEP");
      TraverseTailBin(cctkGH, "CCTK_ANALYSIS");

      stamp_last_finalized(window, cctkGH->cctk_iteration);

      TraverseTailBin(cctkGH, "CCTK_CHECKPOINT");
      const double output_start_time = gettime();
      CCTK_OutputGH(cctkGH);
      const double output_finish_time = gettime();
      total_evolution_output_time += output_finish_time - output_start_time;
    }

    const double waiting_start_time = gettime();
    MPI_Barrier(MPI_COMM_WORLD);
    const double waiting_finish_time = gettime();

    const double finish_time = gettime();
    double num_cells = 0;
    for (const auto &patch : ext.patches())
      for (const auto &level : patch.levels())
        num_cells += level.fab->boxArray().d_numPts();
    total_cell_updates += num_cells;
    ++total_iterations;
    const double iteration_time = finish_time - start_time;
    total_evolution_time += iteration_time;

    // Calculate statistics
    if (average_iteration_time_iterations < 10)
      ++average_iteration_time_iterations;
    // Calculate exponential moving average
    average_iteration_time =
        ((average_iteration_time_iterations - 1) * average_iteration_time +
         iteration_time) /
        average_iteration_time_iterations;

    const double iterations_per_second = 1 / average_iteration_time;
    const double cell_updates_per_second = num_cells * iterations_per_second;
    CCTK_VINFO("Simulation time: %g   "
               "Iterations per second: %g   "
               "Simulation time per second: %g",
               double(cctkGH->cctk_time), iterations_per_second,
               double(cctkGH->cctk_delta_time * iterations_per_second)

    );
    // This is the same as H-AMR's "cell updates per second":
    CCTK_VINFO("Grid cells: %g   "
               "Grid cell updates per second: %g",
               num_cells, cell_updates_per_second);

    const double total_evolution_compute_time =
        total_evolution_time - total_evolution_output_time;
    CCTK_VINFO("Performance:");
    CCTK_VINFO("  total evolution time:            %g sec",
               total_evolution_time);
    CCTK_VINFO("  total evolution compute time:    %g sec",
               total_evolution_compute_time);
    CCTK_VINFO("  total evolution output time:     %g sec",
               total_evolution_output_time);
    CCTK_VINFO("  total iterations:                %d", total_iterations);
    CCTK_VINFO("  total cells updated:             %g", total_cell_updates);
    CCTK_VINFO("  average iterations per second: %g",
               total_iterations / total_evolution_time);
    CCTK_VINFO("  average cell updates per second: %g",
               total_cell_updates / total_evolution_time);
    // TODO: Output this in a proper I/O method
    if (out_performance && CCTK_MyProc(NULL) == 0) {
      const int every =
          out_performance_every == -1 ? out_every : out_performance_every;
      if (every > 0 && cctkGH->cctk_iteration % every == 0)
        performance_file << "  " << total_iterations << ":\n"
                         << "    evolution-seconds: " << total_evolution_time
                         << "\n"
                         << "    evolution-compute-seconds: "
                         << total_evolution_compute_time << "\n"
                         << "    evolution-output-seconds: "
                         << total_evolution_output_time << "\n"
                         << "    evolution-cell-updates: " << total_cell_updates
                         << "\n"
                         << "    evolution-iterations: " << total_iterations
                         << "\n"
                         << std::flush;
    }

  } // main loop

  if (out_performance && CCTK_MyProc(NULL) == 0)
    performance_file.close();

  return 0;
}

int Shutdown(tFleshConfig *config) {
  assert(config);
  cGH *restrict const cctkGH = config->GH[0];
  assert(cctkGH);

  static Timer timer("Shutdown");
  Interval interval(timer);

#pragma omp critical
  CCTK_VINFO("Shutting down...");

  {
    const traversal_scope_t scope(FinestClockGroup());
    TraverseTailBin(cctkGH, "CCTK_TERMINATE");
  }
  {
    const traversal_scope_t scope(window_t::levels(0, 0));
    CCTK_Traverse(cctkGH, "CCTK_SHUTDOWN");
  }
  assert_hierarchy_destroyed();

  return 0;
}

} // namespace CarpetX
