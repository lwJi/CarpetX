#include "hierarchy/loops.hxx"
#include "hierarchy/modes.hxx"
#include "config/stages.hxx"
#include "hierarchy/model.hxx"
#include "comm/fillpatch.hxx"
#include "comm/ghost_prolongation.hxx"
#include "comm/integrator.hxx"
#include "comm/outer_boundary.hxx"
#include "comm/restrict.hxx"
#include "comm/substeps.hxx"
#include "comm/sync.hxx"
#include "comm/group_selection.hxx"
#include "comm/level_runs.hxx"
#include "comm/timelevels.hxx"
#include "traversal/traversal.hxx"
#include "util/refuse.hxx"
#include "util/task_manager.hxx"
#include <carpetx.hxx>
#include "hierarchy/scopes.hxx"

#include <cctk.h>
#include <cctk_Parameters.h>

#include <algorithm>
#include <cassert>
#include <optional>
#include <source_location>
#include <span>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace CarpetX {

static std::vector<gi_t> sync_restrict_groups(const std::vector<gi_t> &groups) {
  std::vector<gi_t> restrict_groups;
  restrict_groups.reserve(groups.size());
  for (const gi_t gi : groups)
    if (driver().record(gi).restrict_on_sync)
      restrict_groups.push_back(gi);
  return restrict_groups;
}

// =======================================================================
// Sync entry points
// =======================================================================

static void SyncGroupsLockstep(const window_t window,
                               const std::span<const gi_t> groups0) {
  DECLARE_CCTK_PARAMETERS;

  const GHExt &ext = driver();

  error_if_sync_nested();

  static Timer timer("Sync");
  Interval interval(timer);

  const char *const label = "SyncGroupsLockstep";
  const std::string sync_label = std::string(label) + " before syncing";
  const std::string prolongation_label =
      std::string(label) + " on coarse level before prolongation";
  const std::string boundary_label =
      std::string(label) + " applies the boundary condition after the ghost "
                           "fill and reads the interior and both ghost regions";

  sync_log_groups(label, groups0);

  std::vector<gi_t> groups = sync_filter_groups(groups0);

  if (CCTK_EQUALS(presync_mode, "presync-only")) {
    loop_serially(window, [&](auto &restrict leveldata) {
      std::vector<gi_t> new_groups;
      for (const gi_t gi : groups) {
        auto &restrict groupdata = leveldata.group(gi);
        bool need_sync = false;
        for (const tl_t tl : groupdata.record.timelevels()) {
          if (need_sync)
            break;
          for (const vi_t vi : groupdata.record.variables()) {
            if (need_sync)
              break;
            const valid_t have = groupdata.timelevel(tl).valid(vi);
            if ((~have & (make_valid_ghosts() | make_valid_outer()))
                    .valid_any()) {
              need_sync = true;
            }
          }
        }
        if (need_sync) {
          new_groups.push_back(gi);
        }
      }
      groups = new_groups;
    });
    if (groups.size() == 0) {
      return;
    }
  }

  if (restrict_during_sync) {
    const std::vector<gi_t> restrict_groups = sync_restrict_groups(groups);
    loop_fine_to_coarse(
        window, [&](const GHExt::PatchData::LevelData &leveldata) {
          assert(do_restrict || !leveldata.level.child());
          restrict_level(leveldata.level, restrict_groups, make_valid_int());
        });
  }

  write_scopes_t scopes;
  for (const gi_t gi : groups) {
    const tl_range sync_tls =
        timelevels_t::all_but_oldest().range(ext.record(gi));
    for (const tl_t tl : sync_tls) {
      require(
          window,
          {.gi = gi, .tl = tl, .where = make_valid_int(), .event = sync_label});
      scopes.open(window, {.gi = gi,
                           .tl = tl,
                           .fills = make_valid_ghosts(),
                           .keeps = make_valid_int() | make_valid_outer(),
                           .event = label});
    }
  }

  task_manager tasks1;
  task_manager tasks2;
  task_manager tasks3;

  for (const gi_t gi : groups) {
    loop_serially(window, [&](auto &restrict leveldata) {
      auto &restrict groupdata = leveldata.group(gi);

      const tl_range sync_tls =
          timelevels_t::all_but_oldest().range(groupdata.record);

      if (leveldata.level.index() == 0) {

        for (const tl_t tl : sync_tls) {
          tasks1.submit_serially(
              [&tasks2, &leveldata, &scopes, gi, tl, &ext]() {
                FillPatch_Sync(tasks2, scopes.mfab(leveldata.level, gi, tl),
                               ext.patch(leveldata.level.patch())
                                   .amrcore()
                                   .Geom(leveldata.level.index()));
              });
        }

      } else {

        const auto &restrict coarseleveldata =
            ext.level(leveldata.level.parent());
        auto &restrict coarsegroupdata = coarseleveldata.group(gi);
        assert(coarsegroupdata.record.numvars == groupdata.record.numvars);

        for (const tl_t tl : sync_tls) {
          error_if_parent_invalid(leveldata, gi, tl,
                                  prolongation_label.c_str());
          tasks1.submit_serially([&tasks2, &tasks3, &leveldata, &groupdata,
                                  &coarsegroupdata, &scopes, gi, tl, &ext]() {
            FillPatch_ProlongateGhosts(
                tasks2, tasks3, groupdata, coarsegroupdata,
                scopes.mfab(leveldata.level, gi, tl),
                coarsegroupdata.timelevel(tl).mfab(),
                ext.patch(leveldata.level.patch())
                    .amrcore()
                    .Geom(leveldata.level.index()),
                ext.patch(leveldata.level.patch())
                    .amrcore()
                    .Geom(leveldata.level.parent().index()));
          });
        }
      }
    });
  }

  tasks1.run_tasks_serially();
  synchronize();
  tasks2.run_tasks_serially();
  synchronize();
  tasks3.run_tasks_serially();
  synchronize();

  scopes.close();
  apply_outer_boundary(window, groups, timelevels_t::all_but_oldest(),
                       boundary_label.c_str());
}

static void error_if_sync_in_tail(const window_t window,
                                  const std::span<const gi_t> groups0) {
  const char *const bin = scheduled_tail_bin();
  if (!bin || groups0.empty())
    return;
  const auto [min_level, max_level] = window.level_bounds();
  if (min_level == 0)
    return;
  std::ostringstream groupnames;
  for (std::size_t n = 0; n < groups0.size(); ++n) {
    const char *const groupname = CCTK_FullGroupName(groups0[n].index());
    if (n != 0)
      groupnames << ", ";
    groupnames << (groupname ? groupname : "(unknown group)");
  }
  const cFunctionData *const func = scheduled_routine();
  CCTK_VERROR(
      "%s::%s (%s) requested SYNC of %s at iteration %d with "
      "CarpetX::use_subcycling = yes, while the end-of-step window holds only "
      "levels [%d,%d). The sync would fill the coarse-fine ghosts of level %d "
      "from level %d, which has stepped ahead and has not been finalized. "
      "Under subcycling CCTK_POSTSTEP, CCTK_ANALYSIS, CCTK_CHECKPOINT and "
      "CCTK_TERMINATE may sync only when the end-of-step window holds every "
      "level. Otherwise remove the SYNC: write pointwise outputs everywhere; "
      "outputs that need a stencil stay interior and their ghosts are "
      "invalid.",
      func ? func->thorn : "(unknown thorn)",
      func ? func->routine : "(unknown routine)", bin, groupnames.str().c_str(),
      current_iteration(), min_level, max_level, min_level, min_level - 1);
}

static void SyncGroupsSubcycling(const window_t window,
                                 const std::span<const gi_t> groups0) {
  DECLARE_CCTK_PARAMETERS;

  const GHExt &ext = driver();

  error_if_sync_in_tail(window, groups0);

  error_if_sync_nested();

  static Timer timer("Sync");
  Interval interval(timer);

  const char *const label = "SyncGroupsSubcycling";
  const std::string sync_label = std::string(label) + " before syncing";
  const std::string prolongation_label =
      std::string(label) + " on coarse level before prolongation";
  const std::string boundary_label =
      std::string(label) + " applies the boundary condition after the ghost "
                           "fill and reads the interior and both ghost regions";

  sync_log_groups(label, groups0);

  std::vector<gi_t> groups = sync_filter_groups(groups0);

  write_scopes_t scopes;
  for (const gi_t gi : groups) {
    const tl_range sync_tls =
        timelevels_t::all_but_oldest().range(ext.record(gi));
    const std::vector<level_run_t> runs = split_levels(
        window, [gi](const GHExt::PatchData::LevelData &leveldata) {
          return integrator_owns_coarse_fine_ghosts(leveldata, gi);
        });
    for (const level_run_t &run : runs) {
      const valid_t fills =
          run.value ? make_valid_same_level() : make_valid_ghosts();
      const valid_t keeps = run.value ? ~make_valid_same_level()
                                      : make_valid_int() | make_valid_outer();
      for (const tl_t tl : sync_tls) {
        scopes.open(run.levels, {.gi = gi,
                                 .tl = tl,
                                 .fills = fills,
                                 .keeps = keeps,
                                 .event = label});
      }
    }
  }

  task_manager tasks1;
  task_manager tasks2;
  task_manager tasks3;

  for (const gi_t gi : groups) {
    loop_serially(window, [&](auto &restrict leveldata) {
      auto &restrict groupdata = leveldata.group(gi);

      const tl_range sync_tls =
          timelevels_t::all_but_oldest().range(groupdata.record);

      if (leveldata.level.index() == 0) {
        for (const tl_t tl : sync_tls) {
          require(window_t::one_level(leveldata.level),
                  {.gi = gi,
                   .tl = tl,
                   .where = make_valid_int(),
                   .event = sync_label});
          tasks1.submit_serially(
              [&tasks2, &leveldata, &scopes, gi, tl, &ext]() {
                FillPatch_Sync(tasks2, scopes.mfab(leveldata.level, gi, tl),
                               ext.patch(leveldata.level.patch())
                                   .amrcore()
                                   .Geom(leveldata.level.index()));
              });
        }

      } else {

        const auto &restrict coarseleveldata =
            ext.level(leveldata.level.parent());
        auto &restrict coarsegroupdata = coarseleveldata.group(gi);
        assert(coarsegroupdata.record.numvars == groupdata.record.numvars);

        const bool integrated_subiter =
            integrator_owns_coarse_fine_ghosts(leveldata, gi);

        if (integrated_subiter) {
          for (const tl_t tl : sync_tls) {
            require(window_t::one_level(leveldata.level),
                    {.gi = gi,
                     .tl = tl,
                     .where = make_valid_int(),
                     .event = sync_label});
            tasks1.submit_serially(
                [&tasks2, &leveldata, &scopes, gi, tl, &ext]() {
                  FillPatch_Sync(tasks2, scopes.mfab(leveldata.level, gi, tl),
                                 ext.patch(leveldata.level.patch())
                                     .amrcore()
                                     .Geom(leveldata.level.index()));
                });
          }
        } else {
          const substep_t substep = substep_of_parent_step(leveldata.level);
          const bool blend =
              !substep.aligned() && coarsegroupdata.num_timelevels() >= 2;
          const CCTK_REAL w_new = CCTK_REAL(substep.end);

          for (const tl_t tl : sync_tls) {
            const bool do_blend = blend && tl == tl_t{0};
            require(window_t::one_level(leveldata.level),
                    {.gi = gi,
                     .tl = tl,
                     .where = make_valid_int(),
                     .event = sync_label});
            error_if_parent_invalid(leveldata, gi, tl,
                                    prolongation_label.c_str());
            if (do_blend)
              error_if_parent_invalid(leveldata, gi, tl_t{1},
                                      prolongation_label.c_str());
            const amrex::MultiFab *const cmfab_old =
                do_blend ? &coarsegroupdata.timelevel(tl_t{1}).mfab() : nullptr;
            const CCTK_REAL tl_w_new = do_blend ? w_new : CCTK_REAL(1);
            tasks1.submit_serially([&tasks2, &tasks3, &leveldata, &groupdata,
                                    &coarsegroupdata, &scopes, gi, tl,
                                    cmfab_old, tl_w_new, &ext]() {
              FillPatch_ProlongateGhosts(
                  tasks2, tasks3, groupdata, coarsegroupdata,
                  scopes.mfab(leveldata.level, gi, tl),
                  coarsegroupdata.timelevel(tl).mfab(),
                  ext.patch(leveldata.level.patch())
                      .amrcore()
                      .Geom(leveldata.level.index()),
                  ext.patch(leveldata.level.patch())
                      .amrcore()
                      .Geom(leveldata.level.parent().index()),
                  cmfab_old, tl_w_new);
            });
          }
        }
      }
    });
  }

  tasks1.run_tasks_serially();
  synchronize();
  tasks2.run_tasks_serially();
  synchronize();
  tasks3.run_tasks_serially();
  synchronize();

  scopes.close();
  apply_outer_boundary(window, groups, timelevels_t::all_but_oldest(),
                       boundary_label.c_str());
}

void SyncGroups(const window_t window, const std::span<const gi_t> groups) {
  if (driver().use_subcycling)
    SyncGroupsSubcycling(window, groups);
  else
    SyncGroupsLockstep(window, groups);
}

namespace {
std::string sync_caller() {
  const cFunctionData *const func = scheduled_routine();
  if (!func)
    return "SYNC";
  return std::string("SYNC from ") + func->thorn + "::" + func->routine;
}
} // namespace

int SyncGroupsOnActiveLevels(const cGH *restrict const cctkGH,
                             const int numgroups, const int *const groups,
                             const int *const directions) {
  assert(in_global_mode(cctkGH) || in_level_mode(cctkGH));
  const std::vector<gi_t> gis(groups, groups + numgroups);
  SyncGroups(scheduled_window(sync_caller().c_str()), gis);
  return numgroups;
}

void SyncGhostsOnly(const window_t window, const std::span<const gi_t> groups0,
                    const timelevels_t tls) {
  DECLARE_CCTK_PARAMETERS;

  const GHExt &ext = driver();

  error_if_sync_nested();

  static Timer timer("Sync");
  Interval interval(timer);

  const char *const label = "SyncGhostsOnly";
  const std::string sync_label = std::string(label) + " before syncing";

  sync_log_groups(label, groups0);

  std::vector<gi_t> groups = sync_filter_groups(groups0);

  write_scopes_t scopes;
  for (const gi_t gi : groups) {
    for (const tl_t tl : tls.range(ext.record(gi))) {
      require(
          window,
          {.gi = gi, .tl = tl, .where = make_valid_int(), .event = sync_label});
      scopes.open(window, {.gi = gi,
                           .tl = tl,
                           .fills = make_valid_same_level(),
                           .keeps = ~make_valid_same_level(),
                           .event = label});
    }
  }

  task_manager tasks1;
  task_manager tasks2;

  for (const gi_t gi : groups) {
    loop_serially(window, [&](auto &restrict leveldata) {
      auto &restrict groupdata = leveldata.group(gi);

      for (const tl_t tl : tls.range(groupdata.record)) {
        tasks1.submit_serially([&tasks2, &leveldata, &scopes, gi, tl, &ext]() {
          FillPatch_Sync(tasks2, scopes.mfab(leveldata.level, gi, tl),
                         ext.patch(leveldata.level.patch())
                             .amrcore()
                             .Geom(leveldata.level.index()));
        });
      }
    });
  }

  tasks1.run_tasks_serially();
  synchronize();
  tasks2.run_tasks_serially();
  synchronize();

  scopes.close();
}

} // namespace CarpetX
