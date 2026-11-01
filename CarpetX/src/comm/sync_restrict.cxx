#include "hierarchy/loops.hxx"
#include "hierarchy/driver.hxx"
#include "hierarchy/group_tags.hxx"
#include "comm/fillpatch.hxx"
#include "comm/level_pairs.hxx"
#include "comm/comm.hxx"
#include "traversal/traversal.hxx"
#include "util/task_manager.hxx"
#include <carpetx.hxx>
#include "hierarchy/valid.hxx"

#include <cctk.h>
#include <cctk_Parameters.h>

#include <AMReX_MultiFabUtil.H>

#include <algorithm>
#include <cassert>
#include <functional>
#include <optional>
#include <span>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace CarpetX {

// =======================================================================
// Sync helpers
// =======================================================================

bool sync_active = false; // Catch recursive calls

struct mark_sync_active {
  mark_sync_active() {
    if (sync_active)
      CCTK_ERROR(
          "Recursive call to SyncGroupsByDirI. Maybe you are syncing grid "
          "functions in the \"restrict\" bin while the parameter "
          "\"restrict_during_sync\" is true?");
    sync_active = true;
  }
  ~mark_sync_active() { sync_active = false; }
};

static void sync_log_groups(const char *label,
                            const std::span<const gi_t> groups0) {
  DECLARE_CCTK_PARAMETERS;
  if (verbose) {
    std::ostringstream buf;
    for (std::size_t n = 0; n < groups0.size(); ++n) {
      if (n != 0)
        buf << ", ";
      buf << CCTK_FullGroupName(groups0[n].index());
    }
#pragma omp critical
    CCTK_VINFO("%s %s", label, buf.str().c_str());
  }
}

static std::vector<gi_t> sync_filter_groups(const std::span<const gi_t> groups0) {
  static const gi_t gi_regrid_error{
      CCTK_GroupIndex("CarpetXRegrid::regrid_error")};
  assert(gi_regrid_error.index() >= 0);

  std::vector<gi_t> groups;
  for (const gi_t gi : groups0) {
    const auto &group = driver().record(gi);
    if (!group.has_storage() || group.kind != group_kind::grid_function)
      continue;
    // Don't restrict the regridding error
    if (gi == gi_regrid_error)
      continue;
    groups.push_back(gi);
  }
  return groups;
}

static void sync_multipatch_boundaries(const std::vector<gi_t> &groups) {
  const GHExt &ext = driver();
  static const bool have_multipatch_boundaries =
      CCTK_IsFunctionAliased("MultiPatch_Interpolate");

  if (have_multipatch_boundaries) {
    std::vector<CCTK_INT> cactusvarinds;
    for (const gi_t gi : groups) {
      const auto &group = ext.record(gi);
      for (const vi_t vi : group.variables())
        cactusvarinds.push_back(group.var(vi).index());
    }
    MultiPatch_Interpolate(ext.get_global_cctkGH(), cactusvarinds.size(),
                           cactusvarinds.data());
  } else {
    assert(ext.num_patches() == 1);
  }
}

tl_range timelevels_t::range(const GHExt::GroupRecord &record,
                             const std::source_location site) const {
  const int ntls = record.num_timelevels();
  if (!tl_)
    return tl_range{tl_t{0}, tl_t{ntls > 1 ? ntls - 1 : ntls}};
  if (!record.timelevels().contains(*tl_))
    refuse(site,
           "time level %d of group %s does not exist: it has %d time levels",
           tl_->index(), record.groupname.c_str(), ntls);
  return tl_range{*tl_, tl_->next()};
}

static valid_t sync_fills(const window_t window, const gi_t gi) {
  std::optional<bool> covered;
  loop_serially(window, [&](auto &restrict leveldata) {
    const bool level_covered =
        leveldata.group(gi).all_faces_have_symmetries_or_boundaries();
    assert(!covered || *covered == level_covered);
    covered = level_covered;
  });
  return make_valid_ghosts() |
         (covered.value_or(false) ? make_valid_outer() : valid_t());
}

struct level_run_t {
  window_t levels;
  bool value;
};

static std::vector<level_run_t> split_levels(
    const window_t window,
    const std::function<bool(const GHExt::PatchData::LevelData &)> &predicate) {
  const auto levels = window.level_bounds();
  std::vector<std::optional<bool> > values(levels.max - levels.min);
  loop_serially(window, [&](auto &restrict leveldata) {
    std::optional<bool> &value =
        values.at(leveldata.level.index() - levels.min);
    const bool level_value = predicate(leveldata);
    assert(!value || *value == level_value);
    value = level_value;
  });
  std::vector<level_run_t> runs;
  for (int lo = levels.min; lo < levels.max;) {
    const std::optional<bool> value = values.at(lo - levels.min);
    int hi = lo + 1;
    while (hi < levels.max && values.at(hi - levels.min) == value)
      ++hi;
    if (value)
      runs.push_back({window.narrow(lo, hi), *value});
    lo = hi;
  }
  return runs;
}

static void
error_if_parent_invalid(const GHExt::PatchData::LevelData &leveldata,
                        const gi_t gi, const tl_t tl, const char *const label) {
  const auto &restrict coarseleveldata =
      driver().level(leveldata.level.parent());
  const auto &restrict coarsegroupdata = coarseleveldata.group(gi);
  const bool recovering =
      recovered_iteration() >= 0 &&
      coarseleveldata.last_finalized_iteration <= recovered_iteration();

  const std::function<std::string()> hint = [&coarsegroupdata, tl,
                                             recovering]() -> std::string {
    if (coarsegroupdata.record.do_checkpoint)
      return "";
    if (tl > tl_t{0})
      return "  Group " + coarsegroupdata.record.groupname +
             " is not checkpointed, so a recovery, or a regrid that remakes "
             "its level, can lose its past timelevels, which no routine can "
             "recompute.\n"
             "  Past timelevels read by another level must be checkpointed.\n";
    if (recovering)
      return "  This run was recovered and group " +
             coarsegroupdata.record.groupname +
             " is not checkpointed.\n"
             "  Groups read by another level must be checkpointed or "
             "recomputed in CCTK_POST_RECOVER_VARIABLES.\n";
    return "";
  };
  require(window_t::one_level(coarseleveldata.level),
          {.gi = gi,
           .tl = tl,
           .where = make_valid_int(),
           .event = label,
           .hint = hint});
}

// =======================================================================
// Restriction group selectors
// =======================================================================

std::vector<gi_t> collect_restrictable_groups() {
  std::vector<gi_t> groups;
  groups.reserve(CCTK_NumGroups());
  for (const auto &group : driver().records(group_kind::grid_function))
    if (group.do_restrict)
      groups.push_back(group.groupindex);
  return groups;
}

static std::vector<gi_t> reprolongate_groups() {
  const GHExt &ext = driver();
  std::vector<gi_t> groups;
  groups.reserve(CCTK_NumGroups());
  for (const auto &group : ext.records(group_kind::grid_function))
    if (group.do_restrict ||
        (ext.use_subcycling && group_is_integrated(group.groupindex)))
      groups.push_back(group.groupindex);
  return groups;
}

static std::vector<gi_t> sync_restrict_groups(const std::vector<gi_t> &groups) {
  std::vector<gi_t> restrict_groups;
  restrict_groups.reserve(groups.size());
  for (const gi_t gi : groups)
    if (driver().record(gi).restrict_on_sync)
      restrict_groups.push_back(gi);
  return restrict_groups;
}

static void restrict_level(level_t level, std::span<const gi_t> groups,
                           valid_t coarse_keeps);

// =======================================================================
// Sync entry points
// =======================================================================

static void SyncGroupsByDirIProlongateOnly_impl(
    const window_t window, const std::span<const gi_t> groups0,
    const bool prolongate_on_same_iteration, const timelevels_t tls) {
  DECLARE_CCTK_PARAMETERS;

  const GHExt &ext = driver();

  mark_sync_active marked;

  static Timer timer("Sync");
  Interval interval(timer);

  sync_log_groups("SyncGroupsProlongateOnly", groups0);

  std::vector<gi_t> groups = sync_filter_groups(groups0);

  const auto prolongates_into =
      [&ext, prolongate_on_same_iteration](
          const GHExt::PatchData::LevelData &leveldata) {
        if (leveldata.level.index() == 0)
          return false;
        if (ext.use_subcycling && prolongate_on_same_iteration) {
          const auto &restrict coarseleveldata =
              ext.level(leveldata.level.parent());
          return leveldata.iteration == coarseleveldata.iteration;
        }
        return true;
      };
  const std::vector<level_run_t> runs = split_levels(window, prolongates_into);
  std::vector<write_scope_t> scopes;
  for (const gi_t gi : groups) {
    const tl_range fill_tls = tls.range(ext.record(gi));
    for (const level_run_t &run : runs) {
      if (!run.value)
        continue;
      const valid_t fills = sync_fills(run.levels, gi);
      for (const tl_t tl : fill_tls) {
        scopes.push_back(open_write(
            run.levels, {.gi = gi,
                         .tl = tl,
                         .fills = fills,
                         .keeps = make_valid_all(),
                         .event = "SyncGroupsByDirIProlongateOnly"}));
      }
    }
  }

  // We need to loop over groups, patches, and levels in a definite
  // order so that AMReX's communication pattern does not get
  // confused. Therefore all the loops here are serial. The only
  // parallelization happens within AMReX and within our boundary
  // conditions. This is not efficient.

  task_manager tasks1;
  task_manager tasks2;
  task_manager tasks3;

  for (const gi_t gi : groups) {
    loop_serially(window, [&](auto &restrict leveldata) {
      auto &restrict groupdata = leveldata.group(gi);

      const tl_range fill_tls = tls.range(groupdata.record);

      if (!prolongates_into(leveldata))
        return;

      const auto &restrict coarseleveldata =
          ext.level(leveldata.level.parent());

      auto &restrict coarsegroupdata = coarseleveldata.group(gi);
      assert(coarsegroupdata.record.numvars == groupdata.record.numvars);

      amrex::Interpolater *const interpolator = groupdata.interpolator;

      for (const tl_t tl : fill_tls) {
        error_if_parent_invalid(leveldata, gi, tl,
                                "SyncGroupsByDirIProlongateOnly on coarse "
                                "level before prolongation");
        tasks1.submit_serially([&tasks2, &tasks3, &leveldata, &groupdata,
                                &coarsegroupdata, interpolator, tl, &ext]() {
          FillPatch_ProlongateOnly(
              tasks2, tasks3, groupdata, coarsegroupdata,
              *groupdata.timelevel(tl).mfab,
              *coarsegroupdata.timelevel(tl).mfab,
              ext.patch(leveldata.level.patch())
                  .amrcore->Geom(leveldata.level.index()),
              ext.patch(leveldata.level.patch())
                  .amrcore->Geom(leveldata.level.parent().index()),
              interpolator, groupdata.bcrecs);
        });

      } // for tl
    });
  } // for gi

  tasks1.run_tasks_serially();
  synchronize();
  tasks2.run_tasks_serially();
  synchronize();
  tasks3.run_tasks_serially();
  synchronize();

  sync_multipatch_boundaries(groups);
  scopes.clear();

  assert(sync_active);
}

void SyncGroupsByDirI(const window_t window,
                      const std::span<const gi_t> groups0) {
  DECLARE_CCTK_PARAMETERS;

  const GHExt &ext = driver();

  mark_sync_active marked;

  static Timer timer("Sync");
  Interval interval(timer);

  sync_log_groups("SyncGroups", groups0);

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
            if (!have.valid_ghosts || !have.valid_outer) {
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

  std::vector<write_scope_t> scopes;
  for (const gi_t gi : groups) {
    const tl_range sync_tls =
        timelevels_t::all_but_oldest().range(ext.record(gi));
    const valid_t fills = sync_fills(window, gi);
    for (const tl_t tl : sync_tls) {
      require(window, {.gi = gi,
                       .tl = tl,
                       .where = make_valid_int(),
                       .event = "SyncGroupsByDirI before syncing"});
      scopes.push_back(
          open_write(window, {.gi = gi,
                              .tl = tl,
                              .fills = fills,
                              .keeps = make_valid_int() | make_valid_outer(),
                              .event = "SyncGroupsByDirI"}));
    }
  }

  // We need to loop over groups, patches, and levels in a definite
  // order so that AMReX's communication pattern does not get
  // confused. Therefore all the loops here are serial. The only
  // parallelization happens within AMReX and within our boundary
  // conditions. This is not efficient.

  task_manager tasks1;
  task_manager tasks2;
  task_manager tasks3;

  for (const gi_t gi : groups) {
    loop_serially(window, [&](auto &restrict leveldata) {
      auto &restrict groupdata = leveldata.group(gi);

      const tl_range sync_tls =
          timelevels_t::all_but_oldest().range(groupdata.record);

      if (leveldata.level.index() == 0) {
        // Copy from adjacent boxes on same level

        for (const tl_t tl : sync_tls) {
          tasks1.submit_serially([&tasks2, &leveldata, &groupdata, tl, &ext]() {
            FillPatch_Sync(tasks2, groupdata, *groupdata.timelevel(tl).mfab,
                           ext.patch(leveldata.level.patch())
                               .amrcore->Geom(leveldata.level.index()));
          });
        } // for tl

      } else { // if leveldata.level > 0
        // Copy from adjacent boxes on same level, and interpolate
        // from next coarser level

        const auto &restrict coarseleveldata =
            ext.level(leveldata.level.parent());
        auto &restrict coarsegroupdata = coarseleveldata.group(gi);
        assert(coarsegroupdata.record.numvars == groupdata.record.numvars);

        amrex::Interpolater *const interpolator = groupdata.interpolator;

        for (const tl_t tl : sync_tls) {
          error_if_parent_invalid(
              leveldata, gi, tl,
              "SyncGroupsByDirI on coarse level before prolongation");
          tasks1.submit_serially([&tasks2, &tasks3, &leveldata, &groupdata,
                                  &coarsegroupdata, interpolator, tl, &ext]() {
            FillPatch_ProlongateGhosts(
                tasks2, tasks3, groupdata, coarsegroupdata,
                *groupdata.timelevel(tl).mfab,
                *coarsegroupdata.timelevel(tl).mfab,
                ext.patch(leveldata.level.patch())
                    .amrcore->Geom(leveldata.level.index()),
                ext.patch(leveldata.level.patch())
                    .amrcore->Geom(leveldata.level.parent().index()),
                interpolator, groupdata.bcrecs);
          });

        } // for tl

      } // if leveldata.level > 0
    });
  } // for gi

  tasks1.run_tasks_serially();
  synchronize();
  tasks2.run_tasks_serially();
  synchronize();
  tasks3.run_tasks_serially();
  synchronize();

  sync_multipatch_boundaries(groups);
  scopes.clear();

  assert(sync_active);
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

static bool
integrator_owns_coarse_fine_ghosts(const GHExt::PatchData::LevelData &leveldata,
                                   const gi_t gi) {
  return leveldata.level.index() > 0 && group_is_integrated(gi) &&
         leveldata.iteration > 0;
}

void SyncGroupsByDirISubcycling(const window_t window,
                                const std::span<const gi_t> groups0) {
  DECLARE_CCTK_PARAMETERS;

  const GHExt &ext = driver();

  error_if_sync_in_tail(window, groups0);

  mark_sync_active marked;

  static Timer timer("Sync");
  Interval interval(timer);

  sync_log_groups("SyncGroupsBySubcycling", groups0);

  std::vector<gi_t> groups = sync_filter_groups(groups0);

  const char *const prolongation_label =
      "SyncGroupsByDirISubcycling on coarse level before prolongation";
  const char *const sync_label = "SyncGroupsByDirISubcycling before syncing";

  struct ghost_check_t {
    window_t levels;
    gi_t gi;
    tl_t tl;
  };
  std::vector<ghost_check_t> integrated_ghosts;
  std::vector<write_scope_t> scopes;
  for (const gi_t gi : groups) {
    const tl_range sync_tls =
        timelevels_t::all_but_oldest().range(ext.record(gi));
    const std::vector<level_run_t> runs = split_levels(
        window, [gi](const GHExt::PatchData::LevelData &leveldata) {
          return integrator_owns_coarse_fine_ghosts(leveldata, gi);
        });
    for (const level_run_t &run : runs) {
      const valid_t fills = sync_fills(run.levels, gi);
      for (const tl_t tl : sync_tls) {
        if (run.value) {
          integrated_ghosts.push_back({run.levels, gi, tl});
          scopes.push_back(
              open_write(run.levels, {.gi = gi,
                                      .tl = tl,
                                      .fills = fills & make_valid_outer(),
                                      .keeps = make_valid_all(),
                                      .event = "SyncGroupsByDirISubcycling"}));
        } else {
          scopes.push_back(open_write(
              run.levels, {.gi = gi,
                           .tl = tl,
                           .fills = fills,
                           .keeps = make_valid_int() | make_valid_outer(),
                           .event = "SyncGroupsByDirISubcycling"}));
        }
      }
    }
  }

  // We need to loop over groups, patches, and levels in a definite
  // order so that AMReX's communication pattern does not get
  // confused. Therefore all the loops here are serial. The only
  // parallelization happens within AMReX and within our boundary
  // conditions. This is not efficient.

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
          tasks1.submit_serially([&tasks2, &leveldata, &groupdata, tl, &ext]() {
            FillPatch_Sync(tasks2, groupdata, *groupdata.timelevel(tl).mfab,
                           ext.patch(leveldata.level.patch())
                               .amrcore->Geom(leveldata.level.index()));
          });
        } // for tl

      } else { // if leveldata.level > 0

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
            tasks1.submit_serially([&tasks2, &leveldata, &groupdata, tl,
                                    &ext]() {
              FillPatch_Sync(tasks2, groupdata, *groupdata.timelevel(tl).mfab,
                             ext.patch(leveldata.level.patch())
                                 .amrcore->Geom(leveldata.level.index()));
            });
          } // for tl
        } else {
          // Copy from adjacent boxes on same level, and interpolate
          // from next coarser level
          amrex::Interpolater *const interpolator = groupdata.interpolator;

          const bool aligned =
              (leveldata.iteration == coarseleveldata.iteration);
          const bool blend = !aligned && coarsegroupdata.num_timelevels() >= 2;
          const rat64 cdt = coarseleveldata.delta_iteration;
          const rat64 t_new = coarseleveldata.iteration;
          const rat64 t_fin = leveldata.iteration;
          const CCTK_REAL w_new = CCTK_REAL((t_fin - t_new + cdt) / cdt);

          for (const tl_t tl : sync_tls) {
            const bool do_blend = blend && tl == tl_t{0};
            require(window_t::one_level(leveldata.level),
                    {.gi = gi,
                     .tl = tl,
                     .where = make_valid_int(),
                     .event = sync_label});
            error_if_parent_invalid(leveldata, gi, tl, prolongation_label);
            if (do_blend)
              error_if_parent_invalid(leveldata, gi, tl_t{1},
                                      prolongation_label);
            const amrex::MultiFab *const cmfab_old =
                do_blend ? coarsegroupdata.timelevel(tl_t{1}).mfab.get()
                         : nullptr;
            const CCTK_REAL tl_w_new = do_blend ? w_new : CCTK_REAL(1);
            tasks1.submit_serially([&tasks2, &tasks3, &leveldata, &groupdata,
                                    &coarsegroupdata, interpolator, tl,
                                    cmfab_old, tl_w_new, &ext]() {
              FillPatch_ProlongateGhosts(
                  tasks2, tasks3, groupdata, coarsegroupdata,
                  *groupdata.timelevel(tl).mfab,
                  *coarsegroupdata.timelevel(tl).mfab,
                  ext.patch(leveldata.level.patch())
                      .amrcore->Geom(leveldata.level.index()),
                  ext.patch(leveldata.level.patch())
                      .amrcore->Geom(leveldata.level.parent().index()),
                  interpolator, groupdata.bcrecs, cmfab_old, tl_w_new);
            });
          } // for tl
        }

      } // if leveldata.level > 0
    });
  } // for gi

  tasks1.run_tasks_serially();
  synchronize();
  tasks2.run_tasks_serially();
  synchronize();
  tasks3.run_tasks_serially();
  synchronize();

  for (const ghost_check_t &check : integrated_ghosts)
    require(check.levels,
            {.gi = check.gi,
             .tl = check.tl,
             .where = make_valid_ghosts(),
             .event = "SyncGroupsByDirISubcycling after copying the same-level "
                      "ghosts of an integrated group on a refined level",
             .hint = []() {
               return "  With CarpetX::use_subcycling = yes the RK step "
                      "owns the coarse-fine ghosts of the groups the "
                      "time integrator registered at iteration > 0: "
                      "CarpetX::rk_step_t::state_updated after each RK "
                      "stage, and the driver's refill after a recovery, "
                      "fill and mark them before any SYNC.\n"
                      "  The SYNC copies only the same-level ghosts of "
                      "such a group and requires its ghosts to be marked "
                      "valid already.\n";
             }});

  sync_multipatch_boundaries(groups);
  scopes.clear();

  assert(sync_active);
}

void SyncGroups(const window_t window, const std::span<const gi_t> groups) {
  if (driver().use_subcycling)
    SyncGroupsByDirISubcycling(window, groups);
  else
    SyncGroupsByDirI(window, groups);
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

void SyncGroupsByDirIProlongateOnly(const window_t window,
                                    const std::span<const gi_t> groups0,
                                    const timelevels_t tls) {
  SyncGroupsByDirIProlongateOnly_impl(window, groups0, false, tls);
}

void SyncGroupsByDirIProlongateOnlyAligned(const window_t window,
                                           const std::span<const gi_t> groups0,
                                           const timelevels_t tls) {
  SyncGroupsByDirIProlongateOnly_impl(window, groups0, true, tls);
}

void SyncGroupsByDirIGhostOnly(const window_t window,
                               const std::span<const gi_t> groups0,
                               const timelevels_t tls) {
  DECLARE_CCTK_PARAMETERS;

  const GHExt &ext = driver();

  mark_sync_active marked;

  static Timer timer("Sync");
  Interval interval(timer);

  sync_log_groups("SyncGroupsGhostOnly", groups0);

  std::vector<gi_t> groups = sync_filter_groups(groups0);

  // We need to loop over groups, patches, and levels in a definite
  // order so that AMReX's communication pattern does not get
  // confused. Therefore all the loops here are serial. The only
  // parallelization happens within AMReX and within our boundary
  // conditions. This is not efficient.

  task_manager tasks1;
  task_manager tasks2;

  for (const gi_t gi : groups) {
    loop_serially(window, [&](auto &restrict leveldata) {
      auto &restrict groupdata = leveldata.group(gi);

      for (const tl_t tl : tls.range(groupdata.record)) {
        tasks1.submit_serially([&tasks2, &leveldata, &groupdata, tl, &ext]() {
          FillPatch_Sync(tasks2, groupdata, *groupdata.timelevel(tl).mfab,
                         ext.patch(leveldata.level.patch())
                             .amrcore->Geom(leveldata.level.index()));
        });
      } // for tl
    });
  } // for gi

  tasks1.run_tasks_serially();
  synchronize();
  tasks2.run_tasks_serially();
  synchronize();

  sync_multipatch_boundaries(groups);

  assert(sync_active);
}

void ProlongateAfterRestriction(const window_t window) {
  const std::vector<gi_t> groups = reprolongate_groups();
  SyncGroupsByDirIProlongateOnly_impl(window, groups, true,
                                      timelevels_t::all_but_oldest());
}

static void restrict_level(const level_t level,
                           const std::span<const gi_t> groups,
                           const valid_t coarse_keeps) {
  static Timer timer("Restrict");
  Interval interval(timer);

  const gi_t gi_regrid_error{CCTK_GroupIndex("CarpetXRegrid::regrid_error")};
  assert(gi_regrid_error.index() >= 0);

  const std::optional<level_t> fine = level.child();
  if (!fine)
    return;
  const GHExt &ext = driver();
  auto &leveldata = ext.level(level);
  const auto &fineleveldata = ext.level(*fine);
  const window_t coarse_level = window_t::one_level(level);
  const window_t fine_level = window_t::one_level(*fine);

  for (const gi_t gi : groups) {
    const auto &groupdata = leveldata.group(gi);
    const auto &finegroupdata = fineleveldata.group(gi);
    const amrex::IntVect reffact{2, 2, 2};

    // Don't restrict the regridding error
    if (gi == gi_regrid_error)
      continue;

    for (const tl_t tl :
         timelevels_t::all_but_oldest().range(groupdata.record)) {

      require(fine_level,
              {.gi = gi,
               .tl = tl,
               .where = make_valid_int(),
               .event = "Restrict on fine level before restricting"});
      require(coarse_level,
              {.gi = gi,
               .tl = tl,
               .where = make_valid_int(),
               .event = "Restrict on coarse level before restricting"});
      const write_scope_t scope =
          open_write(coarse_level, {.gi = gi,
                                    .tl = tl,
                                    .fills = make_valid_int(),
                                    .keeps = coarse_keeps,
                                    .event = "Restrict"});

#if 1
      {
        static Timer timer("Restrict::average_down");
        Interval interval(timer);
        int rank = 0;
        for (int d = 0; d < dim; ++d)
          rank += groupdata.indextype.at(d);
        switch (rank) {
        case 0:
          average_down_nodal(*finegroupdata.timelevel(tl).mfab,
                             *groupdata.timelevel(tl).mfab, reffact);
          break;
        case 1:
          average_down_edges(*finegroupdata.timelevel(tl).mfab,
                             *groupdata.timelevel(tl).mfab, reffact);
          break;
        case 2:
          average_down_faces(*finegroupdata.timelevel(tl).mfab,
                             *groupdata.timelevel(tl).mfab, reffact);
          break;
        case 3:
          average_down(*finegroupdata.timelevel(tl).mfab,
                       *groupdata.timelevel(tl).mfab, 0,
                       groupdata.record.numvars, reffact);
          break;
        default:
          assert(0);
        }
      }
#endif

    } // for tl
  } // for gi
}

void Restrict(const window_t window, const std::span<const gi_t> groups) {
  if (groups.empty())
    return;
  loop_pairs_fine_to_coarse(window, [&](const level_t level) {
    restrict_level(level, groups, make_valid_int());
  });
}

void RestrictNoPoison(const window_t window,
                      const std::span<const gi_t> groups) {
  if (groups.empty())
    return;
  loop_pairs_fine_to_coarse(window, [&](const level_t level) {
    restrict_level(level, groups, make_valid_all());
  });
}

} // namespace CarpetX
