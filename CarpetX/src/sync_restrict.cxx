#include "schedule.hxx"
#include "driver.hxx"
#include "fillpatch.hxx"
#include "subcycling.hxx"
#include "sync_restrict_internal.hxx"
#include "task_manager.hxx"
#include <carpetx.hxx>
#include "valid.hxx"

#include <cctk.h>
#include <cctk_Parameters.h>

#include <AMReX_FluxRegister.H>
#include <AMReX_MultiFabUtil.H>

#include <algorithm>
#include <array>
#include <cassert>
#include <functional>
#include <memory>
#include <optional>
#include <sstream>
#include <utility>
#include <vector>

namespace CarpetX {

// SyncGroupsByDirI's restrict_during_sync branch calls the window operation
// Restrict(cctkGH, min_level, max_level, groups), which
// sync_restrict_internal.hxx declares, on the groups that the selector
// sync_restrict_groups picks from the SYNC's list. To keep the top-down order
// (helpers → selectors → Sync* → Reflux → Restrict*) the operation's
// definition appears below.

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

static void sync_log_groups(const char *label, int numgroups,
                            const int *groups0) {
  DECLARE_CCTK_PARAMETERS;
  if (verbose) {
    std::ostringstream buf;
    for (int n = 0; n < numgroups; ++n) {
      if (n != 0)
        buf << ", ";
      buf << CCTK_FullGroupName(groups0[n]);
    }
#pragma omp critical
    CCTK_VINFO("%s %s", label, buf.str().c_str());
  }
}

static std::vector<int> sync_filter_groups(int numgroups, const int *groups0) {
  static const int gi_regrid_error =
      CCTK_GroupIndex("CarpetXRegrid::regrid_error");
  assert(gi_regrid_error >= 0);

  std::vector<int> groups;
  for (int n = 0; n < numgroups; ++n) {
    const int gi = groups0[n];
    if (ghext->active_timelevels.at(gi) == 0)
      continue;
    if (CCTK_GroupTypeI(gi) != CCTK_GF)
      continue;
    // Don't restrict the regridding error
    if (gi == gi_regrid_error)
      continue;
    groups.push_back(gi);
  }
  return groups;
}

static void sync_multipatch_boundaries(const cGH *cctkGH,
                                       const std::vector<int> &groups) {
  static const bool have_multipatch_boundaries =
      CCTK_IsFunctionAliased("MultiPatch_Interpolate");

  if (have_multipatch_boundaries) {
    std::vector<CCTK_INT> cactusvarinds;
    for (int group : groups) {
      const auto &groupdata =
          *ghext->patchdata.at(0).leveldata.at(0).groupdata.at(group);
      for (int var = 0; var < groupdata.numvars; ++var)
        cactusvarinds.push_back(groupdata.firstvarindex + var);
    }
    MultiPatch_Interpolate(cctkGH, cactusvarinds.size(), cactusvarinds.data());
  } else {
    assert(ghext->num_patches() == 1);
  }
}

static int timelevels_but_oldest(const int gi) {
  const auto &groupdata0 =
      *ghext->patchdata.at(0).leveldata.at(0).groupdata.at(gi);
  assert(!groupdata0.mfab.empty());
  const int ntls = groupdata0.mfab.size();
  return ntls > 1 ? ntls - 1 : ntls;
}

static valid_t sync_fills(const active_levels_t &active_levels, const int gi) {
  std::optional<bool> covered;
  active_levels.loop_serially([&](auto &restrict leveldata) {
    const bool level_covered =
        leveldata.groupdata.at(gi)->all_faces_have_symmetries_or_boundaries();
    assert(!covered || *covered == level_covered);
    covered = level_covered;
  });
  return make_valid_ghosts() |
         (covered.value_or(false) ? make_valid_outer() : valid_t());
}

struct level_run_t {
  active_levels_t levels;
  bool value;
};

static std::vector<level_run_t> split_levels(
    const active_levels_t &active_levels,
    const std::function<bool(const GHExt::PatchData::LevelData &)> &predicate) {
  const int min_level = active_levels.min_level;
  std::vector<std::optional<bool> > values(active_levels.max_level - min_level);
  active_levels.loop_serially([&](auto &restrict leveldata) {
    std::optional<bool> &value = values.at(leveldata.level - min_level);
    const bool level_value = predicate(leveldata);
    assert(!value || *value == level_value);
    value = level_value;
  });
  std::vector<level_run_t> runs;
  for (int lo = min_level; lo < active_levels.max_level;) {
    const std::optional<bool> value = values.at(lo - min_level);
    int hi = lo + 1;
    while (hi < active_levels.max_level && values.at(hi - min_level) == value)
      ++hi;
    if (value)
      runs.push_back({active_levels_t(lo, hi, active_levels.min_patch,
                                      active_levels.max_patch),
                      *value});
    lo = hi;
  }
  return runs;
}

static void error_if_parent_invalid(const active_levels_t &active_levels,
                                    const int gi, const int tl,
                                    const char *const label) {
  using std::max;
  const int min_level = active_levels.min_level;
  const int max_level = active_levels.max_level;
  const int min_patch = active_levels.min_patch;
  const int max_patch = active_levels.max_patch;
  const int first_child = max(min_level, 1);
  if (first_child >= max_level)
    return;
  std::vector<active_levels_t> parents;
  if (first_child == min_level)
    parents.emplace_back(min_level - 1, min_level, min_patch, max_patch);
  if (first_child < max_level - 1)
    parents.emplace_back(first_child, max_level - 1, min_patch, max_patch);
  const auto &groupdata0 =
      *ghext->patchdata.at(0).leveldata.at(0).groupdata.at(gi);
  bool recovering = false;
  if (ghext->recovered_iteration >= 0)
    for (const active_levels_t &window : parents)
      window.loop_serially([&](auto &restrict leveldata) {
        recovering |=
            leveldata.last_finalized_iteration <= ghext->recovered_iteration;
      });

  const std::function<std::string()> hint = [&groupdata0, tl,
                                             recovering]() -> std::string {
    if (groupdata0.do_checkpoint)
      return "";
    if (tl > 0)
      return "  Group " + groupdata0.groupname +
             " is not checkpointed, so a recovery, or a regrid that remakes "
             "its level, can lose its past timelevels, which no routine can "
             "recompute.\n"
             "  Past timelevels read by another level must be checkpointed.\n";
    if (recovering)
      return "  This run was recovered and group " + groupdata0.groupname +
             " is not checkpointed.\n"
             "  Groups read by another level must be checkpointed or "
             "recomputed in CCTK_POST_RECOVER_VARIABLES.\n";
    return "";
  };
  for (const active_levels_t &window : parents)
    require(window, {.gi = gi,
                     .tl = tl,
                     .where = make_valid_int(),
                     .event = label,
                     .hint = hint});
}

// =======================================================================
// Restriction group selectors
// =======================================================================

// Each driver path that restricts chooses its groups with one of these
// selectors, and they are the only code that selects groups by the
// per-group restriction flags that the restrict tag sets (do_restrict,
// restrict_on_sync). The operations below (Restrict, RestrictNoPoison) apply
// no group rule: they restrict exactly the list they are given, so that an
// explicit call restricts a group even when its tags keep the driver away
// from it. The selectors read the flags on patch 0, level 0; every patch and
// level carries the same ones.

// The groups the end-of-step restriction restricts: those whose restrict
// tag is "yes", where a missing restrict tag takes the value of the
// checkpoint tag, and a missing checkpoint tag counts as "yes"
std::vector<int> collect_restrictable_groups() {
  const int numgroups = CCTK_NumGroups();
  std::vector<int> groups;
  groups.reserve(numgroups);
  const auto &patchdata0 = ghext->patchdata.at(0);
  const auto &leveldata0 = patchdata0.leveldata.at(0);
  for (const auto &groupdataptr : leveldata0.groupdata) {
    // Only grid functions
    if (groupdataptr) {
      auto &restrict groupdata = *groupdataptr;
      // Only grid functions with storage
      if (groupdata.mfab.empty())
        continue;
      // Only grid functions with restriction enabled
      if (groupdata.do_restrict)
        groups.push_back(groupdata.groupindex);
    }
  }
  return groups;
}

// The groups the end-of-step re-prolongation refills: every group the
// end-of-step restriction restricts and, under subcycling, every integrated
// group (see ProlongateAfterRestriction)
static std::vector<int> reprolongate_groups() {
  const int numgroups = CCTK_NumGroups();
  std::vector<int> groups;
  groups.reserve(numgroups);
  const auto &patchdata0 = ghext->patchdata.at(0);
  const auto &leveldata0 = patchdata0.leveldata.at(0);
  for (const auto &groupdataptr : leveldata0.groupdata) {
    // Only grid functions
    if (groupdataptr) {
      auto &restrict groupdata = *groupdataptr;
      // Only grid functions with storage
      if (groupdata.mfab.empty())
        continue;
      const int gi = groupdata.groupindex;
      if (groupdata.do_restrict ||
          (ghext->use_subcycling && group_is_integrated(gi)))
        groups.push_back(gi);
    }
  }
  return groups;
}

// The groups of a SYNC that its restriction (restrict_during_sync) restricts:
// every group in `groups` except those tagged restrict="no". Unlike at the
// end of a step, a group without a restrict tag is restricted whatever its
// checkpoint tag. The SYNC still fills the ghosts of every group in `groups`.
static std::vector<int> sync_restrict_groups(const std::vector<int> &groups) {
  std::vector<int> restrict_groups;
  restrict_groups.reserve(groups.size());
  const auto &patchdata0 = ghext->patchdata.at(0);
  const auto &leveldata0 = patchdata0.leveldata.at(0);
  for (const int gi : groups) {
    // The SYNC passes only grid functions with storage
    const auto &groupdataptr = leveldata0.groupdata.at(gi);
    assert(groupdataptr);
    if (groupdataptr->restrict_on_sync)
      restrict_groups.push_back(gi);
  }
  return restrict_groups;
}

// =======================================================================
// Sync entry points
// =======================================================================

static int SyncGroupsByDirIProlongateOnly_impl(
    const active_levels_t &active_levels, const cGH *restrict cctkGH,
    int numgroups, const int *groups0, const int *directions,
    const bool prolongate_on_same_iteration, const int tl_arg = -1) {
  DECLARE_CCTK_PARAMETERS;

  assert(in_global_mode(cctkGH) || in_level_mode(cctkGH));

  mark_sync_active marked;

  static Timer timer("Sync");
  Interval interval(timer);

  assert(cctkGH);
  assert(numgroups >= 0);
  assert(groups0);

  sync_log_groups("SyncGroupsProlongateOnly", numgroups, groups0);

  std::vector<int> groups = sync_filter_groups(numgroups, groups0);

  const auto prolongates_into =
      [prolongate_on_same_iteration](
          const GHExt::PatchData::LevelData &leveldata) {
        if (leveldata.level == 0)
          return false;
        if (ghext->use_subcycling && prolongate_on_same_iteration) {
          const auto &restrict coarseleveldata =
              ghext->patchdata.at(leveldata.patch)
                  .leveldata.at(leveldata.level - 1);
          return leveldata.iteration == coarseleveldata.iteration;
        }
        return true;
      };
  const auto timelevels_to_fill =
      [tl_arg](const GHExt::PatchData::LevelData::GroupData &groupdata) {
        const int ntls = groupdata.mfab.size();
        const int tl_lo = (tl_arg < 0) ? 0 : tl_arg;
        const int tl_hi =
            (tl_arg < 0) ? (ntls > 1 ? ntls - 1 : ntls) : (tl_arg + 1);
        assert(tl_lo >= 0 && tl_hi <= ntls);
        return std::make_pair(tl_lo, tl_hi);
      };

  const std::vector<level_run_t> runs =
      split_levels(active_levels, prolongates_into);
  std::vector<write_scope_t> scopes;
  for (const int gi : groups) {
    const auto [tl_lo, tl_hi] = timelevels_to_fill(
        *ghext->patchdata.at(0).leveldata.at(0).groupdata.at(gi));
    for (const level_run_t &run : runs) {
      if (!run.value)
        continue;
      const valid_t fills = sync_fills(run.levels, gi);
      for (int tl = tl_lo; tl < tl_hi; ++tl) {
        error_if_parent_invalid(run.levels, gi, tl,
                                "SyncGroupsByDirIProlongateOnly on coarse "
                                "level before prolongation");
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

  for (const int gi : groups) {
    active_levels.loop_serially([&](auto &restrict leveldata) {
      auto &restrict groupdata = *leveldata.groupdata.at(gi);
      assert(!groupdata.mfab.empty());

      const auto [tl_lo, tl_hi] = timelevels_to_fill(groupdata);

      if (!prolongates_into(leveldata))
        return;

      const int level = leveldata.level;
      const auto &restrict coarseleveldata =
          ghext->patchdata.at(leveldata.patch).leveldata.at(level - 1);

      auto &restrict coarsegroupdata = *coarseleveldata.groupdata.at(gi);
      assert(!coarsegroupdata.mfab.empty());
      assert(coarsegroupdata.numvars == groupdata.numvars);

      amrex::Interpolater *const interpolator = groupdata.interpolator;

      for (int tl = tl_lo; tl < tl_hi; ++tl) {

        tasks1.submit_serially([&tasks2, &tasks3, &leveldata, &groupdata,
                                &coarsegroupdata, interpolator, tl]() {
          FillPatch_ProlongateOnly(tasks2, tasks3, groupdata, coarsegroupdata,
                                   *groupdata.mfab.at(tl),
                                   *coarsegroupdata.mfab.at(tl),
                                   ghext->patchdata.at(leveldata.patch)
                                       .amrcore->Geom(leveldata.level),
                                   ghext->patchdata.at(leveldata.patch)
                                       .amrcore->Geom(leveldata.level - 1),
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

  sync_multipatch_boundaries(cctkGH, groups);
  scopes.clear();

  assert(sync_active);

  return numgroups; // number of groups synchronized
}

int SyncGroupsByDirI(const active_levels_t &active_levels,
                     const cGH *restrict cctkGH, int numgroups,
                     const int *groups0, const int *directions) {
  DECLARE_CCTK_PARAMETERS;

  assert(in_global_mode(cctkGH) || in_level_mode(cctkGH));

  mark_sync_active marked;

  static Timer timer("Sync");
  Interval interval(timer);

  assert(cctkGH);
  assert(numgroups >= 0);
  assert(groups0);

  sync_log_groups("SyncGroups", numgroups, groups0);

  std::vector<int> groups = sync_filter_groups(numgroups, groups0);

  if (CCTK_EQUALS(presync_mode, "presync-only")) {
    active_levels.loop_serially([&](auto &restrict leveldata) {
      std::vector<int> new_groups;
      for (const int gi : groups) {
        auto &restrict groupdata = *leveldata.groupdata.at(gi);
        assert(!groupdata.mfab.empty());
        bool need_sync = false;
        for (int tl = 0; tl < int(groupdata.valid.size()); tl++) {
          if (need_sync)
            break;
          auto &timeleveldata = groupdata.valid.at(tl);
          for (int vi = 0; vi < int(timeleveldata.size()); vi++) {
            if (need_sync)
              break;
            valid_t have = groupdata.valid.at(tl).at(vi).get();
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
      return 0;
    }
  }

  if (restrict_during_sync) {
    const int min_level = active_levels.min_level;
    const int max_level =
        std::min(active_levels.max_level + 1, ghext->num_levels());
    // The do_restrict parameter governs only restriction the driver starts
    assert(do_restrict || max_level - min_level < 2);
    // Restrict only the groups the restrict tag allows; the ghost fill below
    // still covers all of `groups`
    Restrict(cctkGH, min_level, max_level, sync_restrict_groups(groups));
  }

  std::vector<write_scope_t> scopes;
  for (const int gi : groups) {
    const int sync_tl = timelevels_but_oldest(gi);
    const valid_t fills = sync_fills(active_levels, gi);
    for (int tl = 0; tl < sync_tl; ++tl) {
      error_if_parent_invalid(
          active_levels, gi, tl,
          "SyncGroupsByDirI on coarse level before prolongation");
      require(active_levels, {.gi = gi,
                              .tl = tl,
                              .where = make_valid_int(),
                              .event = "SyncGroupsByDirI before syncing"});
      scopes.push_back(open_write(
          active_levels, {.gi = gi,
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

  for (const int gi : groups) {
    active_levels.loop_serially([&](auto &restrict leveldata) {
      auto &restrict groupdata = *leveldata.groupdata.at(gi);
      assert(!groupdata.mfab.empty());

      // We always sync all directions.
      // If there is more than one time level, then we don't sync the
      // oldest.
      // TODO: during evolution, sync only one time level
      const int ntls = groupdata.mfab.size();
      const int sync_tl = ntls > 1 ? ntls - 1 : ntls;

      if (leveldata.level == 0) {
        // Copy from adjacent boxes on same level

        for (int tl = 0; tl < sync_tl; ++tl) {
          tasks1.submit_serially([&tasks2, &leveldata, &groupdata, tl]() {
            FillPatch_Sync(tasks2, groupdata, *groupdata.mfab.at(tl),
                           ghext->patchdata.at(leveldata.patch)
                               .amrcore->Geom(leveldata.level));
          });
        } // for tl

      } else { // if leveldata.level > 0
        // Copy from adjacent boxes on same level, and interpolate
        // from next coarser level

        const int level = leveldata.level;
        const auto &restrict coarseleveldata =
            ghext->patchdata.at(leveldata.patch).leveldata.at(level - 1);
        auto &restrict coarsegroupdata = *coarseleveldata.groupdata.at(gi);
        assert(!coarsegroupdata.mfab.empty());
        assert(coarsegroupdata.numvars == groupdata.numvars);

        amrex::Interpolater *const interpolator = groupdata.interpolator;

        for (int tl = 0; tl < sync_tl; ++tl) {

          tasks1.submit_serially([&tasks2, &tasks3, &leveldata, &groupdata,
                                  &coarsegroupdata, interpolator, tl]() {
            FillPatch_ProlongateGhosts(tasks2, tasks3, groupdata,
                                       coarsegroupdata, *groupdata.mfab.at(tl),
                                       *coarsegroupdata.mfab.at(tl),
                                       ghext->patchdata.at(leveldata.patch)
                                           .amrcore->Geom(leveldata.level),
                                       ghext->patchdata.at(leveldata.patch)
                                           .amrcore->Geom(leveldata.level - 1),
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

  sync_multipatch_boundaries(cctkGH, groups);
  scopes.clear();

  assert(sync_active);

  return numgroups; // number of groups synchronized
}

static void error_if_sync_in_tail(const active_levels_t &active_levels,
                                  const cGH *restrict const cctkGH,
                                  const int numgroups, const int *groups0) {
  if (!ghext->tail_bin || numgroups == 0)
    return;
  const int min_level = active_levels.min_level;
  const int max_level = active_levels.max_level;
  if (min_level == 0)
    return;
  std::ostringstream groupnames;
  for (int n = 0; n < numgroups; ++n) {
    const char *const groupname = CCTK_FullGroupName(groups0[n]);
    if (n != 0)
      groupnames << ", ";
    groupnames << (groupname ? groupname : "(unknown group)");
  }
  const cFunctionData *const func = ghext->last_called_function;
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
      func ? func->routine : "(unknown routine)", ghext->tail_bin,
      groupnames.str().c_str(), cctkGH->cctk_iteration, min_level, max_level,
      min_level, min_level - 1);
}

static bool
integrator_owns_coarse_fine_ghosts(const GHExt::PatchData::LevelData &leveldata,
                                   const int gi) {
  return leveldata.level > 0 && group_is_integrated(gi) &&
         leveldata.iteration > 0;
}

int SyncGroupsByDirISubcycling(const active_levels_t &active_levels,
                               const cGH *restrict cctkGH, int numgroups,
                               const int *groups0, const int *directions) {
  DECLARE_CCTK_PARAMETERS;

  assert(in_global_mode(cctkGH) || in_level_mode(cctkGH));

  error_if_sync_in_tail(active_levels, cctkGH, numgroups, groups0);

  mark_sync_active marked;

  static Timer timer("Sync");
  Interval interval(timer);

  assert(cctkGH);
  assert(numgroups >= 0);
  assert(groups0);

  sync_log_groups("SyncGroupsBySubcycling", numgroups, groups0);

  std::vector<int> groups = sync_filter_groups(numgroups, groups0);

  const char *const prolongation_label =
      "SyncGroupsByDirISubcycling on coarse level before prolongation";

  struct ghost_check_t {
    active_levels_t levels;
    int gi, tl;
  };
  std::vector<ghost_check_t> integrated_ghosts;
  std::vector<write_scope_t> scopes;
  for (const int gi : groups) {
    const int sync_tl = timelevels_but_oldest(gi);
    const std::vector<level_run_t> runs = split_levels(
        active_levels, [gi](const GHExt::PatchData::LevelData &leveldata) {
          return integrator_owns_coarse_fine_ghosts(leveldata, gi);
        });
    for (const level_run_t &run : runs) {
      const valid_t fills = sync_fills(run.levels, gi);
      for (int tl = 0; tl < sync_tl; ++tl) {
        if (run.value) {
          integrated_ghosts.push_back({run.levels, gi, tl});
          scopes.push_back(
              open_write(run.levels, {.gi = gi,
                                      .tl = tl,
                                      .fills = fills & make_valid_outer(),
                                      .keeps = make_valid_all(),
                                      .event = "SyncGroupsByDirISubcycling"}));
        } else {
          error_if_parent_invalid(run.levels, gi, tl, prolongation_label);
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

  for (const int gi : groups) {
    active_levels.loop_serially([&](auto &restrict leveldata) {
      auto &restrict groupdata = *leveldata.groupdata.at(gi);
      assert(!groupdata.mfab.empty());

      // We always sync all directions.
      // If there is more than one time level, then we don't sync the
      // oldest.
      // TODO: during evolution, sync only one time level
      const int ntls = groupdata.mfab.size();
      const int sync_tl = ntls > 1 ? ntls - 1 : ntls;

      if (leveldata.level == 0) {
        // Copy from adjacent boxes on same level

        for (int tl = 0; tl < sync_tl; ++tl) {
          tasks1.submit_serially([&tasks2, &leveldata, &groupdata, tl]() {
            FillPatch_Sync(tasks2, groupdata, *groupdata.mfab.at(tl),
                           ghext->patchdata.at(leveldata.patch)
                               .amrcore->Geom(leveldata.level));
          });
        } // for tl

      } else { // if leveldata.level > 0

        const int level = leveldata.level;
        const auto &restrict coarseleveldata =
            ghext->patchdata.at(leveldata.patch).leveldata.at(level - 1);
        auto &restrict coarsegroupdata = *coarseleveldata.groupdata.at(gi);
        assert(!coarsegroupdata.mfab.empty());
        assert(coarsegroupdata.numvars == groupdata.numvars);

        const bool integrated_subiter =
            integrator_owns_coarse_fine_ghosts(leveldata, gi);

        if (integrated_subiter) {
          for (int tl = 0; tl < sync_tl; ++tl) {
            tasks1.submit_serially([&tasks2, &leveldata, &groupdata, tl]() {
              FillPatch_Sync(tasks2, groupdata, *groupdata.mfab.at(tl),
                             ghext->patchdata.at(leveldata.patch)
                                 .amrcore->Geom(leveldata.level));
            });
          } // for tl
        } else {
          // Copy from adjacent boxes on same level, and interpolate
          // from next coarser level
          amrex::Interpolater *const interpolator = groupdata.interpolator;

          // Time-blend gate: only a group the time integrator does not own
          // (!group_is_integrated) can reach this branch while misaligned in
          // time with its parent. When the fine level is ahead of the coarse,
          // blend coarse tl=0 (new) with coarse tl=1 (old) into the coarse
          // patch before interpolating. Mirrors AMReX beta=(time-t0)/(t1-t0):
          // with t0 = t_new - cdt (old coarse time), t1 = t_new (new coarse
          // time), time = t_fin (fine time). A single-timelevel group has no
          // tl=1, so it gets coarse tl=0 as it currently stands.
          const bool aligned =
              (leveldata.iteration == coarseleveldata.iteration);
          const bool blend = !aligned && coarsegroupdata.mfab.size() >= 2;
          const rat64 cdt = coarseleveldata.delta_iteration;
          const rat64 t_new = coarseleveldata.iteration;
          const rat64 t_fin = leveldata.iteration;
          const CCTK_REAL w_new = CCTK_REAL((t_fin - t_new + cdt) / cdt);

          for (int tl = 0; tl < sync_tl; ++tl) {
            const bool do_blend = blend && tl == 0;
            if (do_blend)
              error_if_parent_invalid(
                  active_levels_t(leveldata.level, leveldata.level + 1,
                                  leveldata.patch, leveldata.patch + 1),
                  gi, 1, prolongation_label);
            const amrex::MultiFab *const cmfab_old =
                do_blend ? coarsegroupdata.mfab.at(1).get() : nullptr;
            const CCTK_REAL tl_w_new = do_blend ? w_new : CCTK_REAL(1);
            tasks1.submit_serially([&tasks2, &tasks3, &leveldata, &groupdata,
                                    &coarsegroupdata, interpolator, tl,
                                    cmfab_old, tl_w_new]() {
              FillPatch_ProlongateGhosts(
                  tasks2, tasks3, groupdata, coarsegroupdata,
                  *groupdata.mfab.at(tl), *coarsegroupdata.mfab.at(tl),
                  ghext->patchdata.at(leveldata.patch)
                      .amrcore->Geom(leveldata.level),
                  ghext->patchdata.at(leveldata.patch)
                      .amrcore->Geom(leveldata.level - 1),
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

  sync_multipatch_boundaries(cctkGH, groups);
  scopes.clear();

  assert(sync_active);

  return numgroups; // number of groups synchronized
}

int SyncGroups(const active_levels_t &active_levels,
               const cGH *restrict const cctkGH, const int numgroups,
               const int *const groups, const int *const directions) {
  return (ghext->use_subcycling ? SyncGroupsByDirISubcycling
                                : SyncGroupsByDirI)(
      active_levels, cctkGH, numgroups, groups, directions);
}

int SyncGroupsByDirIProlongateOnly(const active_levels_t &active_levels,
                                   const cGH *restrict cctkGH, int numgroups,
                                   const int *groups0, const int *directions,
                                   const int tl) {
  return SyncGroupsByDirIProlongateOnly_impl(active_levels, cctkGH, numgroups,
                                             groups0, directions, false, tl);
}

int SyncGroupsByDirIProlongateOnlyAligned(const active_levels_t &active_levels,
                                          const cGH *restrict cctkGH,
                                          int numgroups, const int *groups0,
                                          const int *directions, const int tl) {
  return SyncGroupsByDirIProlongateOnly_impl(active_levels, cctkGH, numgroups,
                                             groups0, directions, true, tl);
}

int SyncGroupsByDirIGhostOnly(const active_levels_t &active_levels,
                              const cGH *restrict cctkGH, int numgroups,
                              const int *groups0, const int *directions,
                              const int tl_arg) {
  DECLARE_CCTK_PARAMETERS;

  assert(in_global_mode(cctkGH) || in_level_mode(cctkGH));

  mark_sync_active marked;

  static Timer timer("Sync");
  Interval interval(timer);

  assert(cctkGH);
  assert(numgroups >= 0);
  assert(groups0);

  sync_log_groups("SyncGroupsGhostOnly", numgroups, groups0);

  std::vector<int> groups = sync_filter_groups(numgroups, groups0);

  // We need to loop over groups, patches, and levels in a definite
  // order so that AMReX's communication pattern does not get
  // confused. Therefore all the loops here are serial. The only
  // parallelization happens within AMReX and within our boundary
  // conditions. This is not efficient.

  task_manager tasks1;
  task_manager tasks2;

  for (const int gi : groups) {
    active_levels.loop_serially([&](auto &restrict leveldata) {
      auto &restrict groupdata = *leveldata.groupdata.at(gi);
      assert(!groupdata.mfab.empty());

      // We always sync all directions.
      // tl_arg >= 0 syncs only that timelevel; tl_arg = -1 syncs every
      // timelevel except the oldest (or the only one, if ntls == 1).
      // TODO: during evolution, sync only one time level
      const int ntls = groupdata.mfab.size();
      const int tl_lo = (tl_arg < 0) ? 0 : tl_arg;
      const int tl_hi =
          (tl_arg < 0) ? (ntls > 1 ? ntls - 1 : ntls) : (tl_arg + 1);
      assert(tl_lo >= 0 && tl_hi <= ntls);

      // Copy from adjacent boxes on same level
      for (int tl = tl_lo; tl < tl_hi; ++tl) {
        tasks1.submit_serially([&tasks2, &leveldata, &groupdata, tl]() {
          FillPatch_Sync(tasks2, groupdata, *groupdata.mfab.at(tl),
                         ghext->patchdata.at(leveldata.patch)
                             .amrcore->Geom(leveldata.level));
        });
      } // for tl
    });
  } // for gi

  tasks1.run_tasks_serially();
  synchronize();
  tasks2.run_tasks_serially();
  synchronize();

  sync_multipatch_boundaries(cctkGH, groups);

  assert(sync_active);

  return numgroups; // number of groups synchronized
}

void ProlongateAfterRestriction(const active_levels_t &active_levels,
                                const cGH *cctkGH) {
  const std::vector<int> groups = reprolongate_groups();
  SyncGroupsByDirIProlongateOnly_impl(active_levels, cctkGH, groups.size(),
                                      groups.data(), nullptr, true);
}

// =======================================================================
// Reflux
// =======================================================================

// Face areas of one level, area[d] = prod_{j != d} dx_j. Thorns provide
// fluxes per unit area (their RHS is -(F_{i+1/2} - F_{i-1/2}) / dx) while
// amrex::FluxRegister::Reflux divides the register by the coarse cell
// volume, so every contribution is multiplied by its own level's face area;
// the fine faces under one coarse face then sum to the coarse face. CarpetX
// levels are Cartesian, so this is one constant per level and direction.
// The coarse side of AccumulateFluxes multiplies by these scalars; the
// MultiFab form that FineAdd reads, LevelData::face_area, is filled by
// amrex::Geometry::GetFaceArea with the same products (see
// EnsureFluxGeometry), so the two agree bitwise.
static std::array<CCTK_REAL, dim> face_areas(const amrex::Geometry &geom) {
  const CCTK_REAL *const dx = geom.CellSize();
  std::array<CCTK_REAL, dim> area;
  for (int d = 0; d < dim; ++d) {
    area[d] = 1;
    for (int j = 0; j < dim; ++j)
      if (j != d)
        area[d] *= dx[j];
  }
  return area;
}

// See subcycling.hxx for the contract and driver.hxx for the members. The
// four MultiFabs are built by amrex::Geometry::GetFaceArea and GetVolume on
// the level's cell-centred BoxArray (converted to the d-face centring for
// face_area[d]) and DistributionMapping, the layout of every GroupData
// MultiFab of the level up to centring, which is what the area and volume
// overloads of amrex::FluxRegister require: FineAdd indexes `area` with the
// MFIter of the flux MultiFab, Reflux indexes `volume` with the MFIter of
// the state. (The coarse side of AccumulateFluxes uses the scalar
// face_areas() instead, bitwise the same value.) Zero ghosts suffice: every
// read is on a valid face or cell of the level's own boxes.
//
// Bit-identity: for Cartesian geometry (the only one AMReX supports in 3D)
// GetVolume fills dx[0]*dx[1]*dx[2] (AMReX_Geometry.cpp:221), the product
// AMReX's constant-volume Reflux overload forms, so switching Reflux
// overloads leaves the correction bit-identical; GetFaceArea fills
// dx[1]*dx[2], dx[0]*dx[2], dx[0]*dx[1] for d = 0, 1, 2
// (AMReX_Geometry.cpp:327-331), the products face_areas() forms, so the
// scalar and the MultiFab areas agree bitwise.
void EnsureFluxGeometry(const int patch, const int level) {
  const auto &patchdata = ghext->patchdata.at(patch);
  const auto &leveldata = patchdata.leveldata.at(level);

  // All four are allocated together, or none is
  [[maybe_unused]] const auto have_all_or_none = [&]() {
    const bool have = bool(leveldata.cell_volume);
    bool good = true;
    for (int d = 0; d < dim; ++d)
      good = good && bool(leveldata.face_area[d]) == have;
    return good;
  };
  assert(have_all_or_none());
  if (leveldata.cell_volume)
    return;

  const amrex::Geometry &geom = patchdata.amrcore->Geom(level);
  assert(leveldata.fab);
  const amrex::BoxArray &ba = leveldata.fab->boxArray();
  const amrex::DistributionMapping &dm = leveldata.fab->DistributionMap();
  assert(ba.ixType().cellCentered());

  for (int d = 0; d < dim; ++d) {
    leveldata.face_area[d] = std::make_unique<amrex::MultiFab>();
    geom.GetFaceArea(*leveldata.face_area[d], ba, dm, d, 0);
  }
  leveldata.cell_volume = std::make_unique<amrex::MultiFab>();
  geom.GetVolume(*leveldata.cell_volume, ba, dm, 0);

  assert(have_all_or_none());
}

void AccumulateFluxes(const int patch, const int level, const int stage,
                      const CCTK_REAL weight) {
  if (!ghext->do_reflux)
    return;
  assert(stage >= 1 && stage <= num_rk_stages());

  static Timer timer("AccumulateFluxes");
  Interval interval(timer);

  const auto &patchdata = ghext->patchdata.at(patch);
  const auto &leveldata = patchdata.leveldata.at(level);
  const bool have_child = level + 1 < int(patchdata.leveldata.size());
  const auto *const childleveldata =
      have_child ? &patchdata.leveldata.at(level + 1) : nullptr;
  const int tl = 0;
  const active_levels_t this_level(level, level + 1, patch, patch + 1);

  for (int gi = 0; gi < int(leveldata.groupdata.size()); ++gi) {
    // only grid functions live on levels
    if (!leveldata.groupdata.at(gi))
      continue;
    const auto &groupdata = *leveldata.groupdata.at(gi);
    if (groupdata.mfab.empty())
      continue;
    if (groupdata.fluxes[0] < 0)
      continue;

    // This level's two roles: coarse side of the pair (level, level + 1),
    // whose register the child owns, and fine side of (level - 1, level),
    // whose register this level owns. Either may be absent (finest level,
    // coarsest level). Both are absent for a flux-tagged group ODESolvers
    // does not integrate (no register is allocated for it), so such a
    // group is skipped here before its fluxes are checked.
    amrex::FluxRegister *const child_freg =
        childleveldata ? childleveldata->groupdata.at(gi)->freg.get() : nullptr;
    amrex::FluxRegister *const own_freg = groupdata.freg.get();
    if (!child_freg && !own_freg)
      continue;

    // The cached face areas; built on the first registered group met, a
    // no-op afterwards, so a level without a register never allocates
    EnsureFluxGeometry(patch, level);

    for (int d = 0; d < dim; ++d) {
      std::ostringstream buf;
      buf << "AccumulateFluxes: flux of " << groupdata.groupname
          << " in direction " << d << " at RK stage " << stage;
      require(this_level, {.gi = groupdata.fluxes.at(d),
                           .tl = tl,
                           .where = make_valid_int(),
                           .event = buf.str()});
    }

    // Nothing is allocated per call; both roles work on the register's thin
    // faces only.
    //
    // Coarse side, in place of amrex::FluxRegister::CrseInit(ADD). Per
    // direction, CrseInit allocates a full-level MultiFab on the flux's
    // layout holding (flux * mult) * area on every face
    // (AMReX_FluxRegister.cpp:188, :205), and a FabSet on the register's
    // layout, which it zeroes, ParallelCopies the scaled flux into, and adds
    // into the register (:220-255). The child's persistent freg_scratch has
    // the register's layout, so per face orientation the same steps run on
    // it, with the scaling moved after the copy: zero, ParallelCopy the raw
    // flux, scale by (s * mult) * area[d], add. This is bit-identical:
    //  - The copy has the same source BoxArray and DistributionMapping, no
    //    source ghosts and no periodic shift, hence the same copy tags, so
    //    where two source boxes share a face the same box wins. A copy moves
    //    values unchanged, so scaling after it gives the values CrseInit
    //    copies.
    //  - area[d] (face_areas) equals face_area[d] bitwise, and the product
    //    is formed in CrseInit's order, (s * mult) * area.
    //  - Scaling and adding are separate kernels, as in CrseInit. A single
    //    `r += s * mult * area` could be contracted into an FMA
    //    (-ffp-contract=fast, nvcc's default --fmad=true), which would skip
    //    the rounding of the product.
    // A face the copy does not reach (none under proper nesting) adds
    // (0 * mult) * area, a zero like CrseInit's, possibly of the other sign.
    //
    // Fine side: FineAdd reads the cached face-area MultiFab of the level
    // (EnsureFluxGeometry above) through the area overload, with the stage
    // weight as the scalar multiplier; it is local and allocation-free, and
    // forms (mult * area) * flux.
    if (child_freg) {
      amrex::FluxRegister *const child_scratch =
          childleveldata->groupdata.at(gi)->freg_scratch.get();
      assert(child_scratch);
      // The coarse step is the reset: the coarse level's first stage zeroes
      // the register below it, and everything that follows (the remaining
      // coarse stages, the child's substeps) only adds.
      if (stage == 1)
        child_freg->setVal(0);
      const int numvars = groupdata.numvars;
      const CCTK_REAL mult = -weight;
      const std::array<CCTK_REAL, dim> area =
          face_areas(patchdata.amrcore->Geom(level));
      for (int d = 0; d < dim; ++d) {
        const auto &flux_groupdata =
            *leveldata.groupdata.at(groupdata.fluxes.at(d));
        const amrex::MultiFab &flux = *flux_groupdata.mfab.at(tl);
        const CCTK_REAL area_d = area[d];
        for (const auto side :
             {amrex::Orientation::low, amrex::Orientation::high}) {
          const amrex::Orientation face(d, side);
          amrex::FabSet &scratch = (*child_scratch)[face];
          amrex::FabSet &reg = (*child_freg)[face];
          assert(scratch.boxArray() == reg.boxArray());
          assert(scratch.DistributionMap() == reg.DistributionMap());
          assert(scratch.nComp() == numvars && reg.nComp() == numvars);
          scratch.setVal(0);
          // (src, ngrow, scomp, dcomp, ncomp): valid faces only, non-periodic
          scratch.copyFrom(flux, 0, 0, 0, numvars);
          // Box by box over validbox(), as AMReX's own FabSet loops do: the
          // tiled MFIter behind amrex::ParallelFor(MultiFab, ...) builds its
          // tiles from the uncollapsed coarsened fine box of a register's
          // BoxArray, not from its one-face-thick boxes, and would run out
          // of bounds.
          for (amrex::FabSetIter fsi(scratch); fsi.isValid(); ++fsi) {
            const amrex::Box &bx = fsi.validbox();
            const auto sfab = scratch.array(fsi);
            amrex::ParallelFor(
                bx, numvars,
                [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) noexcept {
                  sfab(i, j, k, n) = sfab(i, j, k, n) * mult * area_d;
                });
          }
          // The local `reg += scratch` of identical layouts, the same add
          // CrseInit(ADD) performs
          reg.plusFrom(scratch, 0, 0, numvars);
        }
      }
    }

    if (own_freg) {
      for (int d = 0; d < dim; ++d) {
        const auto &flux_groupdata =
            *leveldata.groupdata.at(groupdata.fluxes.at(d));
        const amrex::MultiFab &flux = *flux_groupdata.mfab.at(tl);
        const amrex::MultiFab &area = *leveldata.face_area[d];
        assert(area.ixType() == flux.ixType());
        assert(area.DistributionMap() == flux.DistributionMap());
        own_freg->FineAdd(flux, area, d, 0, 0, groupdata.numvars, +weight);
      }
    }
  } // for gi
}

static void reflux_level(const cGH *cctkGH, const int level) {
  DECLARE_CCTK_PARAMETERS; // verbose

  static Timer timer("Reflux");
  Interval interval(timer);

  for (const auto &patchdata : ghext->patchdata) {
    if (level + 1 >= int(patchdata.leveldata.size()))
      continue;
    const int patch = patchdata.patch;
    const auto &leveldata = patchdata.leveldata.at(level);
    const auto &fineleveldata = patchdata.leveldata.at(level + 1);
    const amrex::Geometry &geom = patchdata.amrcore->Geom(level);
    const active_levels_t coarse_level(level, level + 1, patch, patch + 1);
    const int tl = 0;

    for (int gi = 0; gi < int(leveldata.groupdata.size()); ++gi) {
      // only grid functions live on levels
      if (!leveldata.groupdata.at(gi))
        continue;
      const auto &groupdata = *leveldata.groupdata.at(gi);
      if (groupdata.mfab.empty())
        continue;
      const auto &finegroupdata = *fineleveldata.groupdata.at(gi);
      assert(!finegroupdata.mfab.empty());

      // If the group has a flux register on the fine level
      if (!finegroupdata.freg)
        continue;

      if (verbose)
        CCTK_VINFO("Reflux: applying the flux register of levels (%d, %d) to "
                   "%s at iteration %d",
                   level, level + 1, groupdata.groupname.c_str(),
                   cctkGH->cctk_iteration);

      require(coarse_level,
              {.gi = gi,
               .tl = tl,
               .where = make_valid_int(),
               .event = "Reflux before refluxing: Coarse level data"});

      // The cached cell volume; built on the first registered group met, a
      // no-op afterwards, so a level without a register never allocates
      EnsureFluxGeometry(patch, level);

      const write_scope_t scope =
          open_write(coarse_level, {.gi = gi,
                                    .tl = tl,
                                    .fills = make_valid_int(),
                                    .keeps = make_valid_all(),
                                    .event = "Reflux"});
      amrex::MultiFab &state = *groupdata.mfab.at(tl);
      const amrex::MultiFab &volume = *leveldata.cell_volume;
      assert(volume.ixType() == state.ixType());
      assert(volume.DistributionMap() == state.DistributionMap());
      finegroupdata.freg->Reflux(state, volume, 1.0, 0, 0, groupdata.numvars,
                                 geom);
    } // for gi
  } // for patchdata
}

// See subcycling.hxx for the contract, including why no ghost is repaired
// here. Every level pair (level, level + 1) with both levels in [min_level,
// max_level), finest pair first, so that a level is corrected by its
// child's register before it is restricted onto (or, under subcycling,
// before its own register is applied to) its parent. The window is the
// alignment test under subcycling: with three levels, [1,3) refluxes the
// pair (1,2) only, and the later [0,3) refluxes (1,2) and then (0,1).
// Without subcycling the caller passes every level. Applying a register
// reads only the register and writes only coarse interior cells, so the
// cascade depends on no ghost.
void Reflux(const cGH *cctkGH, const int min_level, const int max_level) {
  if (!ghext->do_reflux)
    return;
  assert(min_level >= 0 && max_level <= ghext->num_levels());
  for (int level = max_level - 2; level >= min_level; --level)
    reflux_level(cctkGH, level);
}

// =======================================================================
// Restrict
// =======================================================================

static void restrict_level(const cGH *cctkGH, const int level,
                           const std::vector<int> &groups,
                           const valid_t coarse_keeps) {
  static Timer timer("Restrict");
  Interval interval(timer);

  int gi_regrid_error = -1;
  gi_regrid_error = CCTK_GroupIndex("CarpetXRegrid::regrid_error");
  assert(gi_regrid_error >= 0);

  for (const auto &patchdata : ghext->patchdata) {
    const int patch = patchdata.patch;
    if (level + 1 < int(patchdata.leveldata.size())) {
      auto &leveldata = patchdata.leveldata.at(level);
      const auto &fineleveldata = patchdata.leveldata.at(level + 1);
      const active_levels_t active_levels(level, level + 1, patch, patch + 1);
      const active_levels_t active_fine_levels(level + 1, level + 2, patch,
                                               patch + 1);

      for (const int gi : groups) {
        cGroup group;
        int ierr = CCTK_GroupData(gi, &group);
        assert(!ierr);

        assert(group.grouptype == CCTK_GF);

        const auto &groupdata = *leveldata.groupdata.at(gi);
        assert(!groupdata.mfab.empty());
        const auto &finegroupdata = *fineleveldata.groupdata.at(gi);
        assert(!finegroupdata.mfab.empty());
        const amrex::IntVect reffact{2, 2, 2};

        // Don't restrict the regridding error
        if (gi == gi_regrid_error)
          continue;

        const int restrict_tl = timelevels_but_oldest(gi);
        for (int tl = 0; tl < restrict_tl; ++tl) {

          require(active_fine_levels,
                  {.gi = gi,
                   .tl = tl,
                   .where = make_valid_int(),
                   .event = "Restrict on fine level before restricting"});
          require(active_levels,
                  {.gi = gi,
                   .tl = tl,
                   .where = make_valid_int(),
                   .event = "Restrict on coarse level before restricting"});
          const write_scope_t scope =
              open_write(active_levels, {.gi = gi,
                                         .tl = tl,
                                         .fills = make_valid_int(),
                                         .keeps = coarse_keeps,
                                         .event = "Restrict"});

#if 1
          {
            static Timer timer("Restrict::average_down");
            Interval interval(timer);
            // rank: 0: vertex, 1: edge, 2: face, 3: volume
            int rank = 0;
            for (int d = 0; d < dim; ++d)
              rank += groupdata.indextype.at(d);
            switch (rank) {
            case 0:
              average_down_nodal(*finegroupdata.mfab.at(tl),
                                 *groupdata.mfab.at(tl), reffact);
              break;
            case 1:
              average_down_edges(*finegroupdata.mfab.at(tl),
                                 *groupdata.mfab.at(tl), reffact);
              break;
            case 2:
              average_down_faces(*finegroupdata.mfab.at(tl),
                                 *groupdata.mfab.at(tl), reffact);
              break;
            case 3:
              average_down(*finegroupdata.mfab.at(tl), *groupdata.mfab.at(tl),
                           0, groupdata.numvars, reffact);
              break;
            default:
              assert(0);
            }
          }
#endif

        } // for tl
      } // for gi
    } // if level exists
  } // for patchdata
}

void Restrict(const cGH *cctkGH, const int min_level, const int max_level,
              const std::vector<int> &groups) {
  assert(min_level >= 0 && max_level <= ghext->num_levels());
  if (groups.empty())
    return;
  for (int level = max_level - 2; level >= min_level; --level)
    restrict_level(cctkGH, level, groups, make_valid_int());
}

void RestrictNoPoison(const cGH *cctkGH, const int min_level,
                      const int max_level, const std::vector<int> &groups) {
  assert(min_level >= 0 && max_level <= ghext->num_levels());
  if (groups.empty())
    return;
  for (int level = max_level - 2; level >= min_level; --level)
    restrict_level(cctkGH, level, groups, make_valid_all());
}

} // namespace CarpetX
