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
#include <memory>
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

static void sync_multipatch_postcheck(const cGH *cctkGH,
                                      const std::vector<int> &groups,
                                      const char *label) {
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

    for (const int gi : groups) {
      const auto &patchdata0 = ghext->patchdata.at(0);
      const auto &leveldata0 = patchdata0.leveldata.at(0);
      const auto &groupdata0 = *leveldata0.groupdata.at(gi);
      assert(!groupdata0.mfab.empty());
      // We always sync all directions.
      // If there is more than one time level, then we don't sync the
      // oldest.
      // TODO: during evolution, sync only one time level
      const int ntls0 = groupdata0.mfab.size();
      const int sync_tl0 = ntls0 > 1 ? ntls0 - 1 : ntls0;

      for (int tl = 0; tl < sync_tl0; ++tl)
        for (int vi = 0; vi < groupdata0.numvars; ++vi)
          check_valid_gf(*active_levels, gi, vi, tl, [label]() {
            return std::string(label) + " after syncing";
          });

    } // for gi

  } else {
    assert(ghext->num_patches() == 1);
  }
}

// Abort unless the parent of `leveldata` has a valid interior for every
// variable of group gi in timelevel tl, because a prolongation into
// `leveldata` is about to read it. The message names the parent's level and
// timelevel. When gi is not checkpointed, it ends with the rule that was
// broken: for a past timelevel always, and for the current timelevel only
// while the parent has not run its end of step since the run was recovered
// (GHExt::recovered_iteration).
static void
error_if_parent_invalid(const GHExt::PatchData::LevelData &leveldata,
                        const int gi, const int tl, const char *const label) {
  assert(leveldata.level > 0);
  const auto &restrict groupdata = *leveldata.groupdata.at(gi);
  const auto &restrict coarseleveldata =
      ghext->patchdata.at(leveldata.patch).leveldata.at(leveldata.level - 1);
  const auto &restrict coarsegroupdata = *coarseleveldata.groupdata.at(gi);
  assert(!coarsegroupdata.mfab.empty());
  assert(coarsegroupdata.numvars == groupdata.numvars);
  assert(tl >= 0 && tl < int(coarsegroupdata.mfab.size()));

  const bool recovering =
      ghext->recovered_iteration >= 0 &&
      coarseleveldata.last_finalized_iteration <= ghext->recovered_iteration;

  const std::function<std::string()> msg = [label]() {
    return std::string(label);
  };
  const std::function<std::string()> hint = [&coarsegroupdata, tl,
                                             recovering]() -> std::string {
    if (coarsegroupdata.do_checkpoint)
      return "";
    // A past timelevel that another level reads must be checkpointed,
    // whatever invalidated it this time: no routine can recompute it, and
    // only a checkpoint carries it across a recovery or a regrid
    if (tl > 0)
      return "  Group " + coarsegroupdata.groupname +
             " is not checkpointed, so a recovery, or a regrid that remakes "
             "its level, can lose its past timelevels, which no routine can "
             "recompute.\n"
             "  Past timelevels read by another level must be checkpointed.\n";
    // Outside recovery the driver cannot tell why a current timelevel is
    // invalid (a missing routine, a missing POSTREGRID recomputation)
    if (recovering)
      return "  This run was recovered and group " + coarsegroupdata.groupname +
             " is not checkpointed.\n"
             "  Groups read by another level must be checkpointed or "
             "recomputed in CCTK_POST_RECOVER_VARIABLES.\n";
    return "";
  };
  for (int vi = 0; vi < coarsegroupdata.numvars; ++vi)
    error_if_invalid(coarsegroupdata, vi, tl, make_valid_int(), msg, hint);
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

static int
SyncGroupsByDirIProlongateOnly_impl(const cGH *restrict cctkGH, int numgroups,
                                    const int *groups0, const int *directions,
                                    const bool prolongate_on_same_iteration,
                                    const int tl_arg = -1) {
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

  // We need to loop over groups, patches, and levels in a definite
  // order so that AMReX's communication pattern does not get
  // confused. Therefore all the loops here are serial. The only
  // parallelization happens within AMReX and within our boundary
  // conditions. This is not efficient.

  task_manager tasks1;
  task_manager tasks2;
  task_manager tasks3;

  for (const int gi : groups) {
    active_levels->loop_serially([&](auto &restrict leveldata) {
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

  for (const int gi : groups) {
    active_levels->loop_serially([&](auto &restrict leveldata) {
      if (!prolongates_into(leveldata))
        return;
      auto &restrict groupdata = *leveldata.groupdata.at(gi);
      const auto [tl_lo, tl_hi] = timelevels_to_fill(groupdata);
      for (int tl = tl_lo; tl < tl_hi; ++tl) {
        for (int vi = 0; vi < groupdata.numvars; ++vi) {
          groupdata.valid.at(tl).at(vi).set_ghosts(true, []() {
            return "SyncGroupsByDirIProlongateOnly after prolongating: "
                   "Mark ghost zones as valid";
          });
          if (groupdata.all_faces_have_symmetries_or_boundaries())
            groupdata.valid.at(tl).at(vi).set_outer(true, []() {
              return "SyncGroupsByDirIProlongateOnly after prolongating: "
                     "Mark outer boundaries as valid";
            });
        }
      }
    });
  }

  sync_multipatch_postcheck(cctkGH, groups, "SyncGroupsByDirIProlongateOnly");

  assert(sync_active);

  return numgroups; // number of groups synchronized
}

int SyncGroupsByDirI(const cGH *restrict cctkGH, int numgroups,
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

  // Skip groups that have valid ghosts and boundaries
  if (CCTK_EQUALS(presync_mode, "presync-only")) {
    active_levels->loop_serially([&](auto &restrict leveldata) {
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
    const int min_level = active_levels->min_level;
    // Every level of the window from its child, including the window's top
    // level from the next finer level, which lies outside the window after a
    // regrid (BASEGRID/POSTREGRID SYNCs, whose window is the modified levels)
    const int max_level =
        std::min(active_levels->max_level + 1, ghext->num_levels());
    // The do_restrict parameter governs only restriction the driver starts
    assert(do_restrict || max_level - min_level < 2);
    // Restrict only the groups the restrict tag allows; the ghost fill below
    // still covers all of `groups`
    Restrict(cctkGH, min_level, max_level, sync_restrict_groups(groups));
  }

  static const bool have_multipatch_boundaries =
      CCTK_IsFunctionAliased("MultiPatch_Interpolate");

  // Check preconditions
  for (const int gi : groups) {
    const auto &patchdata0 = ghext->patchdata.at(0);
    const auto &leveldata0 = patchdata0.leveldata.at(0);
    const auto &groupdata0 = *leveldata0.groupdata.at(gi);
    assert(!groupdata0.mfab.empty());
    // We always sync all directions.
    // If there is more than one time level, then we don't sync the
    // oldest.
    // TODO: during evolution, sync only one time level
    const int ntls0 = groupdata0.mfab.size();
    const int sync_tl0 = ntls0 > 1 ? ntls0 - 1 : ntls0;

    active_levels->loop_serially([&](auto &restrict leveldata) {
      auto &restrict groupdata = *leveldata.groupdata.at(gi);
      assert(!groupdata.mfab.empty());

      if (leveldata.level > 0)
        for (int tl = 0; tl < sync_tl0; ++tl)
          error_if_parent_invalid(
              leveldata, gi, tl,
              "SyncGroupsByDirI on coarse level before prolongation");

      for (int tl = 0; tl < sync_tl0; ++tl) {
        for (int vi = 0; vi < groupdata.numvars; ++vi) {
          // Synchronization only uses the interior
          error_if_invalid(groupdata, vi, tl, make_valid_int(),
                           []() { return "SyncGroupsByDirI before syncing"; });
          groupdata.valid.at(tl).at(vi).set_invalid(make_valid_ghosts(), []() {
            return "SyncGroupsByDirI before syncing: "
                   "Mark ghost zones as invalid";
          });
        }
      } // for tl
    });

    active_levels_t active_fine_levels = *active_levels;
    using std::max;
    active_fine_levels.min_level = max(active_fine_levels.min_level, 1);
    for (int tl = 0; tl < sync_tl0; ++tl) {
      for (int vi = 0; vi < groupdata0.numvars; ++vi) {
        check_valid_gf(active_fine_levels, gi, vi, tl, []() {
          return "SyncGroupsByDirI on coarse level before prolongation";
        });
        poison_invalid_gf(*active_levels, gi, vi, tl);
        check_valid_gf(*active_levels, gi, vi, tl,
                       []() { return "SyncGroupsByDirI before syncing"; });
      }
    } // for tl
  } // for gi

  // We need to loop over groups, patches, and levels in a definite
  // order so that AMReX's communication pattern does not get
  // confused. Therefore all the loops here are serial. The only
  // parallelization happens within AMReX and within our boundary
  // conditions. This is not efficient.

  task_manager tasks1;
  task_manager tasks2;
  task_manager tasks3;

  for (const int gi : groups) {
    active_levels->loop_serially([&](auto &restrict leveldata) {
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

  // Check postconditions
  for (const int gi : groups) {
    const auto &patchdata0 = ghext->patchdata.at(0);
    const auto &leveldata0 = patchdata0.leveldata.at(0);
    const auto &groupdata0 = *leveldata0.groupdata.at(gi);
    assert(!groupdata0.mfab.empty());
    // We always sync all directions.
    // If there is more than one time level, then we don't sync the
    // oldest.
    // TODO: during evolution, sync only one time level
    const int ntls0 = groupdata0.mfab.size();
    const int sync_tl0 = ntls0 > 1 ? ntls0 - 1 : ntls0;

    active_levels->loop_serially([&](auto &restrict leveldata) {
      auto &restrict groupdata = *leveldata.groupdata.at(gi);
      assert(!groupdata.mfab.empty());

      for (int tl = 0; tl < sync_tl0; ++tl) {
        for (int vi = 0; vi < groupdata.numvars; ++vi) {
          groupdata.valid.at(tl).at(vi).set_ghosts(true, []() {
            return "SyncGroupsByDirI after syncing: "
                   "Mark ghost zones as valid";
          });
          if (groupdata.all_faces_have_symmetries_or_boundaries())
            groupdata.valid.at(tl).at(vi).set_outer(true, []() {
              return "SyncGroupsByDirI after syncing: "
                     "Mark outer boundaries as valid";
            });
        }
      } // for tl
    });

    for (int tl = 0; tl < sync_tl0; ++tl) {
      for (int vi = 0; vi < groupdata0.numvars; ++vi) {
        poison_invalid_gf(*active_levels, gi, vi, tl);
        // TODO: Check after applying multi-patch boundaries
        if (!have_multipatch_boundaries)
          check_valid_gf(*active_levels, gi, vi, tl,
                         []() { return "SyncGroupsByDirI after syncing"; });
      }
    } // for tl
  } // for gi

  sync_multipatch_postcheck(cctkGH, groups, "SyncGroupsByDirI");

  assert(sync_active);

  return numgroups; // number of groups synchronized
}

// Abort if a sync is requested while the driver traverses a bin on the
// end-of-step window (GHExt::tail_bin: CCTK_POSTSTEP, CCTK_ANALYSIS,
// CCTK_CHECKPOINT or CCTK_TERMINATE) under subcycling, unless the window holds
// every level.
// The levels of the window share one time, so a sync among them is sound. A
// window that starts above level 0 has a coarser neighbour outside it that
// has stepped ahead and has not been finalized, and syncing the window's
// coarsest level would fill its coarse-fine ghosts from that level's data of
// another time. This is the condition under which interpolation across
// levels aborts too. Under subcycling every sync passes through
// SyncGroupsByDirISubcycling: a SYNC clause, an explicit CCTK_SyncGroup*
// call, and a sync in a nested schedule group (Initialise rejects
// presync-only mode, whose syncs would bypass it). The check runs before the
// groups are filtered, so that a SYNC of a group without storage or of a
// non-GF group aborts too.
static void error_if_sync_in_tail(const cGH *restrict const cctkGH,
                                  const int numgroups, const int *groups0) {
  if (!ghext->tail_bin || numgroups == 0)
    return;
  assert(active_levels);
  const int min_level = active_levels->min_level;
  const int max_level = active_levels->max_level;
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

int SyncGroupsByDirISubcycling(const cGH *restrict cctkGH, int numgroups,
                               const int *groups0, const int *directions) {
  DECLARE_CCTK_PARAMETERS;

  assert(in_global_mode(cctkGH) || in_level_mode(cctkGH));

  error_if_sync_in_tail(cctkGH, numgroups, groups0);

  mark_sync_active marked;

  static Timer timer("Sync");
  Interval interval(timer);

  assert(cctkGH);
  assert(numgroups >= 0);
  assert(groups0);

  sync_log_groups("SyncGroupsBySubcycling", numgroups, groups0);

  std::vector<int> groups = sync_filter_groups(numgroups, groups0);

  static const bool have_multipatch_boundaries =
      CCTK_IsFunctionAliased("MultiPatch_Interpolate");

  // We need to loop over groups, patches, and levels in a definite
  // order so that AMReX's communication pattern does not get
  // confused. Therefore all the loops here are serial. The only
  // parallelization happens within AMReX and within our boundary
  // conditions. This is not efficient.

  task_manager tasks1;
  task_manager tasks2;
  task_manager tasks3;

  for (const int gi : groups) {
    active_levels->loop_serially([&](auto &restrict leveldata) {
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
          for (int tl = 0; tl < sync_tl; ++tl)
            for (int vi = 0; vi < groupdata.numvars; ++vi)
              error_if_invalid(
                  groupdata, vi, tl, make_valid_ghosts(),
                  []() {
                    return "SyncGroupsByDirISubcycling before syncing an "
                           "integrated group on a refined level";
                  },
                  []() {
                    return "  With CarpetX::use_subcycling = yes the time "
                           "integrator owns the coarse-fine ghosts of the "
                           "groups it integrates at iteration > 0: "
                           "CarpetX::rk_stage_t::fill_next_boundary after "
                           "each RK stage, and CarpetX::rk_recover_boundary "
                           "or CarpetX::prolongate_aligned after a recovery, "
                           "fill and mark them before any SYNC.\n"
                           "  The SYNC copies only the same-level ghosts of "
                           "such a group and requires its ghosts to be marked "
                           "valid already.\n";
                  });
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

          // Each fill below first checks that every parent timelevel it reads
          // has a valid interior. The fills are only queued here, so the
          // checks run before any of them. They hold in every run and have no
          // opt-out; without them an invalid parent is caught only by the
          // poison scan after the sync, on the child, and not at all without
          // poisoning.
          const char *const label =
              "SyncGroupsByDirISubcycling on coarse level before prolongation";

          for (int tl = 0; tl < sync_tl; ++tl) {
            // Only tl=0 is the "new" coarse snapshot whose old partner is tl=1.
            const bool do_blend = blend && tl == 0;
            error_if_parent_invalid(leveldata, gi, tl, label);
            if (do_blend)
              error_if_parent_invalid(leveldata, gi, 1, label);
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

  for (const int gi : groups) {
    const auto &patchdata0 = ghext->patchdata.at(0);
    const auto &leveldata0 = patchdata0.leveldata.at(0);
    const auto &groupdata0 = *leveldata0.groupdata.at(gi);
    assert(!groupdata0.mfab.empty());
    // We always sync all directions.
    // If there is more than one time level, then we don't sync the
    // oldest.
    // TODO: during evolution, sync only one time level
    const int ntls0 = groupdata0.mfab.size();
    const int sync_tl0 = ntls0 > 1 ? ntls0 - 1 : ntls0;

    active_levels->loop_serially([&](auto &restrict leveldata) {
      auto &restrict groupdata = *leveldata.groupdata.at(gi);
      assert(!groupdata.mfab.empty());

      const bool integrated_subiter =
          integrator_owns_coarse_fine_ghosts(leveldata, gi);

      for (int tl = 0; tl < sync_tl0; ++tl) {
        for (int vi = 0; vi < groupdata.numvars; ++vi) {
          if (!integrated_subiter) {
            groupdata.valid.at(tl).at(vi).set_ghosts(true, []() {
              return "SyncGroupsByDirISubcycling after syncing: "
                     "Mark ghost zones as valid";
            });
          }
          if (groupdata.all_faces_have_symmetries_or_boundaries())
            groupdata.valid.at(tl).at(vi).set_outer(true, []() {
              return "SyncGroupsByDirISubcycling after syncing: "
                     "Mark outer boundaries as valid";
            });
        }
      } // for tl
    });

    for (int tl = 0; tl < sync_tl0; ++tl) {
      for (int vi = 0; vi < groupdata0.numvars; ++vi) {
        poison_invalid_gf(*active_levels, gi, vi, tl);
        // TODO: Check after applying multi-patch boundaries
        if (!have_multipatch_boundaries)
          check_valid_gf(*active_levels, gi, vi, tl, []() {
            return "SyncGroupsByDirISubcycling after syncing";
          });
      }
    } // for tl
  } // for gi

  sync_multipatch_postcheck(cctkGH, groups, "SyncGroupsByDirISubcycling");

  assert(sync_active);

  return numgroups; // number of groups synchronized
}

int SyncGroupsByDirIProlongateOnly(const cGH *restrict cctkGH, int numgroups,
                                   const int *groups0, const int *directions,
                                   const int tl) {
  return SyncGroupsByDirIProlongateOnly_impl(cctkGH, numgroups, groups0,
                                             directions, false, tl);
}

int SyncGroupsByDirIProlongateOnlyAligned(const cGH *restrict cctkGH,
                                          int numgroups, const int *groups0,
                                          const int *directions, const int tl) {
  return SyncGroupsByDirIProlongateOnly_impl(cctkGH, numgroups, groups0,
                                             directions, true, tl);
}

int SyncGroupsByDirIGhostOnly(const cGH *restrict cctkGH, int numgroups,
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
    active_levels->loop_serially([&](auto &restrict leveldata) {
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

  sync_multipatch_postcheck(cctkGH, groups, "SyncGroupsByDirIGhostOnly");

  assert(sync_active);

  return numgroups; // number of groups synchronized
}

// Right after the end-of-step restriction, refill the coarse-fine ghosts of
// every active level aligned with its parent by spatial prolongation from the
// restricted parent. The groups are those whose ghosts there are not yet a
// prolongation of the parent's current state:
//  - every restricted group, since the restriction changed the parent;
//  - under subcycling, every integrated group, restricted or not, since
//    ODESolvers left its ghosts there at the dense output (the parent's RK
//    update before its ODESolvers_PostStep) to save communication. Refilling
//    them gives every evolved group the same last fill, the parent's final
//    state, which is what recovery repeats. The callers of Restrict still
//    leave restrict="no" groups out (see the selectors), so the driver never
//    overwrites their coarse interiors.
// The selector reprolongate_groups applies this rule.
void ProlongateAfterRestriction(const cGH *cctkGH) {
  const std::vector<int> groups = reprolongate_groups();
  SyncGroupsByDirIProlongateOnly_impl(cctkGH, groups.size(), groups.data(),
                                      nullptr, true);
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

    // The fluxes of this stage must be valid on the interior: the coarse
    // side reads the faces under the child's boxes, FineAdd the faces on the
    // boundary of this level's own boxes, both interior for a face-centred
    // group.
    for (int d = 0; d < dim; ++d) {
      const auto &flux_groupdata =
          *leveldata.groupdata.at(groupdata.fluxes.at(d));
      assert(!flux_groupdata.mfab.empty());
      for (int vi = 0; vi < groupdata.numvars; ++vi)
        error_if_invalid(flux_groupdata, vi, tl, make_valid_int(), [&]() {
          std::ostringstream buf;
          buf << "AccumulateFluxes: flux of " << groupdata.groupname
              << " in direction " << d << " at RK stage " << stage;
          return buf.str();
        });
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

// Apply the flux register of the pair (level, level + 1) to the coarse
// state on `level`, on every patch that has level + 1: state += register /
// volume on the coarse cells next to the coarse-fine boundary. The register
// was filled by AccumulateFluxes over the coarse step and, under subcycling,
// the fine substeps; nothing is read from the flux groups here. Reached
// through Reflux(cctkGH, min_level, max_level) below.
//
// Only the interior of time level 0 changes (FluxRegister::Reflux writes
// the valid boxes only). No ghost, outer boundary point or validity flag is
// touched: the copies of the corrected cells in neighbouring boxes' ghosts
// (including periodic images), in the outer boundary and in other patches'
// inter-patch ghosts keep their pre-reflux values until the
// ODESolvers_PostStep that follows every reflux syncs the state (see
// subcycling.hxx).
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
    const int tl = 0;

    // Groups whose register was applied on this patch
    std::vector<int> refluxed_groups;

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

      // The coarse state must be valid on the interior
      for (int vi = 0; vi < groupdata.numvars; ++vi)
        error_if_invalid(groupdata, vi, tl, make_valid_int(), []() {
          return "Reflux before refluxing: Coarse level data";
        });

      // The cached cell volume; built on the first registered group met, a
      // no-op afterwards, so a level without a register never allocates
      EnsureFluxGeometry(patch, level);

      // The volume overload reads the cached cell-volume MultiFab of the
      // level (EnsureFluxGeometry above) instead of synthesizing one per
      // call. What AMReX still allocates per call: one zeroed face-centred
      // flux MultiFab on the state's layout per register face (six), into
      // which the register is parallel-copied before the kernel runs.
      amrex::MultiFab &state = *groupdata.mfab.at(tl);
      const amrex::MultiFab &volume = *leveldata.cell_volume;
      assert(volume.ixType() == state.ixType());
      assert(volume.DistributionMap() == state.DistributionMap());
      finegroupdata.freg->Reflux(state, volume, 1.0, 0, 0, groupdata.numvars,
                                 geom);
      refluxed_groups.push_back(gi);
    } // for gi

    // Re-check the coarse state that was just modified, in whatever
    // regions are flagged valid
    const active_levels_t coarse_level(level, level + 1, patch, patch + 1);
    for (const int gi : refluxed_groups) {
      const auto &groupdata = *leveldata.groupdata.at(gi);
      for (int vi = 0; vi < groupdata.numvars; ++vi)
        check_valid_gf(coarse_level, gi, vi, tl, []() {
          return "Reflux after refluxing: Coarse level data";
        });
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

// Restrict `groups` from level + 1 onto level on every patch that has
// level + 1: the coarse interior points the fine level covers get the
// restricted fine values, in every timelevel except the oldest (or in the
// only one). Both interiors must be valid. With do_validity_tracking, the
// coarse ghosts and outer boundary are then marked invalid; without it their
// flags stay set over stale values, and the caller refreshes them. Reached
// through Restrict and RestrictNoPoison below.
static void restrict_level(const cGH *cctkGH, const int level,
                           const std::vector<int> &groups,
                           const bool do_validity_tracking) {
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

        auto &groupdata = *leveldata.groupdata.at(gi);
        assert(!groupdata.mfab.empty());
        const auto &finegroupdata = *fineleveldata.groupdata.at(gi);
        assert(!finegroupdata.mfab.empty());
        const amrex::IntVect reffact{2, 2, 2};

        // Don't restrict the regridding error
        if (gi == gi_regrid_error)
          continue;

        // If there is more than one time level, then we don't restrict the
        // oldest.
        // TODO: during evolution, restrict only one time level
        int ntls = groupdata.mfab.size();
        int restrict_tl = ntls > 1 ? ntls - 1 : ntls;
        for (int tl = 0; tl < restrict_tl; ++tl) {

          for (int vi = 0; vi < groupdata.numvars; ++vi) {

            // Restriction only uses the interior
            error_if_invalid(finegroupdata, vi, tl, make_valid_int(), []() {
              return "Restrict on fine level before restricting";
            });
            poison_invalid_gf(active_fine_levels, gi, vi, tl);
            check_valid_gf(active_fine_levels, gi, vi, tl, []() {
              return "Restrict on fine level before restricting";
            });
            error_if_invalid(groupdata, vi, tl, make_valid_int(), []() {
              return "Restrict on coarse level before restricting";
            });
            poison_invalid_gf(active_levels, gi, vi, tl);
            check_valid_gf(active_levels, gi, vi, tl, []() {
              return "Restrict on coarse level before restricting";
            });
          }

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

          if (do_validity_tracking) {
            // TODO: Also remember old why_valid for interior?
            for (int vi = 0; vi < groupdata.numvars; ++vi) {
              // Should we mark ghosts and maybe outer boundaries as
              // valid as well?
              groupdata.valid.at(tl).at(vi).set_invalid(
                  make_valid_outer() | make_valid_ghosts(),
                  []() { return "Restrict"; });
              poison_invalid_gf(active_levels, gi, vi, tl);
              check_valid_gf(active_levels, gi, vi, tl, []() {
                return "Restrict on coarse level after restricting";
              });
            }
          }

        } // for tl
      } // for gi
    } // if level exists
  } // for patchdata
}

// Restrict exactly `groups`: every level pair (level, level + 1) with both
// levels in [min_level, max_level), finest pair first, each level once and
// on every patch. Applies no group rule and reads no parameter; the callers
// select the groups (see the selectors above). Marks the coarse ghosts and
// outer boundaries invalid.
void Restrict(const cGH *cctkGH, const int min_level, const int max_level,
              const std::vector<int> &groups) {
  assert(min_level >= 0 && max_level <= ghext->num_levels());
  if (groups.empty())
    return;
  for (int level = max_level - 2; level >= min_level; --level)
    restrict_level(cctkGH, level, groups, /*do_validity_tracking=*/true);
}

// Like Restrict, but leaves the validity flags alone (see restrict_level)
void RestrictNoPoison(const cGH *cctkGH, const int min_level,
                      const int max_level, const std::vector<int> &groups) {
  assert(min_level >= 0 && max_level <= ghext->num_levels());
  if (groups.empty())
    return;
  for (int level = max_level - 2; level >= min_level; --level)
    restrict_level(cctkGH, level, groups, /*do_validity_tracking=*/false);
}

} // namespace CarpetX
