#include "schedule.hxx"
#include "driver.hxx"
#include "fillpatch.hxx"
#include "subcycling.hxx"
#include "sync_restrict_internal.hxx"
#include "task_manager.hxx"
#include "timer.hxx"
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
#include <vector>

namespace CarpetX {

bool sync_active = false;

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
      const int ntls0 = groupdata0.mfab.size();
      const int sync_tl0 = ntls0 > 1 ? ntls0 - 1 : ntls0;

      for (int tl = 0; tl < sync_tl0; ++tl)
        for (int vi = 0; vi < groupdata0.numvars; ++vi)
          check_valid_gf(*active_levels, gi, vi, tl, [label]() {
            return std::string(label) + " after syncing";
          });

    }

  } else {
    assert(ghext->num_patches() == 1);
  }
}

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
    if (tl > 0)
      return "  Group " + coarsegroupdata.groupname +
             " is not checkpointed, so a recovery, or a regrid that remakes "
             "its level, can lose its past timelevels, which no routine can "
             "recompute.\n"
             "  Past timelevels read by another level must be checkpointed.\n";
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

std::vector<int> collect_restrictable_groups() {
  const int numgroups = CCTK_NumGroups();
  std::vector<int> groups;
  groups.reserve(numgroups);
  const auto &patchdata0 = ghext->patchdata.at(0);
  const auto &leveldata0 = patchdata0.leveldata.at(0);
  for (const auto &groupdataptr : leveldata0.groupdata) {
    if (groupdataptr) {
      auto &restrict groupdata = *groupdataptr;
      if (groupdata.mfab.empty())
        continue;
      if (groupdata.do_restrict)
        groups.push_back(groupdata.groupindex);
    }
  }
  return groups;
}

static std::vector<int> reprolongate_groups() {
  const int numgroups = CCTK_NumGroups();
  std::vector<int> groups;
  groups.reserve(numgroups);
  const auto &patchdata0 = ghext->patchdata.at(0);
  const auto &leveldata0 = patchdata0.leveldata.at(0);
  for (const auto &groupdataptr : leveldata0.groupdata) {
    if (groupdataptr) {
      auto &restrict groupdata = *groupdataptr;
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

static std::vector<int> sync_restrict_groups(const std::vector<int> &groups) {
  std::vector<int> restrict_groups;
  restrict_groups.reserve(groups.size());
  const auto &patchdata0 = ghext->patchdata.at(0);
  const auto &leveldata0 = patchdata0.leveldata.at(0);
  for (const int gi : groups) {
    const auto &groupdataptr = leveldata0.groupdata.at(gi);
    assert(groupdataptr);
    if (groupdataptr->restrict_on_sync)
      restrict_groups.push_back(gi);
  }
  return restrict_groups;
}

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

  task_manager tasks1;
  task_manager tasks2;
  task_manager tasks3;

  for (const int gi : groups) {
    active_levels->loop_serially([&](auto &restrict leveldata) {
      auto &restrict groupdata = *leveldata.groupdata.at(gi);
      assert(!groupdata.mfab.empty());

      const int ntls = groupdata.mfab.size();
      const int tl_lo = (tl_arg < 0) ? 0 : tl_arg;
      const int tl_hi =
          (tl_arg < 0) ? (ntls > 1 ? ntls - 1 : ntls) : (tl_arg + 1);
      assert(tl_lo >= 0 && tl_hi <= ntls);

      if (leveldata.level == 0) {
        return;
      }

      const int level = leveldata.level;
      const auto &restrict coarseleveldata =
          ghext->patchdata.at(leveldata.patch).leveldata.at(level - 1);

      if (ghext->use_subcycling && prolongate_on_same_iteration) {
        if (leveldata.iteration != coarseleveldata.iteration)
          return;
      }

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

      }
    });
  }

  tasks1.run_tasks_serially();
  synchronize();
  tasks2.run_tasks_serially();
  synchronize();
  tasks3.run_tasks_serially();
  synchronize();

  sync_multipatch_postcheck(cctkGH, groups, "SyncGroupsByDirIProlongateOnly");

  assert(sync_active);

  return numgroups;
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
    const int max_level =
        std::min(active_levels->max_level + 1, ghext->num_levels());
    assert(do_restrict || max_level - min_level < 2);
    Restrict(cctkGH, min_level, max_level, sync_restrict_groups(groups));
  }

  static const bool have_multipatch_boundaries =
      CCTK_IsFunctionAliased("MultiPatch_Interpolate");

  for (const int gi : groups) {
    const auto &patchdata0 = ghext->patchdata.at(0);
    const auto &leveldata0 = patchdata0.leveldata.at(0);
    const auto &groupdata0 = *leveldata0.groupdata.at(gi);
    assert(!groupdata0.mfab.empty());
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
          error_if_invalid(groupdata, vi, tl, make_valid_int(),
                           []() { return "SyncGroupsByDirI before syncing"; });
          groupdata.valid.at(tl).at(vi).set_invalid(make_valid_ghosts(), []() {
            return "SyncGroupsByDirI before syncing: "
                   "Mark ghost zones as invalid";
          });
        }
      }
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
    }
  }

  task_manager tasks1;
  task_manager tasks2;
  task_manager tasks3;

  for (const int gi : groups) {
    active_levels->loop_serially([&](auto &restrict leveldata) {
      auto &restrict groupdata = *leveldata.groupdata.at(gi);
      assert(!groupdata.mfab.empty());

      const int ntls = groupdata.mfab.size();
      const int sync_tl = ntls > 1 ? ntls - 1 : ntls;

      if (leveldata.level == 0) {
        for (int tl = 0; tl < sync_tl; ++tl) {
          tasks1.submit_serially([&tasks2, &leveldata, &groupdata, tl]() {
            FillPatch_Sync(tasks2, groupdata, *groupdata.mfab.at(tl),
                           ghext->patchdata.at(leveldata.patch)
                               .amrcore->Geom(leveldata.level));
          });
        }

      } else {
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

  for (const int gi : groups) {
    const auto &patchdata0 = ghext->patchdata.at(0);
    const auto &leveldata0 = patchdata0.leveldata.at(0);
    const auto &groupdata0 = *leveldata0.groupdata.at(gi);
    assert(!groupdata0.mfab.empty());
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
      }
    });

    for (int tl = 0; tl < sync_tl0; ++tl) {
      for (int vi = 0; vi < groupdata0.numvars; ++vi) {
        poison_invalid_gf(*active_levels, gi, vi, tl);
        if (!have_multipatch_boundaries)
          check_valid_gf(*active_levels, gi, vi, tl,
                         []() { return "SyncGroupsByDirI after syncing"; });
      }
    }
  }

  sync_multipatch_postcheck(cctkGH, groups, "SyncGroupsByDirI");

  assert(sync_active);

  return numgroups;
}

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

  task_manager tasks1;
  task_manager tasks2;
  task_manager tasks3;

  for (const int gi : groups) {
    active_levels->loop_serially([&](auto &restrict leveldata) {
      auto &restrict groupdata = *leveldata.groupdata.at(gi);
      assert(!groupdata.mfab.empty());

      const int ntls = groupdata.mfab.size();
      const int sync_tl = ntls > 1 ? ntls - 1 : ntls;

      if (leveldata.level == 0) {
        for (int tl = 0; tl < sync_tl; ++tl) {
          tasks1.submit_serially([&tasks2, &leveldata, &groupdata, tl]() {
            FillPatch_Sync(tasks2, groupdata, *groupdata.mfab.at(tl),
                           ghext->patchdata.at(leveldata.patch)
                               .amrcore->Geom(leveldata.level));
          });
        }

      } else {

        const int level = leveldata.level;
        const auto &restrict coarseleveldata =
            ghext->patchdata.at(leveldata.patch).leveldata.at(level - 1);
        auto &restrict coarsegroupdata = *coarseleveldata.groupdata.at(gi);
        assert(!coarsegroupdata.mfab.empty());
        assert(coarsegroupdata.numvars == groupdata.numvars);

        const bool integrated_subiter =
            group_is_integrated(gi) && leveldata.iteration > 0;

        if (integrated_subiter) {
          for (int tl = 0; tl < sync_tl; ++tl) {
            tasks1.submit_serially([&tasks2, &leveldata, &groupdata, tl]() {
              FillPatch_Sync(tasks2, groupdata, *groupdata.mfab.at(tl),
                             ghext->patchdata.at(leveldata.patch)
                                 .amrcore->Geom(leveldata.level));
            });
          }
        } else {
          amrex::Interpolater *const interpolator = groupdata.interpolator;

          const bool aligned =
              (leveldata.iteration == coarseleveldata.iteration);
          const bool blend = !aligned && coarsegroupdata.mfab.size() >= 2;
          const rat64 cdt = coarseleveldata.delta_iteration;
          const rat64 t_new = coarseleveldata.iteration;
          const rat64 t_fin = leveldata.iteration;
          const CCTK_REAL w_new = CCTK_REAL((t_fin - t_new + cdt) / cdt);

          const char *const label =
              "SyncGroupsByDirISubcycling on coarse level before prolongation";

          for (int tl = 0; tl < sync_tl; ++tl) {
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

  for (const int gi : groups) {
    const auto &patchdata0 = ghext->patchdata.at(0);
    const auto &leveldata0 = patchdata0.leveldata.at(0);
    const auto &groupdata0 = *leveldata0.groupdata.at(gi);
    assert(!groupdata0.mfab.empty());
    const int ntls0 = groupdata0.mfab.size();
    const int sync_tl0 = ntls0 > 1 ? ntls0 - 1 : ntls0;

    active_levels->loop_serially([&](auto &restrict leveldata) {
      auto &restrict groupdata = *leveldata.groupdata.at(gi);
      assert(!groupdata.mfab.empty());

      const bool integrated_subiter =
          group_is_integrated(gi) && leveldata.iteration > 0;

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
      }
    });

    for (int tl = 0; tl < sync_tl0; ++tl) {
      for (int vi = 0; vi < groupdata0.numvars; ++vi) {
        poison_invalid_gf(*active_levels, gi, vi, tl);
        if (!have_multipatch_boundaries)
          check_valid_gf(*active_levels, gi, vi, tl, []() {
            return "SyncGroupsByDirISubcycling after syncing";
          });
      }
    }
  }

  sync_multipatch_postcheck(cctkGH, groups, "SyncGroupsByDirISubcycling");

  assert(sync_active);

  return numgroups;
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

  task_manager tasks1;
  task_manager tasks2;

  for (const int gi : groups) {
    active_levels->loop_serially([&](auto &restrict leveldata) {
      auto &restrict groupdata = *leveldata.groupdata.at(gi);
      assert(!groupdata.mfab.empty());

      const int ntls = groupdata.mfab.size();
      const int tl_lo = (tl_arg < 0) ? 0 : tl_arg;
      const int tl_hi =
          (tl_arg < 0) ? (ntls > 1 ? ntls - 1 : ntls) : (tl_arg + 1);
      assert(tl_lo >= 0 && tl_hi <= ntls);

      for (int tl = tl_lo; tl < tl_hi; ++tl) {
        tasks1.submit_serially([&tasks2, &leveldata, &groupdata, tl]() {
          FillPatch_Sync(tasks2, groupdata, *groupdata.mfab.at(tl),
                         ghext->patchdata.at(leveldata.patch)
                             .amrcore->Geom(leveldata.level));
        });
      }
    });
  }

  tasks1.run_tasks_serially();
  synchronize();
  tasks2.run_tasks_serially();
  synchronize();

  sync_multipatch_postcheck(cctkGH, groups, "SyncGroupsByDirIGhostOnly");

  assert(sync_active);

  return numgroups;
}

void ProlongateAfterRestriction(const cGH *cctkGH) {
  const std::vector<int> groups = reprolongate_groups();
  SyncGroupsByDirIProlongateOnly_impl(cctkGH, groups.size(), groups.data(),
                                      nullptr, true);
}

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

void EnsureFluxGeometry(const int patch, const int level) {
  const auto &patchdata = ghext->patchdata.at(patch);
  const auto &leveldata = patchdata.leveldata.at(level);

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
  assert(stage >= 1 && stage <= ghext->num_rk_stages);

  static Timer timer("AccumulateFluxes");
  Interval interval(timer);

  const auto &patchdata = ghext->patchdata.at(patch);
  const auto &leveldata = patchdata.leveldata.at(level);
  const bool have_child = level + 1 < int(patchdata.leveldata.size());
  const auto *const childleveldata =
      have_child ? &patchdata.leveldata.at(level + 1) : nullptr;
  const int tl = 0;

  for (int gi = 0; gi < int(leveldata.groupdata.size()); ++gi) {
    if (!leveldata.groupdata.at(gi))
      continue;
    const auto &groupdata = *leveldata.groupdata.at(gi);
    if (groupdata.mfab.empty())
      continue;
    if (groupdata.fluxes[0] < 0)
      continue;

    amrex::FluxRegister *const child_freg =
        childleveldata ? childleveldata->groupdata.at(gi)->freg.get() : nullptr;
    amrex::FluxRegister *const own_freg = groupdata.freg.get();
    if (!child_freg && !own_freg)
      continue;

    EnsureFluxGeometry(patch, level);

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

    if (child_freg) {
      amrex::FluxRegister *const child_scratch =
          childleveldata->groupdata.at(gi)->freg_scratch.get();
      assert(child_scratch);
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
          scratch.copyFrom(flux, 0, 0, 0, numvars);
          for (amrex::FabSetIter fsi(scratch); fsi.isValid(); ++fsi) {
            const amrex::Box &bx = fsi.validbox();
            const auto sfab = scratch.array(fsi);
            amrex::ParallelFor(
                bx, numvars,
                [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) noexcept {
                  sfab(i, j, k, n) = sfab(i, j, k, n) * mult * area_d;
                });
          }
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
  }
}

static void reflux_level(const cGH *cctkGH, const int level) {
  DECLARE_CCTK_PARAMETERS;

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

    std::vector<int> refluxed_groups;

    for (int gi = 0; gi < int(leveldata.groupdata.size()); ++gi) {
      if (!leveldata.groupdata.at(gi))
        continue;
      const auto &groupdata = *leveldata.groupdata.at(gi);
      if (groupdata.mfab.empty())
        continue;
      const auto &finegroupdata = *fineleveldata.groupdata.at(gi);
      assert(!finegroupdata.mfab.empty());

      if (!finegroupdata.freg)
        continue;

      if (verbose)
        CCTK_VINFO("Reflux: applying the flux register of levels (%d, %d) to "
                   "%s at iteration %d",
                   level, level + 1, groupdata.groupname.c_str(),
                   cctkGH->cctk_iteration);

      for (int vi = 0; vi < groupdata.numvars; ++vi)
        error_if_invalid(groupdata, vi, tl, make_valid_int(), []() {
          return "Reflux before refluxing: Coarse level data";
        });

      EnsureFluxGeometry(patch, level);

      amrex::MultiFab &state = *groupdata.mfab.at(tl);
      const amrex::MultiFab &volume = *leveldata.cell_volume;
      assert(volume.ixType() == state.ixType());
      assert(volume.DistributionMap() == state.DistributionMap());
      finegroupdata.freg->Reflux(state, volume, 1.0, 0, 0, groupdata.numvars,
                                 geom);
      refluxed_groups.push_back(gi);
    }

    const active_levels_t coarse_level(level, level + 1, patch, patch + 1);
    for (const int gi : refluxed_groups) {
      const auto &groupdata = *leveldata.groupdata.at(gi);
      for (int vi = 0; vi < groupdata.numvars; ++vi)
        check_valid_gf(coarse_level, gi, vi, tl, []() {
          return "Reflux after refluxing: Coarse level data";
        });
    }
  }
}

void Reflux(const cGH *cctkGH, const int min_level, const int max_level) {
  if (!ghext->do_reflux)
    return;
  assert(min_level >= 0 && max_level <= ghext->num_levels());
  for (int level = max_level - 2; level >= min_level; --level)
    reflux_level(cctkGH, level);
}

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

        if (gi == gi_regrid_error)
          continue;

        int ntls = groupdata.mfab.size();
        int restrict_tl = ntls > 1 ? ntls - 1 : ntls;
        for (int tl = 0; tl < restrict_tl; ++tl) {

          for (int vi = 0; vi < groupdata.numvars; ++vi) {

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

          {
            static Timer timer("Restrict::average_down");
            Interval interval(timer);
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

          if (do_validity_tracking) {
            for (int vi = 0; vi < groupdata.numvars; ++vi) {
              groupdata.valid.at(tl).at(vi).set_invalid(
                  make_valid_outer() | make_valid_ghosts(),
                  []() { return "Restrict"; });
              poison_invalid_gf(active_levels, gi, vi, tl);
              check_valid_gf(active_levels, gi, vi, tl, []() {
                return "Restrict on coarse level after restricting";
              });
            }
          }

        }
      }
    }
  }
}

void Restrict(const cGH *cctkGH, const int min_level, const int max_level,
              const std::vector<int> &groups) {
  assert(min_level >= 0 && max_level <= ghext->num_levels());
  if (groups.empty())
    return;
  for (int level = max_level - 2; level >= min_level; --level)
    restrict_level(cctkGH, level, groups, true);
}

void RestrictNoPoison(const cGH *cctkGH, const int min_level,
                      const int max_level, const std::vector<int> &groups) {
  assert(min_level >= 0 && max_level <= ghext->num_levels());
  if (groups.empty())
    return;
  for (int level = max_level - 2; level >= min_level; --level)
    restrict_level(cctkGH, level, groups, false);
}

} // namespace CarpetX
