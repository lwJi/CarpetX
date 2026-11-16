#include "comm/ghost_prolongation.hxx"

#include "hierarchy/loops.hxx"
#include "hierarchy/model.hxx"
#include "comm/fillpatch.hxx"
#include "comm/integrator.hxx"
#include "comm/outer_boundary.hxx"
#include "comm/substeps.hxx"
#include "comm/group_selection.hxx"
#include "comm/level_runs.hxx"
#include "comm/timelevels.hxx"
#include "traversal/traversal.hxx"
#include "util/task_manager.hxx"
#include <carpetx.hxx>
#include "hierarchy/scopes.hxx"

#include <cctk.h>
#include <cctk_Parameters.h>

#include <cassert>
#include <functional>
#include <span>
#include <string>
#include <vector>

namespace CarpetX {

void error_if_parent_invalid(const GHExt::PatchData::LevelData &leveldata,
                             const gi_t gi, const tl_t tl,
                             const char *const label) {
  const auto &restrict coarseleveldata =
      driver().level(leveldata.level.parent());
  const auto &restrict coarsegroupdata = coarseleveldata.group(gi);
  const bool recovering =
      recovered_iteration() >= 0 &&
      coarseleveldata.last_finalized_iteration() <= recovered_iteration();

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

static std::vector<level_run_t> ProlongateGhosts_impl(
    const window_t window, const std::span<const gi_t> groups0,
    const bool prolongate_on_same_iteration, const timelevels_t tls) {
  DECLARE_CCTK_PARAMETERS;

  const GHExt &ext = driver();

  error_if_sync_nested();

  static Timer timer("Sync");
  Interval interval(timer);

  const char *const label = "ProlongateGhosts";
  const std::string prolongation_label =
      std::string(label) + " on coarse level before prolongation";

  sync_log_groups(label, groups0);

  std::vector<gi_t> groups = sync_filter_groups(groups0);

  const auto prolongates_into =
      [&ext, prolongate_on_same_iteration](
          const GHExt::PatchData::LevelData &leveldata) {
        if (leveldata.level.index() == 0)
          return false;
        if (ext.use_subcycling && prolongate_on_same_iteration)
          return aligned_with_parent(leveldata.level);
        return true;
      };
  const std::vector<level_run_t> runs = split_levels(window, prolongates_into);
  write_scopes_t scopes;
  for (const gi_t gi : groups) {
    const tl_range fill_tls = tls.range(ext.record(gi));
    for (const level_run_t &run : runs) {
      if (!run.value)
        continue;
      for (const tl_t tl : fill_tls) {
        scopes.open(run.levels, {.gi = gi,
                                 .tl = tl,
                                 .fills = make_valid_coarse_fine(),
                                 .keeps = make_valid_all(),
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

      const tl_range fill_tls = tls.range(groupdata.record);

      if (!prolongates_into(leveldata))
        return;

      const auto &restrict coarseleveldata =
          ext.level(leveldata.level.parent());

      auto &restrict coarsegroupdata = coarseleveldata.group(gi);
      assert(coarsegroupdata.record.numvars == groupdata.record.numvars);

      for (const tl_t tl : fill_tls) {
        error_if_parent_invalid(leveldata, gi, tl, prolongation_label.c_str());
        tasks1.submit_serially([&tasks2, &tasks3, &leveldata, &groupdata,
                                &coarsegroupdata, &scopes, gi, tl, &ext]() {
          FillPatch_ProlongateOnly(tasks2, tasks3, groupdata, coarsegroupdata,
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
    });
  }

  tasks1.run_tasks_serially();
  synchronize();
  tasks2.run_tasks_serially();
  synchronize();
  tasks3.run_tasks_serially();
  synchronize();

  scopes.close();
  return runs;
}

void ProlongateGhosts(const window_t window,
                      const std::span<const gi_t> groups0,
                      const timelevels_t tls) {
  ProlongateGhosts_impl(window, groups0, false, tls);
}

void ProlongateAligned(const window_t window,
                       const std::span<const gi_t> groups0,
                       const timelevels_t tls) {
  ProlongateGhosts_impl(window, groups0, true, tls);
}

void ProlongateAfterRestriction(const window_t window) {
  const std::vector<gi_t> groups = sync_filter_groups(reprolongate_groups());
  const timelevels_t tls = timelevels_t::all_but_oldest();
  const std::vector<level_run_t> runs =
      ProlongateGhosts_impl(window, groups, true, tls);
  for (const level_run_t &run : runs)
    if (run.value)
      fill_outer_boundary(run.levels, groups, tls,
                          "ProlongateAfterRestriction applies the boundary "
                          "condition after the prolongation and reads the "
                          "interior and both ghost regions");
}

} // namespace CarpetX
