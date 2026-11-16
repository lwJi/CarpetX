#include "comm/restrict.hxx"

#include "hierarchy/model.hxx"
#include "comm/level_pairs.hxx"
#include "comm/timelevels.hxx"
#include <carpetx.hxx>
#include "hierarchy/scopes.hxx"

#include <cctk.h>

#include <AMReX_MultiFabUtil.H>

#include <cassert>
#include <optional>
#include <span>
#include <vector>

namespace CarpetX {

std::vector<gi_t> collect_restrictable_groups() {
  std::vector<gi_t> groups;
  groups.reserve(CCTK_NumGroups());
  for (const auto &group : driver().records(group_kind::grid_function))
    if (group.do_restrict)
      groups.push_back(group.groupindex);
  return groups;
}

void restrict_level(const level_t level, const std::span<const gi_t> groups,
                    const valid_t coarse_keeps) {
  static Timer timer("Restrict");
  Interval interval(timer);

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
      write_scope_t scope = open_write(coarse_level, {.gi = gi,
                                                      .tl = tl,
                                                      .fills = make_valid_int(),
                                                      .keeps = coarse_keeps,
                                                      .event = "Restrict"});

      {
        static Timer timer("Restrict::average_down");
        Interval interval(timer);
        int rank = 0;
        for (int d = 0; d < dim; ++d)
          rank += groupdata.grid_function.indextype.at(d);
        switch (rank) {
        case 0:
          average_down_nodal(finegroupdata.timelevel(tl).mfab(),
                             scope.mfab(level), reffact);
          break;
        case 1:
          average_down_edges(finegroupdata.timelevel(tl).mfab(),
                             scope.mfab(level), reffact);
          break;
        case 2:
          average_down_faces(finegroupdata.timelevel(tl).mfab(),
                             scope.mfab(level), reffact);
          break;
        case 3:
          average_down(finegroupdata.timelevel(tl).mfab(), scope.mfab(level), 0,
                       groupdata.record.numvars, reffact);
          break;
        default:
          assert(0);
        }
      }
    }
  }
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
