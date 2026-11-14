#include "comm/integrator.hxx"

#include "hierarchy/model.hxx"
#include "hierarchy/loops.hxx"
#include "util/refuse.hxx"
#include <carpetx.hxx>

#include <cctk.h>

#include <cassert>
#include <optional>
#include <source_location>
#include <vector>

namespace CarpetX {

bool group_is_integrated(const gi_t gi) {
  assert(gi.index() >= 0 && gi.index() < CCTK_NumGroups());
  const std::optional<GHExt::integrator_t> &integrator = driver().integrator();
  return integrator && integrator->rhs.contains(gi);
}

stage_range registered_stages() {
  const std::optional<GHExt::integrator_t> &integrator = driver().integrator();
  return integrator ? integrator->stages() : all_rk_stages;
}

rk_dense_output_t registered_dense_output() {
  const std::optional<GHExt::integrator_t> &integrator = driver().integrator();
  return integrator ? integrator->dense_output : rk_dense_output_t::none;
}

std::vector<gi_t> integrated_states(const GHExt::integrator_t &integrator) {
  std::vector<gi_t> states;
  for (const auto &[state, rhs] : integrator.rhs)
    states.push_back(state);
  return states;
}

integrated_groups_t
integrated_groups_with_storage(const level_t level,
                               const GHExt::integrator_t &integrator) {
  const GHExt &ext = driver();
  ext.level(level);
  integrated_groups_t groups;
  for (const auto &[state, rhs] : integrator.rhs) {
    if (!ext.record(state).has_storage())
      continue;
    if (!ext.record(rhs).has_storage())
      refuse(std::source_location::current(),
             "integrated group %s has storage on patch %d, refinement level "
             "%d, but its right-hand side %s has none",
             CCTK_FullGroupName(state.index()), level.patch().index(),
             level.index(), CCTK_FullGroupName(rhs.index()));
    groups.states.push_back(state);
    groups.rhss.push_back(rhs);
  }
  return groups;
}

std::vector<rk_level_t>
integrated_levels(const window_t window,
                  const GHExt::integrator_t &integrator) {
  std::vector<rk_level_t> levels;
  loop_coarse_to_fine(
      window, [&](const GHExt::PatchData::LevelData &leveldata) {
        levels.push_back({leveldata.level, integrated_groups_with_storage(
                                               leveldata.level, integrator)});
      });
  return levels;
}

} // namespace CarpetX
