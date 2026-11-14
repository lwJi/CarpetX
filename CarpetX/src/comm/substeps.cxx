#include "comm/substeps.hxx"

#include "hierarchy/model.hxx"
#include <carpetx.hxx>

#include <cassert>

namespace CarpetX {

substep_t substep_of_parent_step(const level_t child) {
  const GHExt &ext = driver();
  const GHExt::PatchData::LevelData &parentdata = ext.level(child.parent());
  const GHExt::PatchData::LevelData &childdata = ext.level(child);
  const rat64 parent_step_begin =
      parentdata.iteration - parentdata.delta_iteration;
  const rat64 end =
      (childdata.iteration - parent_step_begin) / parentdata.delta_iteration;
  const rat64 begin =
      end - childdata.delta_iteration / parentdata.delta_iteration;
  assert(begin >= 0 && end <= 1 &&
         "the child has taken a substep of its parent's current step");
  return {begin, end};
}

bool window_is_restricted(const window_t window) {
  const auto [min_level, max_level] = window.level_bounds();
  assert(min_level < max_level);
  return max_level == driver().num_levels() && max_level - min_level >= 2;
}

} // namespace CarpetX
