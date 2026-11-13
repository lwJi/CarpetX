#include "hierarchy/rk_levels.hxx"

#include "hierarchy/driver.hxx"
#include "hierarchy/integrator.hxx"
#include "hierarchy/loops.hxx"

#include <vector>

namespace CarpetX {

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
