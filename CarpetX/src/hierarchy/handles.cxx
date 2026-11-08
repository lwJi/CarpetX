#include <carpetx.hxx>

#include "hierarchy/driver.hxx"
#include "hierarchy/loops.hxx"
#include "traversal/traversal.hxx"
#include "util/refuse.hxx"

#include <cctk.h>

#include <functional>
#include <optional>
#include <source_location>

namespace CarpetX {

namespace {

void require_window_bounds(const window_t::bounds_t levels,
                           const window_t::bounds_t patches,
                           const std::source_location site) {
  const GHExt &ext = driver();
  if (levels.min < 0 || levels.max > ext.num_levels() ||
      levels.min > levels.max)
    refuse(site, "CarpetX window levels [%d, %d) must lie within [0, %d)",
           levels.min, levels.max, ext.num_levels());
  if (patches.min < 0 || patches.max > ext.num_patches() ||
      patches.min > patches.max)
    refuse(site, "CarpetX window patches [%d, %d) must lie within [0, %d)",
           patches.min, patches.max, ext.num_patches());
}

void require_window_level(const window_t::bounds_t levels,
                          const window_t::bounds_t patches,
                          const std::source_location site) {
  if (levels.min >= levels.max || patches.min >= patches.max)
    refuse(site,
           "CarpetX window of patches [%d, %d), refinement levels [%d, %d) "
           "holds no level",
           patches.min, patches.max, levels.min, levels.max);
}

} // namespace

level_t level_t::parent(const std::source_location site) const {
  if (level_ == 0)
    refuse(site, "CarpetX level 0 of patch %d has no parent level",
           patch_.index());
  return level_t{patch_, level_ - 1};
}

std::optional<level_t> level_t::child() const {
  if (level_ + 1 < driver().num_levels(patch_))
    return level_t{patch_, level_ + 1};
  return std::nullopt;
}

variable_t variable(const var_t var) {
  const gi_t gi{CCTK_GroupIndexFromVarI(var.index())};
  if (gi.index() < 0)
    CCTK_VERROR("CarpetX::variable: variable index %d does not exist: there "
                "are %d variables",
                var.index(), CCTK_NumVars());
  return variable_t{gi, vi_t{var.index() - CCTK_FirstVarIndexI(gi.index())}};
}

window_t window_t::one_level(const level_t level,
                             const std::source_location site) {
  const patch_t patch = level.patch();
  require_window_bounds({level.index(), level.index() + 1},
                        {patch.index(), patch.index() + 1}, site);
  return window_t{level.index(), level.index() + 1, patch.index(),
                  patch.index() + 1};
}

window_t window_t::levels(const int min_level, const int max_level,
                          const std::source_location site) {
  const int num_patches = driver().num_patches();
  require_window_bounds({min_level, max_level}, {0, num_patches}, site);
  return window_t{min_level, max_level, 0, num_patches};
}

window_t window_t::narrow(const int min_level, const int max_level,
                          const std::source_location site) const {
  require_window_bounds({min_level, max_level}, patch_bounds(), site);
  return window_t{min_level, max_level, min_patch_, max_patch_};
}

level_t window_t::coarsest(const std::source_location site) const {
  require_window_level(level_bounds(), patch_bounds(), site);
  return level_t{patch_t{min_patch_}, min_level_};
}

level_t window_t::finest(const std::source_location site) const {
  require_window_level(level_bounds(), patch_bounds(), site);
  return level_t{patch_t{min_patch_}, max_level_ - 1};
}

void window_t::loop_serially(
    const std::function<void(level_t level)> &kernel) const {
  loop_coarse_to_fine(kernel);
}

void window_t::loop_coarse_to_fine(
    const std::function<void(level_t level)> &kernel) const {
  CarpetX::loop_coarse_to_fine(
      *this, [&](const GHExt::PatchData::LevelData &leveldata) {
        kernel(leveldata.level);
      });
}

void window_t::loop_parallel(
    const std::function<void(level_t level, tile_t tile, const cGH *cctkGH)>
        &kernel) const {
  CarpetX::loop_parallel(
      *this, [&](const GHExt::PatchData::LevelData &leveldata,
                 const tile_t tile, const cGH *const cctkGH) {
        kernel(leveldata.level, tile, cctkGH);
      });
}

window_t active_window() {
  return scheduled_window("CarpetX::active_window()");
}

window_t all_levels() { return window_t::levels(0, driver().num_levels()); }

} // namespace CarpetX
