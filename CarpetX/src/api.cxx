#include <carpetx.hxx>

#include "driver.hxx"
#include "schedule.hxx"

#include <cctk.h>

#include <optional>
#include <vector>

namespace CarpetX {

namespace {

GHExt &driver() {
  if (!ghext)
    CCTK_ERROR("The CarpetX grid hierarchy does not exist: the public API is "
               "available between the driver's SetupGH and its Shutdown");
  return *ghext;
}

GHExt::PatchData &resolve_patch(const int patch) {
  GHExt &ext = driver();
  if (patch < 0 || patch >= ext.num_patches())
    CCTK_VERROR("CarpetX patch %d does not exist: there are %d patches", patch,
                ext.num_patches());
  return ext.patchdata.at(patch);
}

GHExt::PatchData::LevelData &resolve(const level_t level) {
  GHExt::PatchData &patchdata = resolve_patch(level.patch);
  const int nlevels = patchdata.leveldata.size();
  if (level.level < 0 || level.level >= nlevels)
    CCTK_VERROR("CarpetX level %d does not exist on patch %d: it has %d levels",
                level.level, level.patch, nlevels);
  return patchdata.leveldata.at(level.level);
}

active_levels_t to_active_levels(const window_t window) {
  GHExt &ext = driver();
  if (window.min_level < 0 || window.max_level > ext.num_levels() ||
      window.min_level > window.max_level)
    CCTK_VERROR("CarpetX window levels [%d, %d) must lie within [0, %d)",
                window.min_level, window.max_level, ext.num_levels());
  if (window.min_patch < 0 || window.max_patch > ext.num_patches() ||
      window.min_patch > window.max_patch)
    CCTK_VERROR("CarpetX window patches [%d, %d) must lie within [0, %d)",
                window.min_patch, window.max_patch, ext.num_patches());
  return active_levels_t(window.min_level, window.max_level, window.min_patch,
                         window.max_patch);
}

class window_scope {
  std::optional<active_levels_t> saved;

public:
  explicit window_scope(const window_t window) : saved(active_levels) {
    active_levels = to_active_levels(window);
  }
  window_scope(const window_scope &) = delete;
  window_scope &operator=(const window_scope &) = delete;
  ~window_scope() { active_levels = saved; }
};

} // namespace

int num_patches() { return driver().num_patches(); }

int num_levels(const int patch) {
  return resolve_patch(patch).leveldata.size();
}

level_t parent(const level_t level) {
  resolve(level);
  if (level.level == 0)
    CCTK_VERROR("CarpetX level 0 of patch %d has no parent level", level.patch);
  return level_t{level.patch, level.level - 1};
}

rat64 iteration(const level_t level) { return resolve(level).iteration; }

rat64 delta_iteration(const level_t level) {
  return resolve(level).delta_iteration;
}

const amrex::Geometry &geometry(const level_t level) {
  resolve(level);
  return resolve_patch(level.patch).amrcore->Geom(level.level);
}

std::array<std::array<symmetry_t, Loop::dim>, 2>
symmetries(const level_t level) {
  resolve(level);
  return resolve_patch(level.patch).symmetries;
}

void window_t::loop_serially(
    const std::function<void(level_t level)> &kernel) const {
  loop_coarse_to_fine(kernel);
}

void window_t::loop_coarse_to_fine(
    const std::function<void(level_t level)> &kernel) const {
  const window_scope scope(*this);
  to_active_levels(*this).loop_coarse_to_fine(
      [&](const GHExt::PatchData::LevelData &leveldata) {
        kernel(level_t{leveldata.patch, leveldata.level});
      });
}

void window_t::loop_parallel(
    const std::function<void(level_t level, int index, int component,
                             const cGH *cctkGH)> &kernel) const {
  const window_scope scope(*this);
  to_active_levels(*this).loop_parallel(
      [&](const int patch, const int level, const int index,
          const int component, const cGH *const cctkGH) {
        kernel(level_t{patch, level}, index, component, cctkGH);
      });
}

window_t active_window() {
  if (!active_levels)
    CCTK_ERROR("CarpetX::active_window() called outside a traversal: no "
               "levels are active");
  return window_t{active_levels->min_level, active_levels->max_level,
                  active_levels->min_patch, active_levels->max_patch};
}

window_t all_levels() {
  GHExt &ext = driver();
  return window_t{0, ext.num_levels(), 0, ext.num_patches()};
}

void restrict_no_poison(const window_t window, const std::vector<int> &groups) {
  for (const int gi : groups)
    if (CCTK_GroupTypeI(gi) != CCTK_GF)
      CCTK_VERROR("CarpetX::restrict_no_poison: group %d (%s) is not a grid "
                  "function group",
                  gi, CCTK_FullGroupName(gi));
  const window_scope scope(window);
  RestrictNoPoison(driver().get_global_cctkGH(), window.min_level,
                   window.max_level, groups);
}

} // namespace CarpetX
