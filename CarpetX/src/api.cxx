#include <carpetx.hxx>

#include "api.hxx"
#include "driver.hxx"
#include "schedule.hxx"

#include <cctk.h>

#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace CarpetX {

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

namespace {

void require_grid_function(const char *const what, const int gi) {
  if (gi < 0 || gi >= CCTK_NumGroups())
    CCTK_VERROR("%s: group index %d does not exist: there are %d groups", what,
                gi, CCTK_NumGroups());
  if (CCTK_GroupTypeI(gi) != CCTK_GF)
    CCTK_VERROR("%s: group %s is not a grid function group", what,
                CCTK_FullGroupName(gi));
}

GHExt::PatchData::LevelData::GroupData &
resolve(const char *const what, GHExt::PatchData::LevelData &leveldata,
        const int gi) {
  require_grid_function(what, gi);
  GHExt::PatchData::LevelData::GroupData *const groupdata =
      leveldata.groupdata.at(gi).get();
  if (!groupdata || groupdata->mfab.empty())
    CCTK_VERROR("%s: group %s has no storage on patch %d, refinement level %d",
                what, CCTK_FullGroupName(gi), leveldata.patch, leveldata.level);
  return *groupdata;
}

void require_timelevel(const char *const what,
                       const GHExt::PatchData::LevelData::GroupData &groupdata,
                       const int tl) {
  const int ntls = groupdata.mfab.size();
  if (tl < 0 || tl >= ntls)
    CCTK_VERROR("%s: time level %d of group %s does not exist on patch %d, "
                "refinement level %d: it has %d time levels",
                what, tl, groupdata.groupname.c_str(), groupdata.patch,
                groupdata.level, ntls);
}

std::function<std::string()> reason(const char *const why) {
  const std::string what = why ? why : "";
  const int iteration = driver().get_global_cctkGH()->cctk_iteration;
  return [what, iteration]() {
    return what + " (iteration " + std::to_string(iteration) + ")";
  };
}

} // namespace

GHExt::PatchData::LevelData::GroupData &resolve(const char *const what,
                                                const group_t group) {
  return resolve(what, resolve(group.level), group.gi);
}

void require_no_open_write(
    const char *const what,
    const GHExt::PatchData::LevelData::GroupData &groupdata) {
  if (groupdata.open_writes > 0)
    CCTK_VERROR("%s: group %s has %d open write scope%s (CarpetX::write) on "
                "patch %d, refinement level %d; destroy the write_t first",
                what, groupdata.groupname.c_str(), groupdata.open_writes,
                groupdata.open_writes == 1 ? "" : "s", groupdata.patch,
                groupdata.level);
}

active_levels_t require_operable(const char *const what, const window_t window,
                                 const std::vector<int> &groups) {
  for (const int gi : groups)
    require_grid_function(what, gi);
  const active_levels_t active_levels(window);
  active_levels.loop_serially([&](GHExt::PatchData::LevelData &leveldata) {
    for (const int gi : groups)
      if (const auto *const groupdata = leveldata.groupdata.at(gi).get())
        require_no_open_write(what, *groupdata);
  });
  return active_levels;
}

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

int numvars(const group_t group) {
  return resolve("CarpetX::numvars", group).numvars;
}

std::array<int, Loop::dim> indextype(const group_t group) {
  return resolve("CarpetX::indextype", group).indextype;
}

std::array<int, Loop::dim> nghostzones(const group_t group) {
  return resolve("CarpetX::nghostzones", group).nghostzones;
}

void window_t::loop_serially(
    const std::function<void(level_t level)> &kernel) const {
  loop_coarse_to_fine(kernel);
}

void window_t::loop_coarse_to_fine(
    const std::function<void(level_t level)> &kernel) const {
  active_levels_t(*this).loop_coarse_to_fine(
      [&](const GHExt::PatchData::LevelData &leveldata) {
        kernel(level_t{leveldata.patch, leveldata.level});
      });
}

void window_t::loop_parallel(
    const std::function<void(level_t level, int index, int component,
                             const cGH *cctkGH)> &kernel) const {
  active_levels_t(*this).loop_parallel([&](const int patch, const int level,
                                           const int index, const int component,
                                           const cGH *const cctkGH) {
    kernel(level_t{patch, level}, index, component, cctkGH);
  });
}

window_t active_window() {
  if (!active_levels)
    CCTK_ERROR("CarpetX::active_window() called outside a traversal: no "
               "levels are active");
  return active_levels->window();
}

window_t all_levels() {
  GHExt &ext = driver();
  return window_t{0, ext.num_levels(), 0, ext.num_patches()};
}

const amrex::MultiFab &data(const group_t group, const int tl) {
  const auto &groupdata = resolve("CarpetX::data", group);
  require_timelevel("CarpetX::data", groupdata, tl);
  return *groupdata.mfab.at(tl);
}

struct write_t::impl {
  group_t group;
  int tl;
  valid_t leaves_valid;
  std::function<std::string()> why;
};

write_t::write_t(std::unique_ptr<impl> p) noexcept : p(std::move(p)) {}

write_t::write_t(write_t &&) noexcept = default;

amrex::MultiFab &write_t::mfab() {
  if (!p)
    CCTK_ERROR("CarpetX::write_t::mfab() called on a moved-from write scope");
  return *resolve("CarpetX::write_t::mfab", p->group).mfab.at(p->tl);
}

write_t::~write_t() {
  if (!p)
    return;
  auto &groupdata = resolve("CarpetX::write_t::~write_t", p->group);
  --groupdata.open_writes;
  for (auto &why_valid : groupdata.valid.at(p->tl))
    why_valid.set_all(p->leaves_valid, p->why);
}

write_t write(const group_t group, const int tl, const valid_t leaves_valid,
              const char *const what) {
  auto &groupdata = resolve("CarpetX::write", group);
  require_timelevel("CarpetX::write", groupdata, tl);
  ++groupdata.open_writes;
  return write_t(std::unique_ptr<write_t::impl>(
      new write_t::impl{group, tl, leaves_valid, reason(what)}));
}

struct scratch_t::impl {
  group_t group;
  std::size_t slot;
};

namespace {

GHExt::PatchData::LevelData::GroupData::scratch_slot_t &
resolve_scratch(const char *const what, const group_t group,
                const std::size_t slot) {
  auto &pool = resolve(what, group).scratch_pool;
  if (slot >= pool.size() || !pool.at(slot).in_use)
    CCTK_VERROR("%s: scratch buffer %d of group %s on patch %d, refinement "
                "level %d no longer exists: the level was remade while the "
                "scratch_t was held",
                what, int(slot), CCTK_FullGroupName(group.gi),
                group.level.patch, group.level.level);
  return pool.at(slot);
}

} // namespace

scratch_t::scratch_t(std::unique_ptr<impl> p) noexcept : p(std::move(p)) {}

scratch_t::scratch_t(scratch_t &&) noexcept = default;

amrex::MultiFab &scratch_t::mfab() {
  if (!p)
    CCTK_ERROR("CarpetX::scratch_t::mfab() called on a moved-from scratch "
               "buffer");
  return *resolve_scratch("CarpetX::scratch_t::mfab", p->group, p->slot).mfab;
}

const amrex::MultiFab &scratch_t::mfab() const {
  if (!p)
    CCTK_ERROR("CarpetX::scratch_t::mfab() called on a moved-from scratch "
               "buffer");
  return *resolve_scratch("CarpetX::scratch_t::mfab", p->group, p->slot).mfab;
}

scratch_t::~scratch_t() {
  if (!p)
    return;
  resolve_scratch("CarpetX::scratch_t::~scratch_t", p->group, p->slot).in_use =
      false;
}

scratch_t scratch(const group_t group) {
  auto &groupdata = resolve("CarpetX::scratch", group);
  auto &pool = groupdata.scratch_pool;
  std::size_t slot = 0;
  while (slot < pool.size() && pool.at(slot).in_use)
    ++slot;
  if (slot == pool.size()) {
    const amrex::MultiFab &mfab0 = *groupdata.mfab.at(0);
    pool.push_back({std::make_unique<amrex::MultiFab>(
                        mfab0.boxArray(), mfab0.DistributionMap(),
                        mfab0.nComp(), mfab0.nGrowVect()),
                    false});
  }
  pool.at(slot).in_use = true;
  return scratch_t(std::unique_ptr<scratch_t::impl>(
      new scratch_t::impl{group, slot}));
}

valid_t valid(const group_t group, const int tl, const int vi) {
  const auto &groupdata = resolve("CarpetX::valid", group);
  require_timelevel("CarpetX::valid", groupdata, tl);
  if (vi < 0 || vi >= groupdata.numvars)
    CCTK_VERROR("CarpetX::valid: variable index %d of group %s does not "
                "exist: it has %d variables",
                vi, groupdata.groupname.c_str(), groupdata.numvars);
  return groupdata.valid.at(tl).at(vi).get();
}

void require_valid(const group_t group, const int tl, const valid_t where,
                   const char *const why) {
  const auto &groupdata = resolve("CarpetX::require_valid", group);
  require_timelevel("CarpetX::require_valid", groupdata, tl);
  const std::string what = why ? why : "";
  const auto msg = [what]() { return what; };
  for (int vi = 0; vi < groupdata.numvars; ++vi)
    error_if_invalid(groupdata, vi, tl, where, msg);
  const active_levels_t one_level(group.level.level, group.level.level + 1,
                                  group.level.patch, group.level.patch + 1);
  for (int vi = 0; vi < groupdata.numvars; ++vi)
    check_valid_gf(one_level, group.gi, vi, tl, msg);
}

void invalidate(const window_t window, const std::vector<int> &groups,
                const int tl, const char *const why) {
  const active_levels_t active_levels =
      require_operable("CarpetX::invalidate", window, groups);
  const auto what = reason(why);
  active_levels.loop_serially([&](GHExt::PatchData::LevelData &leveldata) {
    for (const int gi : groups) {
      auto &groupdata = resolve("CarpetX::invalidate", leveldata, gi);
      require_timelevel("CarpetX::invalidate", groupdata, tl);
      for (auto &why_valid : groupdata.valid.at(tl))
        why_valid.set_all(valid_t(), what);
    }
  });
}

void sync(const window_t window, const std::vector<int> &groups) {
  const active_levels_t active_levels =
      require_operable("CarpetX::sync", window, groups);
  SyncGroups(active_levels, driver().get_global_cctkGH(), groups.size(),
             groups.data(), nullptr);
}

void sync_ghosts_only(const window_t window, const std::vector<int> &groups,
                      const int tl) {
  const active_levels_t active_levels =
      require_operable("CarpetX::sync_ghosts_only", window, groups);
  SyncGroupsByDirIGhostOnly(active_levels, driver().get_global_cctkGH(),
                            groups.size(), groups.data(), nullptr, tl);
}

void prolongate_ghosts(const window_t window, const std::vector<int> &groups,
                       const int tl) {
  const active_levels_t active_levels =
      require_operable("CarpetX::prolongate_ghosts", window, groups);
  SyncGroupsByDirIProlongateOnly(active_levels, driver().get_global_cctkGH(),
                                 groups.size(), groups.data(), nullptr, tl);
}

void prolongate_aligned(const window_t window, const std::vector<int> &groups,
                        const int tl) {
  const active_levels_t active_levels =
      require_operable("CarpetX::prolongate_aligned", window, groups);
  SyncGroupsByDirIProlongateOnlyAligned(
      active_levels, driver().get_global_cctkGH(), groups.size(), groups.data(),
      nullptr, tl);
}

void restrict_no_poison(const window_t window, const std::vector<int> &groups) {
  require_operable("CarpetX::restrict_no_poison", window, groups);
  RestrictNoPoison(driver().get_global_cctkGH(), window.min_level,
                   window.max_level, groups);
}

} // namespace CarpetX
