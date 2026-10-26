#include <carpetx.hxx>

#include "api.hxx"
#include "driver.hxx"
#include "schedule.hxx"

#include <cctk.h>

#include <cstddef>
#include <memory>
#include <source_location>
#include <span>
#include <string>
#include <utility>

namespace CarpetX {

namespace {

std::string event_name(const char *const what) { return what ? what : ""; }

} // namespace

void require_no_open_write(
    const GHExt::PatchData::LevelData::GroupData &groupdata) {
  const int open_writes = groupdata.open_write_count();
  if (open_writes > 0)
    refuse(std::source_location::current(),
           "group %s has %d open write scope%s (CarpetX::write) on patch %d, "
           "refinement level %d; destroy the write_t first",
           groupdata.record.groupname.c_str(), open_writes,
           open_writes == 1 ? "" : "s", groupdata.patch, groupdata.level);
}

active_levels_t require_operable(const window_t window,
                                 const std::span<const gi_t> groups) {
  for (const gi_t gi : groups)
    driver().record(gi);
  const active_levels_t active_levels(window);
  active_levels.loop_serially([&](GHExt::PatchData::LevelData &leveldata) {
    for (const gi_t gi : groups)
      if (driver().record(gi).has_storage())
        require_no_open_write(leveldata.group(gi));
  });
  return active_levels;
}

int num_patches() { return driver().num_patches(); }

int num_levels(const int patch) {
  const operation_scope_t op("CarpetX::num_levels");
  return driver().patch(patch).num_levels();
}

level_t parent(const level_t level) {
  const operation_scope_t op("CarpetX::parent");
  driver().level(level);
  if (level.level == 0)
    CCTK_VERROR("CarpetX level 0 of patch %d has no parent level", level.patch);
  return level_t{level.patch, level.level - 1};
}

rat64 iteration(const level_t level) {
  const operation_scope_t op("CarpetX::iteration");
  return driver().level(level).iteration;
}

rat64 delta_iteration(const level_t level) {
  const operation_scope_t op("CarpetX::delta_iteration");
  return driver().level(level).delta_iteration;
}

const amrex::Geometry &geometry(const level_t level) {
  const operation_scope_t op("CarpetX::geometry");
  driver().level(level);
  return driver().patch(level.patch).amrcore->Geom(level.level);
}

std::array<std::array<symmetry_t, Loop::dim>, 2>
symmetries(const level_t level) {
  const operation_scope_t op("CarpetX::symmetries");
  driver().level(level);
  return driver().patch(level.patch).symmetries;
}

int numvars(const group_t group) {
  const operation_scope_t op("CarpetX::numvars");
  return driver().group(group).record.numvars;
}

std::array<int, Loop::dim> indextype(const group_t group) {
  const operation_scope_t op("CarpetX::indextype");
  return driver().group(group).indextype;
}

std::array<int, Loop::dim> nghostzones(const group_t group) {
  const operation_scope_t op("CarpetX::nghostzones");
  return driver().group(group).nghostzones;
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
  const std::optional<active_levels_t> &levels = scheduled_levels();
  if (!levels)
    CCTK_ERROR("CarpetX::active_window() called outside a traversal: no "
               "levels are active");
  return levels->window();
}

window_t all_levels() {
  GHExt &ext = driver();
  return window_t{0, ext.num_levels(), 0, ext.num_patches()};
}

const amrex::MultiFab &data(const group_t group, const int tl) {
  const operation_scope_t op("CarpetX::data");
  return *driver().group(group).timelevel(tl).mfab;
}

struct write_t::impl {
  group_t group;
  int tl;
  write_scope_t scope;
};

active_levels_t one_level(const level_t level) {
  return active_levels_t(level.level, level.level + 1, level.patch,
                         level.patch + 1);
}

write_t::write_t(std::unique_ptr<impl> p) noexcept : p(std::move(p)) {}

write_t::write_t(write_t &&) noexcept = default;

amrex::MultiFab &write_t::mfab() {
  const operation_scope_t op("CarpetX::write_t::mfab");
  if (!p)
    CCTK_ERROR("CarpetX::write_t::mfab() called on a moved-from write scope");
  return *driver().group(p->group).timelevel(p->tl).mfab;
}

write_t::~write_t() = default;

write_t write(const group_t group, const int tl, const valid_t leaves_valid,
              const char *const what) {
  const operation_scope_t op("CarpetX::write");
  auto &groupdata = driver().group(group);
  const int open_writes = groupdata.timelevel(tl).open_writes;
  if (open_writes > 0)
    CCTK_VERROR("CarpetX::write: time level %d of group %s has %d open write "
                "scope%s (CarpetX::write) on patch %d, refinement level %d; "
                "destroy that write_t before opening another",
                tl, groupdata.record.groupname.c_str(), open_writes,
                open_writes == 1 ? "" : "s", groupdata.patch, groupdata.level);
  return write_t(std::unique_ptr<write_t::impl>(new write_t::impl{
      group, tl,
      open_write(one_level(group.level), {.gi = group.gi,
                                          .tl = tl,
                                          .fills = leaves_valid,
                                          .keeps = leaves_valid,
                                          .event = event_name(what)})}));
}

struct scratch_t::impl {
  group_t group;
  std::size_t slot;
};

namespace {

GHExt::PatchData::LevelData::GroupData::scratch_slot_t &
resolve_scratch(const group_t group, const std::size_t slot) {
  auto &pool = driver().group(group).scratch_pool;
  if (slot >= pool.size() || !pool.at(slot).in_use)
    refuse(std::source_location::current(),
           "scratch buffer %d of group %s on patch %d, refinement level %d no "
           "longer exists: the level was remade while the scratch_t was held",
           int(slot), CCTK_FullGroupName(group.gi.index()), group.level.patch,
           group.level.level);
  return pool.at(slot);
}

} // namespace

scratch_t::scratch_t(std::unique_ptr<impl> p) noexcept : p(std::move(p)) {}

scratch_t::scratch_t(scratch_t &&) noexcept = default;

amrex::MultiFab &scratch_t::mfab() {
  const operation_scope_t op("CarpetX::scratch_t::mfab");
  if (!p)
    CCTK_ERROR("CarpetX::scratch_t::mfab() called on a moved-from scratch "
               "buffer");
  return *resolve_scratch(p->group, p->slot).mfab;
}

const amrex::MultiFab &scratch_t::mfab() const {
  const operation_scope_t op("CarpetX::scratch_t::mfab");
  if (!p)
    CCTK_ERROR("CarpetX::scratch_t::mfab() called on a moved-from scratch "
               "buffer");
  return *resolve_scratch(p->group, p->slot).mfab;
}

scratch_t::~scratch_t() {
  if (!p)
    return;
  const operation_scope_t op("CarpetX::scratch_t::~scratch_t");
  resolve_scratch(p->group, p->slot).in_use = false;
}

scratch_t scratch(const group_t group) {
  const operation_scope_t op("CarpetX::scratch");
  auto &groupdata = driver().group(group);
  auto &pool = groupdata.scratch_pool;
  std::size_t slot = 0;
  while (slot < pool.size() && pool.at(slot).in_use)
    ++slot;
  if (slot == pool.size()) {
    const amrex::MultiFab &mfab0 = *groupdata.timelevel(0).mfab;
    pool.push_back({std::make_unique<amrex::MultiFab>(
                        mfab0.boxArray(), mfab0.DistributionMap(),
                        mfab0.nComp(), mfab0.nGrowVect()),
                    false});
  }
  pool.at(slot).in_use = true;
  return scratch_t(
      std::unique_ptr<scratch_t::impl>(new scratch_t::impl{group, slot}));
}

valid_t valid(const group_t group, const int tl, const int vi) {
  const operation_scope_t op("CarpetX::valid");
  const auto &groupdata = driver().group(group);
  const auto &timelevel = groupdata.timelevel(tl);
  if (vi < 0 || vi >= groupdata.record.numvars)
    CCTK_VERROR("CarpetX::valid: variable index %d of group %s does not "
                "exist: it has %d variables",
                vi, groupdata.record.groupname.c_str(),
                groupdata.record.numvars);
  return timelevel.valid(vi);
}

void require_valid(const group_t group, const int tl, const valid_t where,
                   const char *const why) {
  const operation_scope_t op("CarpetX::require_valid");
  driver().group(group).timelevel(tl);
  require(one_level(group.level),
          {.gi = group.gi, .tl = tl, .where = where, .event = event_name(why)});
}

void invalidate(const window_t window, const std::span<const gi_t> groups,
                const int tl, const char *const why) {
  const operation_scope_t op("CarpetX::invalidate");
  const active_levels_t active_levels = require_operable(window, groups);
  const std::string event = event_name(why);
  for (const gi_t gi : groups) {
    active_levels.loop_serially([&](GHExt::PatchData::LevelData &leveldata) {
      leveldata.group(gi).timelevel(tl);
    });
    invalidate(active_levels,
               {.gi = gi, .tl = tl, .where = make_valid_all(), .event = event});
  }
}

void sync(const window_t window, const std::span<const gi_t> groups) {
  const operation_scope_t op("CarpetX::sync");
  const active_levels_t active_levels = require_operable(window, groups);
  SyncGroups(active_levels, driver().get_global_cctkGH(), groups);
}

void sync_ghosts_only(const window_t window,
                      const std::span<const gi_t> groups, const int tl) {
  const operation_scope_t op("CarpetX::sync_ghosts_only");
  const active_levels_t active_levels = require_operable(window, groups);
  SyncGroupsByDirIGhostOnly(active_levels, driver().get_global_cctkGH(), groups,
                            tl);
}

void prolongate_ghosts(const window_t window,
                       const std::span<const gi_t> groups, const int tl) {
  const operation_scope_t op("CarpetX::prolongate_ghosts");
  const active_levels_t active_levels = require_operable(window, groups);
  SyncGroupsByDirIProlongateOnly(active_levels, driver().get_global_cctkGH(),
                                 groups, tl);
}

void prolongate_aligned(const window_t window,
                        const std::span<const gi_t> groups, const int tl) {
  const operation_scope_t op("CarpetX::prolongate_aligned");
  const active_levels_t active_levels = require_operable(window, groups);
  SyncGroupsByDirIProlongateOnlyAligned(active_levels,
                                        driver().get_global_cctkGH(), groups, tl);
}

void restrict_no_poison(const window_t window,
                        const std::span<const gi_t> groups) {
  const operation_scope_t op("CarpetX::restrict_no_poison");
  require_operable(window, groups);
  RestrictNoPoison(driver().get_global_cctkGH(), window.min_level,
                   window.max_level, groups);
}

} // namespace CarpetX
