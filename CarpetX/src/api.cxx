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

namespace {

std::string event_name(const char *const what) { return what ? what : ""; }

} // namespace

void require_no_open_write(
    const char *const what,
    const GHExt::PatchData::LevelData::GroupData &groupdata) {
  const int open_writes = groupdata.open_write_count();
  if (open_writes > 0)
    CCTK_VERROR("%s: group %s has %d open write scope%s (CarpetX::write) on "
                "patch %d, refinement level %d; destroy the write_t first",
                what, groupdata.group.groupname.c_str(), open_writes,
                open_writes == 1 ? "" : "s", groupdata.patch, groupdata.level);
}

active_levels_t require_operable(const char *const what, const window_t window,
                                 const std::vector<int> &groups) {
  for (const int gi : groups)
    driver().group(gi, what);
  const active_levels_t active_levels(window);
  active_levels.loop_serially([&](GHExt::PatchData::LevelData &leveldata) {
    for (const int gi : groups)
      if (driver().group(gi, what).has_storage())
        require_no_open_write(what, leveldata.group(gi, what));
  });
  return active_levels;
}

int num_patches() { return driver().num_patches(); }

int num_levels(const int patch) {
  return driver().patch(patch, "CarpetX::num_levels").num_levels();
}

level_t parent(const level_t level) {
  driver().level(level, "CarpetX::parent");
  if (level.level == 0)
    CCTK_VERROR("CarpetX level 0 of patch %d has no parent level", level.patch);
  return level_t{level.patch, level.level - 1};
}

rat64 iteration(const level_t level) {
  return driver().level(level, "CarpetX::iteration").iteration;
}

rat64 delta_iteration(const level_t level) {
  return driver().level(level, "CarpetX::delta_iteration").delta_iteration;
}

const amrex::Geometry &geometry(const level_t level) {
  driver().level(level, "CarpetX::geometry");
  return driver()
      .patch(level.patch, "CarpetX::geometry")
      .amrcore->Geom(level.level);
}

std::array<std::array<symmetry_t, Loop::dim>, 2>
symmetries(const level_t level) {
  driver().level(level, "CarpetX::symmetries");
  return driver().patch(level.patch, "CarpetX::symmetries").symmetries;
}

int numvars(const group_t group) {
  return driver().group(group, "CarpetX::numvars").group.numvars;
}

std::array<int, Loop::dim> indextype(const group_t group) {
  return driver().group(group, "CarpetX::indextype").indextype;
}

std::array<int, Loop::dim> nghostzones(const group_t group) {
  return driver().group(group, "CarpetX::nghostzones").nghostzones;
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
  return *driver()
              .group(group, "CarpetX::data")
              .timelevel(tl, "CarpetX::data")
              .mfab;
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
  if (!p)
    CCTK_ERROR("CarpetX::write_t::mfab() called on a moved-from write scope");
  return *driver()
              .group(p->group, "CarpetX::write_t::mfab")
              .timelevel(p->tl, "CarpetX::write_t::mfab")
              .mfab;
}

write_t::~write_t() = default;

write_t write(const group_t group, const int tl, const valid_t leaves_valid,
              const char *const what) {
  auto &groupdata = driver().group(group, "CarpetX::write");
  const int open_writes = groupdata.timelevel(tl, "CarpetX::write").open_writes;
  if (open_writes > 0)
    CCTK_VERROR("CarpetX::write: time level %d of group %s has %d open write "
                "scope%s (CarpetX::write) on patch %d, refinement level %d; "
                "destroy that write_t before opening another",
                tl, groupdata.group.groupname.c_str(), open_writes,
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
resolve_scratch(const char *const what, const group_t group,
                const std::size_t slot) {
  auto &pool = driver().group(group, what).scratch_pool;
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
  auto &groupdata = driver().group(group, "CarpetX::scratch");
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
  const auto &groupdata = driver().group(group, "CarpetX::valid");
  const auto &timelevel = groupdata.timelevel(tl, "CarpetX::valid");
  if (vi < 0 || vi >= groupdata.group.numvars)
    CCTK_VERROR("CarpetX::valid: variable index %d of group %s does not "
                "exist: it has %d variables",
                vi, groupdata.group.groupname.c_str(), groupdata.group.numvars);
  return timelevel.valid(vi);
}

void require_valid(const group_t group, const int tl, const valid_t where,
                   const char *const why) {
  driver()
      .group(group, "CarpetX::require_valid")
      .timelevel(tl, "CarpetX::require_valid");
  require(one_level(group.level),
          {.gi = group.gi, .tl = tl, .where = where, .event = event_name(why)});
}

void invalidate(const window_t window, const std::vector<int> &groups,
                const int tl, const char *const why) {
  const active_levels_t active_levels =
      require_operable("CarpetX::invalidate", window, groups);
  const std::string event = event_name(why);
  for (const int gi : groups) {
    active_levels.loop_serially([&](GHExt::PatchData::LevelData &leveldata) {
      leveldata.group(gi, "CarpetX::invalidate")
          .timelevel(tl, "CarpetX::invalidate");
    });
    invalidate(active_levels,
               {.gi = gi, .tl = tl, .where = make_valid_all(), .event = event});
  }
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
