#include <carpetx.hxx>

#include "api/api.hxx"
#include "hierarchy/driver.hxx"
#include "hierarchy/loops.hxx"
#include "comm/comm.hxx"
#include "traversal/traversal.hxx"

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

void require_window_bounds(const int min_level, const int max_level,
                           const int min_patch, const int max_patch,
                           const std::source_location site) {
  const GHExt &ext = driver();
  if (min_level < 0 || max_level > ext.num_levels() || min_level > max_level)
    refuse(site, "CarpetX window levels [%d, %d) must lie within [0, %d)",
           min_level, max_level, ext.num_levels());
  if (min_patch < 0 || max_patch > ext.num_patches() || min_patch > max_patch)
    refuse(site, "CarpetX window patches [%d, %d) must lie within [0, %d)",
           min_patch, max_patch, ext.num_patches());
}

void require_window_level(const int min_level, const int max_level,
                          const int min_patch, const int max_patch,
                          const std::source_location site) {
  if (min_level >= max_level || min_patch >= max_patch)
    refuse(site,
           "CarpetX window of patches [%d, %d), refinement levels [%d, %d) "
           "holds no level",
           min_patch, max_patch, min_level, max_level);
}

} // namespace

void require_no_open_write(
    const GHExt::PatchData::LevelData::GroupData &groupdata) {
  const int open_writes = groupdata.open_write_count();
  if (open_writes > 0)
    refuse(std::source_location::current(),
           "group %s has %d open write scope%s (CarpetX::write) on patch %d, "
           "refinement level %d; destroy the write_t first",
           groupdata.record.groupname.c_str(), open_writes,
           open_writes == 1 ? "" : "s", groupdata.level.patch().index(),
           groupdata.level.index());
}

void require_operable(const window_t window,
                      const std::span<const gi_t> groups) {
  for (const gi_t gi : groups)
    driver().record(gi);
  loop_serially(window, [&](GHExt::PatchData::LevelData &leveldata) {
    for (const gi_t gi : groups)
      if (driver().record(gi).has_storage())
        require_no_open_write(leveldata.group(gi));
  });
}

variable_t variable(const var_t var) {
  const int gi = CCTK_GroupIndexFromVarI(var.index());
  if (gi < 0)
    CCTK_VERROR("CarpetX::variable: variable index %d does not exist: there "
                "are %d variables",
                var.index(), CCTK_NumVars());
  return variable_t{gi_t{gi}, vi_t{var.index() - CCTK_FirstVarIndexI(gi)}};
}

int num_patches() { return driver().num_patches(); }

int num_levels(const patch_t patch) {
  const operation_scope_t op("CarpetX::num_levels");
  return driver().patch(patch).num_levels();
}

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
  return driver().patch(level.patch()).amrcore->Geom(level.index());
}

std::array<std::array<symmetry_t, Loop::dim>, 2>
symmetries(const level_t level) {
  const operation_scope_t op("CarpetX::symmetries");
  driver().level(level);
  return driver().patch(level.patch()).symmetries;
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

window_t window_t::one_level(const level_t level,
                             const std::source_location site) {
  const int patch = level.patch().index();
  require_window_bounds(level.index(), level.index() + 1, patch, patch + 1,
                        site);
  return window_t{level.index(), level.index() + 1, patch, patch + 1};
}

window_t window_t::levels(const int min_level, const int max_level,
                          const std::source_location site) {
  const int num_patches = driver().num_patches();
  require_window_bounds(min_level, max_level, 0, num_patches, site);
  return window_t{min_level, max_level, 0, num_patches};
}

window_t window_t::narrow(const int min_level, const int max_level,
                          const std::source_location site) const {
  require_window_bounds(min_level, max_level, min_patch_, max_patch_, site);
  return window_t{min_level, max_level, min_patch_, max_patch_};
}

level_t window_t::coarsest(const std::source_location site) const {
  require_window_level(min_level_, max_level_, min_patch_, max_patch_, site);
  return level_t{patch_t{min_patch_}, min_level_};
}

level_t window_t::finest(const std::source_location site) const {
  require_window_level(min_level_, max_level_, min_patch_, max_patch_, site);
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
    const std::function<void(level_t level, int index, int component,
                             const cGH *cctkGH)> &kernel) const {
  CarpetX::loop_parallel(
      *this, [&](const GHExt::PatchData::LevelData &leveldata, const int index,
                 const int component, const cGH *const cctkGH) {
        kernel(leveldata.level, index, component, cctkGH);
      });
}

window_t active_window() {
  return scheduled_window("CarpetX::active_window()");
}

window_t all_levels() { return window_t::levels(0, driver().num_levels()); }

const amrex::MultiFab &data(const group_t group, const tl_t tl) {
  const operation_scope_t op("CarpetX::data");
  return *driver().group(group).timelevel(tl).mfab;
}

struct write_t::impl {
  group_t group;
  tl_t tl;
  write_scope_t scope;
};

write_t::write_t(std::unique_ptr<impl> p) noexcept : p(std::move(p)) {}

write_t::write_t(write_t &&) noexcept = default;

amrex::MultiFab &write_t::mfab() & {
  const operation_scope_t op("CarpetX::write_t::mfab");
  if (!p)
    CCTK_ERROR("CarpetX::write_t::mfab() called on a moved-from write scope");
  return *driver().group(p->group).timelevel(p->tl).mfab;
}

write_t::~write_t() = default;

write_t write(const group_t group, const tl_t tl, const valid_t leaves_valid,
              const char *const what) {
  const operation_scope_t op("CarpetX::write");
  auto &groupdata = driver().group(group);
  const int open_writes = groupdata.timelevel(tl).open_writes;
  if (open_writes > 0)
    CCTK_VERROR("CarpetX::write: time level %d of group %s has %d open write "
                "scope%s (CarpetX::write) on patch %d, refinement level %d; "
                "destroy that write_t before opening another",
                tl.index(), groupdata.record.groupname.c_str(), open_writes,
                open_writes == 1 ? "" : "s", groupdata.level.patch().index(),
                groupdata.level.index());
  return write_t(std::unique_ptr<write_t::impl>(
      new write_t::impl{group, tl,
                        open_write(window_t::one_level(group.level),
                                   {.gi = group.gi,
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
           int(slot), CCTK_FullGroupName(group.gi.index()),
           group.level.patch().index(), group.level.index());
  return pool.at(slot);
}

} // namespace

scratch_t::scratch_t(std::unique_ptr<impl> p) noexcept : p(std::move(p)) {}

scratch_t::scratch_t(scratch_t &&) noexcept = default;

amrex::MultiFab &scratch_t::mfab() & {
  const operation_scope_t op("CarpetX::scratch_t::mfab");
  if (!p)
    CCTK_ERROR("CarpetX::scratch_t::mfab() called on a moved-from scratch "
               "buffer");
  return *resolve_scratch(p->group, p->slot).mfab;
}

const amrex::MultiFab &scratch_t::mfab() const & {
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
    const amrex::MultiFab &mfab0 = *groupdata.timelevel(tl_t{0}).mfab;
    pool.push_back({std::make_unique<amrex::MultiFab>(
                        mfab0.boxArray(), mfab0.DistributionMap(),
                        mfab0.nComp(), mfab0.nGrowVect()),
                    false});
  }
  pool.at(slot).in_use = true;
  return scratch_t(
      std::unique_ptr<scratch_t::impl>(new scratch_t::impl{group, slot}));
}

valid_t valid(const group_t group, const tl_t tl, const vi_t vi) {
  const operation_scope_t op("CarpetX::valid");
  const auto &groupdata = driver().group(group);
  const auto &timelevel = groupdata.timelevel(tl);
  if (!groupdata.record.variables().contains(vi))
    CCTK_VERROR("CarpetX::valid: variable index %d of group %s does not "
                "exist: it has %d variables",
                vi.index(), groupdata.record.groupname.c_str(),
                groupdata.record.numvars);
  return timelevel.valid(vi);
}

void require_valid(const group_t group, const tl_t tl, const valid_t where,
                   const char *const why) {
  const operation_scope_t op("CarpetX::require_valid");
  driver().group(group).timelevel(tl);
  require(window_t::one_level(group.level),
          {.gi = group.gi, .tl = tl, .where = where, .event = event_name(why)});
}

void invalidate(const window_t window, const std::span<const gi_t> groups,
                const tl_t tl, const char *const why) {
  const operation_scope_t op("CarpetX::invalidate");
  require_operable(window, groups);
  const std::string event = event_name(why);
  for (const gi_t gi : groups) {
    loop_serially(window, [&](GHExt::PatchData::LevelData &leveldata) {
      leveldata.group(gi).timelevel(tl);
    });
    invalidate(window,
               {.gi = gi, .tl = tl, .where = make_valid_all(), .event = event});
  }
}

void sync(const window_t window, const std::span<const gi_t> groups) {
  const operation_scope_t op("CarpetX::sync");
  require_operable(window, groups);
  SyncGroups(window, driver().get_global_cctkGH(), groups);
}

namespace {

void sync_ghosts_only_timelevels(const window_t window,
                                 const std::span<const gi_t> groups,
                                 const timelevels_t tls) {
  const operation_scope_t op("CarpetX::sync_ghosts_only");
  require_operable(window, groups);
  SyncGroupsByDirIGhostOnly(window, driver().get_global_cctkGH(), groups, tls);
}

void prolongate_ghosts_timelevels(const window_t window,
                                  const std::span<const gi_t> groups,
                                  const timelevels_t tls) {
  const operation_scope_t op("CarpetX::prolongate_ghosts");
  require_operable(window, groups);
  SyncGroupsByDirIProlongateOnly(window, driver().get_global_cctkGH(), groups,
                                 tls);
}

void prolongate_aligned_timelevels(const window_t window,
                                   const std::span<const gi_t> groups,
                                   const timelevels_t tls) {
  const operation_scope_t op("CarpetX::prolongate_aligned");
  require_operable(window, groups);
  SyncGroupsByDirIProlongateOnlyAligned(window, driver().get_global_cctkGH(),
                                        groups, tls);
}

} // namespace

void sync_ghosts_only(const window_t window,
                      const std::span<const gi_t> groups) {
  sync_ghosts_only_timelevels(window, groups, timelevels_t::all_but_oldest());
}

void sync_ghosts_only(const window_t window, const std::span<const gi_t> groups,
                      const tl_t tl) {
  sync_ghosts_only_timelevels(window, groups, timelevels_t::one(tl));
}

void prolongate_ghosts(const window_t window,
                       const std::span<const gi_t> groups) {
  prolongate_ghosts_timelevels(window, groups, timelevels_t::all_but_oldest());
}

void prolongate_ghosts(const window_t window,
                       const std::span<const gi_t> groups, const tl_t tl) {
  prolongate_ghosts_timelevels(window, groups, timelevels_t::one(tl));
}

void prolongate_aligned(const window_t window,
                        const std::span<const gi_t> groups) {
  prolongate_aligned_timelevels(window, groups, timelevels_t::all_but_oldest());
}

void prolongate_aligned(const window_t window,
                        const std::span<const gi_t> groups, const tl_t tl) {
  prolongate_aligned_timelevels(window, groups, timelevels_t::one(tl));
}

void restrict_no_poison(const window_t window,
                        const std::span<const gi_t> groups) {
  const operation_scope_t op("CarpetX::restrict_no_poison");
  require_operable(window, groups);
  RestrictNoPoison(driver().get_global_cctkGH(), window, groups);
}

} // namespace CarpetX
