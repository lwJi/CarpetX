#include "driver.hxx"
#include "schedule.hxx"
#include "sync_restrict_internal.hxx"
#include "valid.hxx"

#include <carpetx.hxx>

#include <span>
#include <type_traits>
#include <utility>

namespace CarpetX {
namespace {

template <class Group>
concept record_takes = requires(GHExt & ext, Group group) {
  ext.record(group);
};

template <class Group>
concept level_group_takes =
    requires(GHExt::PatchData::LevelData & leveldata, Group group) {
  leveldata.group(group);
};

template <class Timelevel>
concept group_timelevel_takes =
    requires(GHExt::PatchData::LevelData::GroupData & groupdata, Timelevel tl) {
  groupdata.timelevel(tl);
};

template <class Timelevel>
concept array_timelevel_takes =
    requires(GHExt::GlobalData::ArrayGroupData & groupdata, Timelevel tl) {
  groupdata.timelevel(tl);
};

template <class... Args>
concept sync_by_dir_takes =
    requires(const active_levels_t &levels, const cGH *cctkGH, Args... args) {
  SyncGroupsByDirI(levels, cctkGH, args...);
};

template <class... Args>
concept ghost_only_takes =
    requires(const active_levels_t &levels, const cGH *cctkGH,
             std::span<const gi_t> groups, Args... args) {
  SyncGroupsByDirIGhostOnly(levels, cctkGH, groups, args...);
};

template <class Group, class Timelevel>
concept write_spec_names = requires(Group group, Timelevel tl) {
  write_spec_t{.gi = group, .tl = tl};
};

static_assert(record_takes<gi_t>);
static_assert(!record_takes<int>);

static_assert(level_group_takes<gi_t>);
static_assert(!level_group_takes<int>);

static_assert(group_timelevel_takes<tl_t>);
static_assert(!group_timelevel_takes<gi_t>);
static_assert(!group_timelevel_takes<int>);

static_assert(array_timelevel_takes<tl_t>);
static_assert(!array_timelevel_takes<gi_t>);
static_assert(!array_timelevel_takes<int>);

static_assert(sync_by_dir_takes<std::span<const gi_t> >);
static_assert(!sync_by_dir_takes<int, const int *, const int *>);

static_assert(ghost_only_takes<timelevels_t>);
static_assert(!ghost_only_takes<>);
static_assert(!ghost_only_takes<tl_t>);
static_assert(!ghost_only_takes<int>);

static_assert(write_spec_names<gi_t, tl_t>);
static_assert(!write_spec_names<tl_t, gi_t>);
static_assert(!write_spec_names<int, tl_t>);
static_assert(!write_spec_names<gi_t, int>);

static_assert(std::is_same_v<decltype(GHExt::GroupRecord::groupindex), gi_t>);
static_assert(std::is_same_v<decltype(tiletag_t::gi), gi_t>);
static_assert(std::is_same_v<decltype(tiletag_t::tl), tl_t>);
static_assert(sizeof(tiletag_t) == 6 * sizeof(int));

static_assert(std::is_same_v<
              decltype(std::declval<const GHExt::GroupRecord &>().timelevels()),
              tl_range>);
static_assert(
    std::is_same_v<decltype(*std::declval<tl_range>().begin()), tl_t>);
static_assert(tl_range{tl_t{0}, tl_t{2}}.contains(tl_t{1}));
static_assert(!tl_range{tl_t{0}, tl_t{2}}.contains(tl_t{2}));

} // namespace
} // namespace CarpetX
