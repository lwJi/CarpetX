#include "driver.hxx"
#include "schedule.hxx"
#include "valid.hxx"

#include <carpetx.hxx>

#include <span>
#include <type_traits>

namespace CarpetX {
namespace {

template <class Group>
concept record_takes = requires(GHExt &ext, Group group) { ext.record(group); };

template <class Group>
concept level_group_takes =
    requires(GHExt::PatchData::LevelData &leveldata, Group group) {
      leveldata.group(group);
    };

template <class... Args>
concept sync_by_dir_takes =
    requires(const active_levels_t &levels, const cGH *cctkGH, Args... args) {
      SyncGroupsByDirI(levels, cctkGH, args...);
    };

template <class Group>
concept write_spec_names = requires(Group group, int tl) {
  write_spec_t{.gi = group, .tl = tl};
};

static_assert(record_takes<gi_t>);
static_assert(!record_takes<int>);

static_assert(level_group_takes<gi_t>);
static_assert(!level_group_takes<int>);

static_assert(sync_by_dir_takes<std::span<const gi_t> >);
static_assert(!sync_by_dir_takes<int, const int *, const int *>);

static_assert(write_spec_names<gi_t>);
static_assert(!write_spec_names<int>);

static_assert(std::is_same_v<decltype(GHExt::GroupRecord::groupindex), gi_t>);
static_assert(std::is_same_v<decltype(tiletag_t::gi), gi_t>);

} // namespace
} // namespace CarpetX
