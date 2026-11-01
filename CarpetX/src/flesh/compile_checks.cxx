#include "io/checkpoint.hxx"
#include "hierarchy/driver.hxx"
#include "hierarchy/loops.hxx"
#include "comm/reflux.hxx"
#include "comm/subcycling.hxx"
#include "comm/comm.hxx"
#include "hierarchy/valid.hxx"

#include <carpetx.hxx>

#include <functional>
#include <optional>
#include <span>
#include <type_traits>
#include <utility>

namespace CarpetX {
namespace {

template <class Group>
concept record_takes = requires(GHExt & ext, Group group) {
  ext.record(group);
};

template <class... Args>
concept ghext_level_takes = requires(GHExt & ext, Args... args) {
  ext.level(args...);
};

template <class Patch>
concept ghext_patch_takes = requires(GHExt & ext, Patch patch) {
  ext.patch(patch);
};

template <class Patch>
concept ghext_num_levels_takes = requires(const GHExt &ext, Patch patch) {
  ext.num_levels(patch);
};

template <class Level>
concept patch_level_takes =
    requires(GHExt::PatchData & patchdata, Level level) {
  patchdata.level(level);
};

template <class Ext, class... Args>
concept patch_cctkGH_takes = requires(const Ext &ext, Args... args) {
  ext.get_patch_cctkGH(args...);
};

template <class Ext, class... Args>
concept local_cctkGH_takes = requires(const Ext &ext, Args... args) {
  ext.get_local_cctkGH(args...);
};

template <class... Args>
concept ensure_rk_buffers_takes = requires(Args... args) {
  EnsureRKBuffers(args...);
};

template <class... Args>
concept accumulate_fluxes_takes = requires(Args... args) {
  AccumulateFluxes(args...);
};

template <class... Args> concept rk_source_band_takes = requires(Args... args) {
  rk_source_band(args...);
};

template <class... Args>
concept finer_behind_takes =
    requires(const checkpoint_record_t &record, Args... args) {
  finer_level_is_behind(record, args...);
};

template <class Kernel>
concept loop_parallel_takes = requires(window_t window, Kernel kernel) {
  loop_parallel(window, kernel);
};

template <class Kernel>
concept loop_fine_to_coarse_takes = requires(window_t window, Kernel kernel) {
  loop_fine_to_coarse(window, kernel);
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
    requires(window_t window, const cGH *cctkGH, Args... args) {
  SyncGroupsByDirI(window, cctkGH, args...);
};

template <class... Args>
concept ghost_only_takes =
    requires(window_t window, const cGH *cctkGH, std::span<const gi_t> groups,
             Args... args) {
  SyncGroupsByDirIGhostOnly(window, cctkGH, groups, args...);
};

template <class... Args>
concept restrict_takes = requires(const cGH *cctkGH, Args... args) {
  Restrict(cctkGH, args...);
};

template <class... Args>
concept restrict_no_poison_takes = requires(const cGH *cctkGH, Args... args) {
  RestrictNoPoison(cctkGH, args...);
};

template <class... Args>
concept reflux_takes = requires(const cGH *cctkGH, Args... args) {
  Reflux(cctkGH, args...);
};

template <class... Args>
concept window_is_restricted_takes = requires(Args... args) {
  window_is_restricted(args...);
};

template <class Window>
concept require_takes = requires(Window window, require_spec_t spec) {
  require(window, spec);
};

template <class Group, class Timelevel>
concept write_spec_names = requires(Group group, Timelevel tl) {
  write_spec_t{.gi = group, .tl = tl};
};

template <class Variable>
concept write_spec_variable = requires(gi_t gi, tl_t tl, Variable vi) {
  write_spec_t{.gi = gi, .tl = tl, .vi = vi};
};

template <class Variable>
concept record_var_takes =
    requires(const GHExt::GroupRecord &record, Variable vi) {
  record.var(vi);
};

template <class Variable>
concept timelevel_valid_takes = requires(
    const GHExt::PatchData::LevelData::GroupData::timelevel_t &timelevel,
    Variable vi) {
  timelevel.valid(vi);
};

template <class Variable>
concept array_timelevel_valid_takes =
    requires(const GHExt::GlobalData::ArrayGroupData::timelevel_t &timelevel,
             Variable vi) {
  timelevel.valid(vi);
};

template <class Variable>
concept grid_ptr_takes =
    requires(const GridPtrDesc1 &grid, const amrex::Array4<CCTK_REAL> &vars,
             Variable vi) {
  grid.ptr(vars, vi);
};

static_assert(record_takes<gi_t>);
static_assert(!record_takes<int>);

static_assert(ghext_level_takes<level_t>);
static_assert(!ghext_level_takes<int, int>);
static_assert(!ghext_level_takes<patch_t, int>);

static_assert(ghext_patch_takes<patch_t>);
static_assert(!ghext_patch_takes<int>);

static_assert(ghext_num_levels_takes<patch_t>);
static_assert(!ghext_num_levels_takes<int>);

static_assert(patch_level_takes<level_t>);
static_assert(!patch_level_takes<int>);

static_assert(patch_cctkGH_takes<GHExt::PatchData::LevelData>);
static_assert(!patch_cctkGH_takes<GHExt, int, int>);
static_assert(local_cctkGH_takes<GHExt::PatchData::LevelData, int>);
static_assert(!local_cctkGH_takes<GHExt, int, int, int>);

static_assert(ensure_rk_buffers_takes<level_t, gi_t>);
static_assert(!ensure_rk_buffers_takes<int, int, gi_t>);

static_assert(accumulate_fluxes_takes<level_t, int, CCTK_REAL>);
static_assert(!accumulate_fluxes_takes<int, int, int, CCTK_REAL>);

static_assert(rk_source_band_takes<level_t, gi_t, band_kind>);
static_assert(!rk_source_band_takes<int, int, gi_t, band_kind>);

static_assert(finer_behind_takes<level_t>);
static_assert(!finer_behind_takes<int, int>);

static_assert(loop_parallel_takes<std::function<void(
                  GHExt::PatchData::LevelData &, int, int, const cGH *)> >);
static_assert(!loop_parallel_takes<
              std::function<void(int, int, int, int, const cGH *)> >);
static_assert(
    !loop_parallel_takes<std::function<void(level_t, int, int, const cGH *)> >);

static_assert(loop_fine_to_coarse_takes<
              std::function<void(GHExt::PatchData::LevelData &)> >);
static_assert(!loop_fine_to_coarse_takes<std::function<void(level_t)> >);

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

static_assert(restrict_takes<window_t, std::span<const gi_t> >);
static_assert(!restrict_takes<int, int, std::span<const gi_t> >);
static_assert(restrict_no_poison_takes<window_t, std::span<const gi_t> >);
static_assert(!restrict_no_poison_takes<int, int, std::span<const gi_t> >);
static_assert(reflux_takes<window_t>);
static_assert(!reflux_takes<int, int>);
static_assert(window_is_restricted_takes<window_t>);
static_assert(!window_is_restricted_takes<int, int>);
static_assert(require_takes<window_t>);
static_assert(!require_takes<level_t>);

static_assert(ghost_only_takes<timelevels_t>);
static_assert(!ghost_only_takes<>);
static_assert(!ghost_only_takes<tl_t>);
static_assert(!ghost_only_takes<int>);

static_assert(write_spec_names<gi_t, tl_t>);
static_assert(!write_spec_names<tl_t, gi_t>);
static_assert(!write_spec_names<int, tl_t>);
static_assert(!write_spec_names<gi_t, int>);

static_assert(write_spec_variable<vi_t>);
static_assert(!write_spec_variable<gi_t>);
static_assert(!write_spec_variable<var_t>);
static_assert(!write_spec_variable<int>);

static_assert(record_var_takes<vi_t>);
static_assert(!record_var_takes<gi_t>);
static_assert(!record_var_takes<var_t>);
static_assert(!record_var_takes<int>);

static_assert(timelevel_valid_takes<vi_t>);
static_assert(!timelevel_valid_takes<tl_t>);
static_assert(!timelevel_valid_takes<int>);

static_assert(array_timelevel_valid_takes<vi_t>);
static_assert(!array_timelevel_valid_takes<tl_t>);
static_assert(!array_timelevel_valid_takes<int>);

static_assert(grid_ptr_takes<vi_t>);
static_assert(!grid_ptr_takes<int>);

static_assert(std::is_same_v<decltype(GHExt::GroupRecord::groupindex), gi_t>);
static_assert(std::is_same_v<decltype(GHExt::PatchData::patch), patch_t>);
static_assert(
    std::is_same_v<decltype(GHExt::PatchData::LevelData::level), level_t>);
static_assert(
    std::is_same_v<decltype(GHExt::PatchData::LevelData::GroupData::level),
                   const level_t>);
static_assert(std::is_same_v<decltype(tiletag_t::level), level_t>);
static_assert(std::is_same_v<decltype(tiletag_t::gi), gi_t>);
static_assert(std::is_same_v<decltype(tiletag_t::vi), vi_t>);
static_assert(std::is_same_v<decltype(tiletag_t::tl), tl_t>);
static_assert(sizeof(tiletag_t) == 6 * sizeof(int));
static_assert(std::is_same_v<decltype(write_spec_t::vi), std::optional<vi_t> >);
static_assert(
    std::is_same_v<decltype(require_spec_t::vi), std::optional<vi_t> >);
static_assert(
    std::is_same_v<decltype(invalidate_spec_t::vi), std::optional<vi_t> >);

static_assert(std::is_same_v<
              decltype(std::declval<const GHExt::GroupRecord &>().variables()),
              index_range<vi_t> >);
static_assert(std::is_same_v<decltype(std::declval<const GHExt::GroupRecord &>()
                                          .var(std::declval<vi_t>())),
                             var_t>);

static_assert(std::is_same_v<
              decltype(std::declval<const GHExt::GroupRecord &>().timelevels()),
              tl_range>);
static_assert(
    std::is_same_v<decltype(*std::declval<tl_range>().begin()), tl_t>);
static_assert(tl_range{tl_t{0}, tl_t{2}}.contains(tl_t{1}));
static_assert(!tl_range{tl_t{0}, tl_t{2}}.contains(tl_t{2}));

} // namespace
} // namespace CarpetX
