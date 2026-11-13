#include "io/checkpoint.hxx"
#include "hierarchy/binding.hxx"
#include "hierarchy/clocks.hxx"
#include "hierarchy/driver.hxx"
#include "hierarchy/grid.hxx"
#include "hierarchy/group_tags.hxx"
#include "hierarchy/loops.hxx"
#include "hierarchy/operations.hxx"
#include "comm/buffers.hxx"
#include "comm/reflux.hxx"
#include "comm/subcycling.hxx"
#include "comm/comm.hxx"
#include "regrid/regrid.hxx"
#include "schedule/schedule.hxx"
#include "hierarchy/valid.hxx"

#include <carpetx.hxx>

#include <functional>
#include <optional>
#include <source_location>
#include <span>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace CarpetX {
namespace {

template <class Group>
concept record_takes = requires(GHExt &ext, Group group) { ext.record(group); };

template <class... Args>
concept ghext_level_takes =
    requires(GHExt &ext, Args... args) { ext.level(args...); };

template <class Patch>
concept ghext_patch_takes =
    requires(GHExt &ext, Patch patch) { ext.patch(patch); };

template <class Patch>
concept ghext_num_levels_takes =
    requires(const GHExt &ext, Patch patch) { ext.num_levels(patch); };

template <class Level>
concept patch_level_takes = requires(GHExt::PatchData &patchdata, Level level) {
  patchdata.level(level);
};

template <class Ext, class... Args>
concept patch_cctkGH_takes =
    requires(const Ext &ext, Args... args) { ext.get_patch_cctkGH(args...); };

template <class Ext, class... Args>
concept local_cctkGH_takes =
    requires(const Ext &ext, Args... args) { ext.get_local_cctkGH(args...); };

template <class... Args>
concept accumulate_fluxes_takes =
    requires(Args... args) { AccumulateFluxes(args...); };

template <class... Args>
concept store_rk_stage_takes =
    requires(Args... args) { StoreRKStage(args...); };

template <class... Args>
concept fill_rk_boundary_takes =
    requires(Args... args) { FillRKBoundary(args...); };

template <class Index>
concept adds_int = requires(Index index) { index + 1; };

template <class... Args>
concept checkpoint_bands_takes =
    requires(Args... args) { checkpoint_bands(args...); };

template <class... Args>
concept restore_bands_takes =
    requires(Args... args) { restore_bands(args...); };

template <class Bands, class... Args>
concept bands_sources_takes =
    requires(const Bands &bands, Args... args) { bands.sources(args...); };

template <class Bands, class... Args>
concept bands_stored_takes =
    requires(Bands &bands, Args... args) { bands.stored(args...); };

template <class Bands, class... Args>
concept bands_restore_takes =
    requires(Bands &bands, Args... args) { bands.restore(args...); };

template <class Bands, class... Args>
concept bands_holds_takes =
    requires(const Bands &bands, Args... args) { bands.holds(args...); };

template <class Bands>
concept reaches_bands_parent_step =
    requires(Bands &bands) { bands.parent_step_; };

template <class Bands>
concept reaches_bands_stages_stored =
    requires(Bands &bands) { bands.stages_stored_; };

template <class... Args>
concept finer_behind_takes =
    requires(const checkpoint_record_t &record, Args... args) {
      finer_level_is_behind(record, args...);
    };

template <class Kernel>
concept loop_parallel_takes =
    requires(window_t window, Kernel kernel) { loop_parallel(window, kernel); };

template <class Kernel>
concept loop_fine_to_coarse_takes = requires(window_t window, Kernel kernel) {
  loop_fine_to_coarse(window, kernel);
};

template <class Group>
concept level_group_takes = requires(GHExt::PatchData::LevelData &leveldata,
                                     Group group) { leveldata.group(group); };

template <class Timelevel>
concept group_timelevel_takes =
    requires(GHExt::PatchData::LevelData::GroupData &groupdata, Timelevel tl) {
      groupdata.timelevel(tl);
    };

template <class Timelevel>
concept array_timelevel_takes =
    requires(GHExt::GlobalData::ArrayGroupData &groupdata, Timelevel tl) {
      groupdata.timelevel(tl);
    };

template <class... Args>
concept ghost_only_takes = requires(Args... args) { SyncGhostsOnly(args...); };

template <class... Args>
concept restrict_takes = requires(Args... args) { Restrict(args...); };

template <class... Args>
concept restrict_no_poison_takes =
    requires(Args... args) { RestrictNoPoison(args...); };

template <class... Args>
concept reflux_takes = requires(Args... args) { Reflux(args...); };

using comm_groups_entry_t = void(window_t, std::span<const gi_t>);
using comm_timelevels_entry_t = void(window_t, std::span<const gi_t>,
                                     timelevels_t);

template <class... Args>
concept window_is_restricted_takes =
    requires(Args... args) { window_is_restricted(args...); };

template <class Window>
concept require_takes =
    requires(Window window, require_spec_t spec) { require(window, spec); };

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
    requires(const GHExt::GroupRecord &record, Variable vi) { record.var(vi); };

template <class Variable>
concept timelevel_valid_takes = requires(
    const GHExt::PatchData::LevelData::GroupData::timelevel_t &timelevel,
    Variable vi) { timelevel.valid(vi); };

template <class Variable>
concept array_timelevel_valid_takes =
    requires(const GHExt::GlobalData::ArrayGroupData::timelevel_t &timelevel,
             Variable vi) { timelevel.valid(vi); };

template <class Variable>
concept grid_ptr_takes =
    requires(const GridPtrDesc1 &grid, const amrex::Array4<CCTK_REAL> &vars,
             Variable vi) { grid.ptr(vars, vi); };

template <class Ext>
concept assigns_use_subcycling =
    requires(Ext &ext) { ext.use_subcycling = true; };

template <class Ext>
concept assigns_do_reflux = requires(Ext &ext) { ext.do_reflux = true; };

template <class Ext>
concept assigns_active_timelevels =
    requires(Ext &ext, gi_t gi) { ext.active_timelevels(gi) = 0; };

template <class Ext>
concept assigns_pending_timelevels =
    requires(Ext &ext) { ext.pending_timelevels_->at(0) = 0; };

template <class Ext>
concept assigns_storage_frozen =
    requires(Ext &ext) { ext.storage_frozen() = true; };

template <class Ext>
concept assigns_initialized = requires(Ext &ext) { ext.initialized_ = true; };

template <class Ext>
concept assigns_is_initialized =
    requires(Ext &ext) { ext.is_initialized() = true; };

template <class Ext>
concept assigns_epoch_member = requires(Ext &ext) { ext.epoch_ = 0; };

template <class Ext>
concept assigns_epoch = requires(Ext &ext) { ext.epoch() = 0; };

template <class Ext>
concept bumps_epoch = requires(Ext &ext) { ++ext.epoch(); };

template <class... Args>
concept create_hierarchy_takes =
    requires(Args... args) { create_hierarchy(args...); };

template <class Patch>
concept assigns_level_modified =
    requires(Patch &patch) { patch.level_modified.at(0) = true; };

template <class Patch>
concept marks_level_modified =
    requires(Patch &patch, level_t level) { patch.mark_level_modified(level); };

template <class Patch>
concept resets_level_modified =
    requires(Patch &patch) { patch.reset_level_modified(); };

template <class Ext>
concept regrids_amrcore = requires(Ext &ext, patch_t patch) {
  ext.patch(patch).amrcore->regrid(0, 0.0);
};

template <class Ext>
concept writes_timelevel = requires(Ext &ext, group_t group, tl_t tl) {
  ext.group(group).timelevel(tl).mfab->setVal(0);
};

template <class Ext>
concept writes_scratch = requires(Ext &ext, group_t group) {
  ext.group(group).scratch_pool.at(0).mfab->setVal(0);
};

template <class Scope, class... Args>
concept hands_out_mfab = requires(Scope scope, Args... args) {
  std::forward<Scope>(scope).mfab(args...);
};

template <class Scope>
concept hands_out_data =
    requires(Scope scope) { std::forward<Scope>(scope).data(); };

template <class Ext>
concept marks_initialized = requires(Ext &ext) { ext.mark_initialized(); };

template <class Ext>
concept assigns_integrator = requires(Ext &ext) { ext.integrator_.reset(); };

template <class Ext>
concept assigns_integrator_open =
    requires(Ext &ext) { ext.integrator_open_ = true; };

template <class Ext>
concept records_integrator = requires(Ext &ext) {
  ext.record_integrator(GHExt::integrator_t{}, std::source_location::current());
};

template <class Ext>
concept freezes_integrator = requires(Ext &ext) {
  ext.freeze_integrator(std::source_location::current());
};

template <class Ext>
concept opens_mutable_door = requires { Ext::mutable_instance(); };

template <class Writer>
concept regrids = requires(Writer &writer, level_t level) {
  writer.regrid(level, CCTK_REAL{});
};

template <class Writer>
concept adds_patches =
    requires(Writer &writer, patch_t patch) { writer.add_patch(patch); };

template <class Writer>
concept erases_levels = requires(Writer &writer, level_t level) {
  writer.erase_levels_from(level);
};

template <class Writer>
concept makes_levels =
    requires(Writer &writer, level_t level, const amrex::BoxArray &ba,
             const amrex::DistributionMapping &dm) {
      writer.make_level(level, ba, dm, std::string());
    };

template <class Writer>
concept hands_out_levels = requires(Writer &writer) { writer.levels; };

template <class Writer>
concept hands_out_hierarchy = requires(Writer &writer) { writer.hierarchy; };

template <class Writer>
concept makes_amrcore =
    std::is_invocable_v<amrcore_factory, Writer &, patch_t,
                        const amrex::RealBox &, int, const amrex::Vector<int> &,
                        const amrex::Array<int, dim> &>;

template <class Group>
concept rotates = requires(Group &group) { group.rotate_timelevels(); };

template <class Binding>
concept rotates_levels_unlent = requires(GHExt &ext, window_t window, gi_t gi) {
  Binding::rotate(ext, window, gi);
};

template <class Binding>
concept rotates_arrays_unlent =
    requires(GHExt &ext, gi_t gi) { Binding::rotate(ext, gi); };

template <class T>
concept holds_indextype = requires(const T &t) { t.indextype; };

template <class T>
concept holds_nghostzones = requires(const T &t) { t.nghostzones; };

template <class T>
concept holds_boundaries = requires(const T &t) { t.boundaries; };

template <class T>
concept holds_parities = requires(const T &t) { t.parities; };

template <class T>
concept holds_dirichlet_values = requires(const T &t) { t.dirichlet_values; };

template <class T>
concept holds_robin_values = requires(const T &t) { t.robin_values; };

template <class T>
concept holds_fluxes = requires(const T &t) { t.fluxes; };

template <class Group>
concept parses_grid_function_tags =
    requires(Group gi) { get_group_indextype(gi); } ||
    requires(Group gi) { get_group_nghostzones(gi); } ||
    requires(Group gi) { get_group_boundaries(gi); } ||
    requires(Group gi) { get_group_parities(gi); } ||
    requires(Group gi) { get_group_dirichlet_values(gi); } ||
    requires(Group gi) { get_group_robin_values(gi); } ||
    requires(Group gi) { get_group_fluxes(gi); };

template <class Record>
concept makes_grid_function =
    requires(gi_t gi) { Record::make_grid_function(gi, 0); };

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
static_assert(local_cctkGH_takes<GHExt::PatchData::LevelData, tile_t>);
static_assert(!local_cctkGH_takes<GHExt::PatchData::LevelData, int>);
static_assert(!local_cctkGH_takes<GHExt::PatchData::LevelData, component_t>);
static_assert(!local_cctkGH_takes<GHExt, int, int, int>);

static_assert(std::is_same_v<decltype(group_buffers_t::rk_bands),
                             std::optional<rk_bands_t>>);

static_assert(
    std::is_same_v<decltype(grid_buffer_t::mfab), const amrex::MultiFab *>);
static_assert(
    std::is_same_v<decltype(checkpoint_band_t::band), const amrex::MultiFab *>);
static_assert(
    std::is_same_v<decltype(restore_band_t::band), amrex::MultiFab *>);
static_assert(std::is_same_v<decltype(checkpoint_bands(std::declval<level_t>(),
                                                       std::declval<gi_t>(),
                                                       std::declval<rat64>())),
                             std::vector<checkpoint_band_t>>);
static_assert(std::is_same_v<decltype(restore_bands(std::declval<level_t>(),
                                                    std::declval<gi_t>(),
                                                    std::declval<rat64>())),
                             std::vector<restore_band_t>>);

static_assert(accumulate_fluxes_takes<level_t, stage_t, CCTK_REAL>);
static_assert(!accumulate_fluxes_takes<level_t, int, CCTK_REAL>);
static_assert(!accumulate_fluxes_takes<level_t, tl_t, CCTK_REAL>);
static_assert(!accumulate_fluxes_takes<int, int, int, CCTK_REAL>);

static_assert(store_rk_stage_takes<level_t, std::span<const gi_t>,
                                   std::span<const gi_t>, stage_t>);
static_assert(!store_rk_stage_takes<level_t, std::span<const gi_t>,
                                    std::span<const gi_t>, int>);

static_assert(fill_rk_boundary_takes<level_t, std::span<const gi_t>, stage_t,
                                     const std::string &>);
static_assert(!fill_rk_boundary_takes<level_t, std::span<const gi_t>, stage_t>);
static_assert(!fill_rk_boundary_takes<level_t, std::span<const gi_t>, int,
                                      const std::string &>);
static_assert(!fill_rk_boundary_takes<level_t, std::span<const gi_t>, tl_t,
                                      const std::string &>);
static_assert(
    !fill_rk_boundary_takes<level_t, std::span<const gi_t>, tl_t, stage_t,
                            CCTK_REAL, CCTK_REAL, const std::string &>);
static_assert(
    !fill_rk_boundary_takes<level_t, std::span<const gi_t>, stage_t, CCTK_REAL,
                            CCTK_REAL, const std::string &>);
static_assert(!fill_rk_boundary_takes<level_t, std::span<const gi_t>, stage_t,
                                      rat64, const std::string &>);

static_assert(!adds_int<stage_t>);
static_assert(!std::is_convertible_v<stage_t, int>);
static_assert(!std::is_convertible_v<int, stage_t>);
static_assert(!std::is_convertible_v<stage_t, tl_t>);
static_assert(!std::is_convertible_v<tl_t, stage_t>);
static_assert(!std::is_convertible_v<stage_t, component_t>);
static_assert(!std::is_convertible_v<component_t, stage_t>);
static_assert(!std::is_convertible_v<stage_t, tile_t>);
static_assert(!std::is_convertible_v<tile_t, stage_t>);
static_assert(next_index(stage_t{1}) == stage_t{2});
static_assert(all_rk_stages.contains(stage_t{1}));
static_assert(all_rk_stages.contains(stage_t{max_num_rk_stages}));
static_assert(!all_rk_stages.contains(stage_t{0}));
static_assert(!all_rk_stages.contains(stage_t{max_num_rk_stages + 1}));
static_assert(rk_stages(3).contains(stage_t{3}));
static_assert(rk_stages(3).first() == stage_t{1});
static_assert(rk_stages(3).last() == stage_t{3});
static_assert(rk_stages(3).offset(stage_t{1}) == 0);
static_assert(all_rk_stages.offset(stage_t{max_num_rk_stages}) ==
              max_num_rk_stages - 1);
static_assert(all_rk_stages.first() == stage_t{1});
static_assert(all_rk_stages.last() == stage_t{max_num_rk_stages});
static_assert(!rk_stages(3).contains(stage_t{4}));
static_assert(!rk_stages(0).contains(stage_t{1}));

static_assert(!substep_t{0, rat64(1) / 2}.aligned());
static_assert(substep_t{rat64(1) / 2, 1}.aligned());
static_assert(rk_substep_ratio * time_refinement == 1);
static_assert(substeps_per_coarsest_step(0) == 1);
static_assert(substeps_per_coarsest_step(3) ==
              time_refinement * time_refinement * time_refinement);

static_assert(checkpoint_bands_takes<level_t, gi_t, rat64>);
static_assert(!checkpoint_bands_takes<level_t, gi_t>);
static_assert(!checkpoint_bands_takes<int, int, gi_t, rat64>);
static_assert(!checkpoint_bands_takes<level_t, int, rat64>);

static_assert(restore_bands_takes<level_t, gi_t, rat64>);
static_assert(!restore_bands_takes<level_t, gi_t>);
static_assert(!restore_bands_takes<level_t, int, rat64>);
static_assert(!restore_bands_takes<int, int, gi_t, rat64>);
static_assert(bands_stored_takes<rk_bands_t, rat64>);
static_assert(!bands_stored_takes<rk_bands_t>);
static_assert(bands_restore_takes<rk_bands_t, rat64>);
static_assert(!bands_restore_takes<rk_bands_t>);
static_assert(!bands_holds_takes<rk_bands_t, rat64>);
static_assert(bands_sources_takes<rk_bands_t, rat64, fill_site_t>);
static_assert(!bands_sources_takes<rk_bands_t>);
static_assert(!reaches_bands_parent_step<rk_bands_t>);
static_assert(!reaches_bands_stages_stored<rk_bands_t>);

static_assert(finer_behind_takes<level_t>);
static_assert(!finer_behind_takes<int, int>);

static_assert(loop_parallel_takes<std::function<void(
                  const GHExt::PatchData::LevelData &, tile_t, const cGH *)>>);
static_assert(!loop_parallel_takes<std::function<void(
                  const GHExt::PatchData::LevelData &, int, const cGH *)>>);
static_assert(
    !loop_parallel_takes<std::function<void(const GHExt::PatchData::LevelData &,
                                            int, int, const cGH *)>>);
static_assert(
    !loop_parallel_takes<std::function<void(const GHExt::PatchData::LevelData &,
                                            component_t, const cGH *)>>);
static_assert(!loop_parallel_takes<
              std::function<void(const GHExt::PatchData::LevelData &,
                                 component_t, tile_t, const cGH *)>>);
static_assert(!loop_parallel_takes<std::function<
                  void(GHExt::PatchData::LevelData &, tile_t, const cGH *)>>);
static_assert(
    !loop_parallel_takes<std::function<void(int, int, int, const cGH *)>>);
static_assert(
    !loop_parallel_takes<std::function<void(level_t, tile_t, const cGH *)>>);

static_assert(!adds_int<component_t>);
static_assert(!adds_int<tile_t>);
static_assert(!std::is_convertible_v<component_t, int>);
static_assert(!std::is_convertible_v<int, component_t>);
static_assert(!std::is_convertible_v<tile_t, int>);
static_assert(!std::is_convertible_v<int, tile_t>);
static_assert(!std::is_convertible_v<component_t, tile_t>);
static_assert(!std::is_convertible_v<tile_t, component_t>);
static_assert(std::is_constructible_v<tile_t, component_t, int>);
static_assert(!std::is_constructible_v<tile_t, int>);
static_assert(!std::is_constructible_v<tile_t, int, int>);
static_assert(!std::is_constructible_v<tile_t, int, component_t>);
static_assert(
    std::is_same_v<decltype(std::declval<tile_t>().component()), component_t>);
static_assert(tile_t{component_t{3}, 7}.component() == component_t{3});
static_assert(tile_t{component_t{3}, 7}.index() == 7);
static_assert(tile_t{component_t{3}, 7} != tile_t{component_t{4}, 7});
static_assert(tile_t{component_t{3}, 7} != tile_t{component_t{3}, 8});
static_assert(
    std::is_same_v<decltype(tile_of(std::declval<amrex::MFIter &>())), tile_t>);
static_assert(
    std::is_same_v<decltype(std::declval<const MFPointer &>().index()),
                   component_t>);
static_assert(std::is_constructible_v<
              GridPtrDesc1, const GHExt::PatchData::LevelData &,
              const GHExt::PatchData::LevelData::GroupData &, component_t>);
static_assert(!std::is_constructible_v<
              GridPtrDesc1, const GHExt::PatchData::LevelData &,
              const GHExt::PatchData::LevelData::GroupData &, int>);
static_assert(std::is_same_v<decltype(components_of(0)), component_range>);
static_assert(components_of(3).contains(component_t{2}));
static_assert(!components_of(3).contains(component_t{3}));

static_assert(loop_fine_to_coarse_takes<
              std::function<void(const GHExt::PatchData::LevelData &)>>);
static_assert(!loop_fine_to_coarse_takes<
              std::function<void(GHExt::PatchData::LevelData &)>>);
static_assert(!loop_fine_to_coarse_takes<std::function<void(level_t)>>);

static_assert(level_group_takes<gi_t>);
static_assert(!level_group_takes<int>);

static_assert(group_timelevel_takes<tl_t>);
static_assert(!group_timelevel_takes<gi_t>);
static_assert(!group_timelevel_takes<int>);

static_assert(array_timelevel_takes<tl_t>);
static_assert(!array_timelevel_takes<gi_t>);
static_assert(!array_timelevel_takes<int>);

static_assert(restrict_takes<window_t, std::span<const gi_t>>);
static_assert(!restrict_takes<int, int, std::span<const gi_t>>);
static_assert(restrict_no_poison_takes<window_t, std::span<const gi_t>>);
static_assert(!restrict_no_poison_takes<int, int, std::span<const gi_t>>);
static_assert(reflux_takes<window_t>);
static_assert(!reflux_takes<int, int>);
static_assert(window_is_restricted_takes<window_t>);
static_assert(!window_is_restricted_takes<int, int>);
static_assert(require_takes<window_t>);
static_assert(!require_takes<level_t>);

static_assert(ghost_only_takes<window_t, std::span<const gi_t>, timelevels_t>);
static_assert(!ghost_only_takes<window_t, std::span<const gi_t>>);
static_assert(!ghost_only_takes<window_t, std::span<const gi_t>, tl_t>);
static_assert(!ghost_only_takes<window_t, std::span<const gi_t>, int>);

static_assert(std::is_same_v<decltype(Restrict), comm_groups_entry_t>);
static_assert(
    std::is_same_v<decltype(ProlongateAfterRestriction), void(window_t)>);
static_assert(std::is_same_v<decltype(RestrictNoPoison), comm_groups_entry_t>);
static_assert(std::is_same_v<decltype(SyncGroups), comm_groups_entry_t>);
static_assert(
    std::is_same_v<decltype(ProlongateGhosts), comm_timelevels_entry_t>);
static_assert(
    std::is_same_v<decltype(ProlongateAligned), comm_timelevels_entry_t>);
static_assert(
    std::is_same_v<decltype(SyncGhostsOnly), comm_timelevels_entry_t>);
static_assert(std::is_same_v<decltype(Reflux), void(window_t)>);
static_assert(std::is_same_v<decltype(RecoverRKBoundary), void(window_t)>);

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
static_assert(std::is_same_v<decltype(tiletag_t::tile), tile_t>);
static_assert(std::is_same_v<decltype(tiletag_t::gi), gi_t>);
static_assert(std::is_same_v<decltype(tiletag_t::vi), vi_t>);
static_assert(std::is_same_v<decltype(tiletag_t::tl), tl_t>);
static_assert(sizeof(tiletag_t) == 7 * sizeof(int));
static_assert(std::is_same_v<decltype(write_spec_t::vi), std::optional<vi_t>>);
static_assert(
    std::is_same_v<decltype(require_spec_t::vi), std::optional<vi_t>>);
static_assert(
    std::is_same_v<decltype(invalidate_spec_t::vi), std::optional<vi_t>>);

static_assert(std::is_same_v<
              decltype(std::declval<const GHExt::GroupRecord &>().variables()),
              index_range<vi_t>>);
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

static_assert(std::is_same_v<decltype(std::declval<const GHExt::PatchData &>()
                                          .level_handles()),
                             level_range>);
static_assert(
    std::is_same_v<decltype(*std::declval<level_range>().begin()), level_t>);
static_assert(next_index(level_t{patch_t{0}, 0}) == level_t{patch_t{0}, 1});
static_assert(level_range{level_t{patch_t{1}, 0}, level_t{patch_t{1}, 2}}
                  .contains(level_t{patch_t{1}, 1}));
static_assert(!level_range{level_t{patch_t{1}, 0}, level_t{patch_t{1}, 2}}
                   .contains(level_t{patch_t{0}, 1}));
static_assert(
    std::is_same_v<decltype(std::declval<const GHExt &>().patch_handles()),
                   index_range<patch_t>>);
static_assert(next_index(patch_t{0}) == patch_t{1});

static_assert(std::is_same_v<decltype(GHExt::use_subcycling), const bool>);
static_assert(std::is_same_v<decltype(GHExt::do_reflux), const bool>);
static_assert(!assigns_use_subcycling<GHExt>);
static_assert(!assigns_do_reflux<GHExt>);
static_assert(!std::is_default_constructible_v<GHExt>);
static_assert(std::is_constructible_v<GHExt, GHExt::config_t>);
static_assert(!std::is_convertible_v<GHExt::config_t, GHExt>);
static_assert(!std::is_constructible_v<GHExt, bool, bool>);
static_assert(create_hierarchy_takes<GHExt::config_t>);
static_assert(!create_hierarchy_takes<bool, bool>);
static_assert(
    std::is_same_v<decltype(create_hierarchy(std::declval<GHExt::config_t>())),
                   std::unique_ptr<hierarchy_writers_t>>);
static_assert(
    std::is_same_v<decltype(hierarchy_writers_t::clock), clock_writer_t>);
static_assert(!std::is_default_constructible_v<clock_writer_t>);
static_assert(!std::is_copy_constructible_v<clock_writer_t>);
static_assert(!std::is_move_constructible_v<clock_writer_t>);
static_assert(std::is_same_v<decltype(hierarchy_writers_t::structure),
                             structure_writer_t>);
static_assert(!std::is_default_constructible_v<structure_writer_t>);
static_assert(!std::is_copy_constructible_v<structure_writer_t>);
static_assert(!std::is_move_constructible_v<structure_writer_t>);
static_assert(!hands_out_levels<hierarchy_writers_t>);
static_assert(!hands_out_levels<structure_writer_t>);
static_assert(!std::is_default_constructible_v<level_writer_t>);
static_assert(!std::is_copy_constructible_v<level_writer_t>);
static_assert(!std::is_move_constructible_v<level_writer_t>);
static_assert(regrids<structure_writer_t>);
static_assert(!regrids<level_writer_t>);
static_assert(adds_patches<structure_writer_t>);
static_assert(!adds_patches<level_writer_t>);
static_assert(erases_levels<level_writer_t>);
static_assert(!erases_levels<structure_writer_t>);
static_assert(makes_levels<level_writer_t>);
static_assert(!makes_levels<structure_writer_t>);
static_assert(makes_amrcore<level_writer_t>);
static_assert(!makes_amrcore<structure_writer_t>);
static_assert(std::is_same_v<decltype(&make_cactus_amrcore), amrcore_factory>);
static_assert(
    std::is_same_v<decltype(hierarchy_writers_t::registry), registry_writer_t>);
static_assert(!std::is_default_constructible_v<registry_writer_t>);
static_assert(!std::is_copy_constructible_v<registry_writer_t>);
static_assert(!std::is_move_constructible_v<registry_writer_t>);
static_assert(
    std::is_same_v<decltype(hierarchy_writers_t::scratch), scratch_writer_t>);
static_assert(!std::is_default_constructible_v<scratch_writer_t>);
static_assert(!std::is_copy_constructible_v<scratch_writer_t>);
static_assert(!std::is_move_constructible_v<scratch_writer_t>);
static_assert(!std::is_default_constructible_v<hierarchy_writers_t>);
static_assert(!std::is_copy_constructible_v<hierarchy_writers_t>);
static_assert(!std::is_move_constructible_v<hierarchy_writers_t>);
static_assert(std::is_same_v<decltype(Initialise),
                             int(tFleshConfig *, clock_writer_t &,
                                 structure_writer_t &, registry_writer_t &)>);
static_assert(
    std::is_same_v<decltype(Evolve), int(tFleshConfig *, clock_writer_t &,
                                         structure_writer_t &)>);

static_assert(
    std::is_same_v<decltype(std::declval<const GHExt &>().active_timelevels(
                       std::declval<gi_t>())),
                   int>);
static_assert(std::is_same_v<
              decltype(std::declval<const GHExt &>().storage_frozen()), bool>);
static_assert(!assigns_active_timelevels<GHExt>);
static_assert(!assigns_pending_timelevels<GHExt>);
static_assert(!assigns_storage_frozen<GHExt>);

static_assert(
    std::is_same_v<decltype(std::declval<const GHExt &>().integrator()),
                   const std::optional<GHExt::integrator_t> &>);
static_assert(std::is_same_v<
              decltype(std::declval<const GHExt::integrator_t &>().stages()),
              stage_range>);
static_assert(!assigns_integrator<GHExt>);
static_assert(!assigns_integrator_open<GHExt>);
static_assert(!records_integrator<GHExt>);
static_assert(!freezes_integrator<GHExt>);

static_assert(std::is_same_v<
              decltype(std::declval<const GHExt &>().is_initialized()), bool>);
static_assert(!assigns_initialized<GHExt>);
static_assert(!assigns_is_initialized<GHExt>);

static_assert(
    std::is_same_v<decltype(std::declval<const GHExt &>().epoch()), CCTK_INT>);
static_assert(!assigns_epoch_member<GHExt>);
static_assert(!assigns_epoch<GHExt>);
static_assert(!bumps_epoch<GHExt>);

static_assert(std::is_same_v<decltype(std::declval<const GHExt::PatchData &>()
                                          .modified_levels()),
                             std::optional<window_t::bounds_t>>);
static_assert(!assigns_level_modified<GHExt::PatchData>);
static_assert(!marks_level_modified<GHExt::PatchData>);
static_assert(!resets_level_modified<GHExt::PatchData>);

static_assert(std::is_same_v<decltype(driver()), const GHExt &>);
static_assert(regrids_amrcore<GHExt>);
static_assert(!regrids_amrcore<std::remove_reference_t<decltype(driver())>>);
static_assert(writes_timelevel<GHExt>);
static_assert(!writes_timelevel<std::remove_reference_t<decltype(driver())>>);
static_assert(writes_scratch<GHExt>);
static_assert(!writes_scratch<std::remove_reference_t<decltype(driver())>>);
static_assert(!std::is_copy_constructible_v<write_scope_t>);
static_assert(!std::is_copy_assignable_v<write_scope_t>);
static_assert(hands_out_mfab<write_scope_t &, level_t>);
static_assert(!hands_out_mfab<write_scope_t, level_t>);
static_assert(!hands_out_mfab<const write_scope_t &, level_t>);
static_assert(hands_out_mfab<write_scopes_t &, level_t, gi_t, tl_t>);
static_assert(!hands_out_mfab<write_scopes_t, level_t, gi_t, tl_t>);
static_assert(!hands_out_mfab<const write_scopes_t &, level_t, gi_t, tl_t>);
static_assert(hands_out_data<array_write_scope_t &>);
static_assert(!hands_out_data<array_write_scope_t>);
static_assert(!hands_out_data<const array_write_scope_t &>);
static_assert(!marks_initialized<GHExt>);
static_assert(!marks_initialized<std::remove_reference_t<decltype(driver())>>);
static_assert(!opens_mutable_door<GHExt>);

static_assert(!rotates<GHExt::PatchData::LevelData::GroupData>);
static_assert(!rotates<GHExt::GlobalData::ArrayGroupData>);
static_assert(!rotates_levels_unlent<cctkGH_binding>);
static_assert(!rotates_arrays_unlent<cctkGH_binding>);
static_assert(!std::is_constructible_v<clock_writer_t, GHExt &>);
static_assert(!std::is_constructible_v<structure_writer_t, GHExt &>);
static_assert(!std::is_constructible_v<level_writer_t, GHExt &>);
static_assert(!std::is_constructible_v<registry_writer_t, GHExt &>);
static_assert(!std::is_constructible_v<scratch_writer_t, GHExt &>);
static_assert(!std::is_constructible_v<hierarchy_writers_t, GHExt &>);
static_assert(
    !std::is_constructible_v<hierarchy_writers_t, std::unique_ptr<GHExt>>);
static_assert(!hands_out_hierarchy<hierarchy_writers_t>);

static_assert(std::is_same_v<decltype(std::declval<const GHExt::GroupRecord &>()
                                          .grid_function()),
                             const GHExt::GroupRecord::grid_function_t &>);
static_assert(std::is_same_v<
              decltype(GHExt::PatchData::LevelData::GroupData::grid_function),
              const GHExt::GroupRecord::grid_function_t &>);
static_assert(holds_indextype<GHExt::GroupRecord::grid_function_t>);
static_assert(!holds_indextype<GHExt::PatchData::LevelData::GroupData>);
static_assert(holds_nghostzones<GHExt::GroupRecord::grid_function_t>);
static_assert(!holds_nghostzones<GHExt::PatchData::LevelData::GroupData>);
static_assert(holds_boundaries<GHExt::GroupRecord::grid_function_t>);
static_assert(!holds_boundaries<GHExt::PatchData::LevelData::GroupData>);
static_assert(holds_parities<GHExt::GroupRecord::grid_function_t>);
static_assert(!holds_parities<GHExt::PatchData::LevelData::GroupData>);
static_assert(holds_dirichlet_values<GHExt::GroupRecord::grid_function_t>);
static_assert(!holds_dirichlet_values<GHExt::PatchData::LevelData::GroupData>);
static_assert(holds_robin_values<GHExt::GroupRecord::grid_function_t>);
static_assert(!holds_robin_values<GHExt::PatchData::LevelData::GroupData>);
static_assert(holds_fluxes<GHExt::GroupRecord::grid_function_t>);
static_assert(!holds_fluxes<GHExt::PatchData::LevelData::GroupData>);
static_assert(!parses_grid_function_tags<gi_t>);
static_assert(!makes_grid_function<GHExt::GroupRecord>);

} // namespace
} // namespace CarpetX
