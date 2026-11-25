#ifndef CARPETX_CARPETX_COMM_BUFFERS_INSIDE_HXX
#define CARPETX_CARPETX_COMM_BUFFERS_INSIDE_HXX

#include "internal.hxx"

#include "config/stages.hxx"
#include "hierarchy/model.hxx"
#include "comm/rk_band_placement_inside.hxx"
#include "comm/rk_bands_inside.hxx"
#include <carpetx.hxx>

#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_FabArrayBase.H>
#include <AMReX_FluxRegister.H>
#include <AMReX_IntVect.H>
#include <AMReX_MultiFab.H>

#include <optional>
#include <source_location>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace CarpetX {

struct restore_band_t;
class checkpoint_read_t;

struct flux_buffers_t {
  flux_buffers_t(const amrex::BoxArray &ba,
                 const amrex::DistributionMapping &dm,
                 const amrex::IntVect &ratio, int fine_level, int numvars);

  amrex::FluxRegister reg, scratch;
};

class flux_register_writer_t;
using sealed_flux_buffers_t =
    written_by<flux_buffers_t, flux_register_writer_t>;

class flux_register_writer_t {
  flux_register_writer_t() = default;
  friend void AccumulateFluxes(level_t level, stage_t stage, CCTK_REAL weight);
  friend std::vector<restore_band_t>
  restore_bands(level_t level, gi_t gi, rat64 parent_step,
                const checkpoint_read_t &reading);

public:
  flux_register_writer_t(const flux_register_writer_t &) = delete;
  flux_register_writer_t &operator=(const flux_register_writer_t &) = delete;

  flux_buffers_t &open(const sealed_flux_buffers_t &sealed) const {
    return sealed.open(*this);
  }

  static void reflux_into(amrex::MultiFab &state,
                          const sealed_flux_buffers_t &sealed,
                          const amrex::MultiFab &volume, const int numvars,
                          const amrex::Geometry &geom) {
    flux_register_writer_t().open(sealed).reg.Reflux(state, volume, 1.0, 0, 0,
                                                     numvars, geom);
  }
};

class rk_fill_patch_writer_t;
using sealed_rk_fill_patches_t =
    written_by<rk_fill_patches_t, rk_fill_patch_writer_t>;

class rk_fill_patch_writer_t {
  rk_fill_patch_writer_t() = default;
  friend void FillRKBoundary(level_t child, std::span<const gi_t> states,
                             stage_t completed, const std::string &event);

public:
  rk_fill_patch_writer_t(const rk_fill_patch_writer_t &) = delete;
  rk_fill_patch_writer_t &operator=(const rk_fill_patch_writer_t &) = delete;

  rk_fill_patches_t &open(const sealed_rk_fill_patches_t &sealed) const {
    return sealed.open(*this);
  }
};

class rk_source_writer_t;
using sealed_rk_sources_t = written_by<rk_sources_t, rk_source_writer_t>;

class rk_source_writer_t {
  rk_source_writer_t() = default;
  friend void StoreRKOldState(level_t level, std::span<const gi_t> var_groups,
                              tl_t tl);
  friend void StoreRKStage(level_t level, std::span<const gi_t> var_groups,
                           std::span<const gi_t> rhs_groups, stage_t stage);
  friend std::vector<restore_band_t>
  restore_bands(level_t level, gi_t gi, rat64 parent_step,
                const checkpoint_read_t &reading);

public:
  rk_source_writer_t(const rk_source_writer_t &) = delete;
  rk_source_writer_t &operator=(const rk_source_writer_t &) = delete;

  rk_sources_t &open(const sealed_rk_sources_t &sealed) const {
    return sealed.open(*this);
  }
};

struct rk_buffers_t {
  rk_buffers_t(const rk_band_layout_t &layout, const int numvars,
               const stage_range stages)
      : sources(std::in_place, layout, numvars, stages),
        fill_patches(std::in_place, layout, numvars) {}

  sealed_rk_sources_t sources;
  sealed_rk_fill_patches_t fill_patches;
};

struct group_buffers_t {
  std::optional<sealed_flux_buffers_t> flux;
  std::optional<rk_buffers_t> rk;
};

struct level_buffers_t {
  std::vector<group_buffers_t> groups;
  std::optional<amrex::MultiFab> cell_volume;
};

const level_buffers_t &buffers_of(const GHExt::PatchData::LevelData &leveldata);

const group_buffers_t &
buffers_of(const GHExt::PatchData::LevelData &leveldata, gi_t gi,
           std::source_location site = std::source_location::current());

} // namespace CarpetX

#endif
