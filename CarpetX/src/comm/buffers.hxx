#ifndef CARPETX_CARPETX_COMM_BUFFERS_HXX
#define CARPETX_CARPETX_COMM_BUFFERS_HXX

#include "internal.hxx"

#include "hierarchy/model.hxx"
#include "comm/rk_bands.hxx"
#include <carpetx.hxx>

#include <AMReX_FluxRegister.H>
#include <AMReX_MultiFab.H>

#include <array>
#include <memory>
#include <optional>
#include <source_location>
#include <vector>

namespace CarpetX {

struct restore_band_t;
class checkpoint_read_t;

struct flux_registers_t {
  const_propagating_ptr<amrex::FluxRegister> reg;
};

struct flux_scratch_t {
  const_propagating_ptr<amrex::FluxRegister> reg;
};

class flux_register_writer_t;
using sealed_flux_registers_t =
    written_by<flux_registers_t, flux_register_writer_t>;
using sealed_flux_scratch_t =
    written_by<flux_scratch_t, flux_register_writer_t>;

class flux_register_writer_t {
  flux_register_writer_t() = default;
  friend void AccumulateFluxes(level_t level, stage_t stage, CCTK_REAL weight);
  friend std::vector<restore_band_t>
  restore_bands(level_t level, gi_t gi, rat64 parent_step,
                const checkpoint_read_t &reading);

public:
  flux_register_writer_t(const flux_register_writer_t &) = delete;
  flux_register_writer_t &operator=(const flux_register_writer_t &) = delete;

  flux_registers_t &open(const sealed_flux_registers_t &sealed) const {
    return sealed.open(*this);
  }
  flux_scratch_t &open(const sealed_flux_scratch_t &sealed) const {
    return sealed.open(*this);
  }
};

struct group_buffers_t {
  sealed_flux_registers_t registers;
  sealed_flux_scratch_t scratch;
  std::optional<rk_bands_t> rk_bands;
};

struct flux_geometry_t {
  std::array<amrex::MultiFab, dim> face_area;
  amrex::MultiFab cell_volume;
};

struct level_buffers_t {
  std::vector<group_buffers_t> groups;
  std::optional<flux_geometry_t> geometry;
};

level_buffers_ptr make_level_buffers(level_t level, const amrex::BoxArray &ba,
                                     const amrex::DistributionMapping &dm);

level_buffers_t &buffers_of(const GHExt::PatchData::LevelData &leveldata);

group_buffers_t &
buffers_of(const GHExt::PatchData::LevelData &leveldata, gi_t gi,
           std::source_location site = std::source_location::current());

} // namespace CarpetX

#endif
