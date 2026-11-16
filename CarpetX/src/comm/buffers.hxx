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

struct group_buffers_t {
  std::unique_ptr<amrex::FluxRegister> freg, freg_scratch;
  std::optional<rk_bands_t> rk_bands;
};

struct level_buffers_t {
  std::vector<group_buffers_t> groups;
  std::array<std::unique_ptr<amrex::MultiFab>, dim> face_area;
  std::unique_ptr<amrex::MultiFab> cell_volume;
};

level_buffers_ptr make_level_buffers(level_t level, const amrex::BoxArray &ba,
                                     const amrex::DistributionMapping &dm);

level_buffers_t &buffers_of(const GHExt::PatchData::LevelData &leveldata);

group_buffers_t &
buffers_of(const GHExt::PatchData::LevelData &leveldata, gi_t gi,
           std::source_location site = std::source_location::current());

} // namespace CarpetX

#endif
