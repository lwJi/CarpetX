#ifndef CARPETX_CARPETX_COMM_BUFFERS_HXX
#define CARPETX_CARPETX_COMM_BUFFERS_HXX

#include "internal.hxx"

#include "hierarchy/driver.hxx"
#include <carpetx.hxx>

#include <AMReX_FluxRegister.H>
#include <AMReX_MultiFab.H>

#include <array>
#include <memory>
#include <vector>

namespace CarpetX {

struct group_buffers_t {
  std::array<std::unique_ptr<amrex::MultiFab>, max_num_rk_stages>
      ks_source_band;
  std::unique_ptr<amrex::MultiFab> old_source_band;
  std::unique_ptr<amrex::MultiFab> rk_crse_patch, rk_fine_patch;
  std::unique_ptr<amrex::FluxRegister> freg_scratch;
};

struct level_buffers_t {
  std::vector<group_buffers_t> groups;
  std::array<std::unique_ptr<amrex::MultiFab>, dim> face_area;
  std::unique_ptr<amrex::MultiFab> cell_volume;
};

level_buffers_t &buffers_of(const GHExt::PatchData::LevelData &leveldata);

group_buffers_t &buffers_of(const GHExt::PatchData::LevelData &leveldata,
                            gi_t gi);

} // namespace CarpetX

#endif
