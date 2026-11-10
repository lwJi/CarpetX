#ifndef CARPETX_CARPETX_COMM_BUFFERS_HXX
#define CARPETX_CARPETX_COMM_BUFFERS_HXX

#include "internal.hxx"

#include "hierarchy/driver.hxx"
#include <carpetx.hxx>

#include <AMReX_FabArrayBase.H>
#include <AMReX_FluxRegister.H>
#include <AMReX_Geometry.H>
#include <AMReX_Interpolater.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Periodicity.H>

#include <array>
#include <memory>
#include <optional>
#include <source_location>
#include <span>
#include <vector>

namespace CarpetX {

struct checkpoint_band_t;

class rk_bands_t {
public:
  rk_bands_t(const amrex::FabArrayBase::FPinfo &fp, int numvars,
             stage_range stages);

  void store_old(const amrex::MultiFab &parent_state,
                 const amrex::Periodicity &period);
  void store_stage(stage_t stage, const amrex::MultiFab &parent_rhs,
                   const amrex::Periodicity &period);

  struct sources_t {
    const amrex::MultiFab &old;
    std::span<const amrex::MultiFab> ks;
  };
  sources_t sources() const;

  amrex::MultiFab &crse_patch();
  amrex::MultiFab &fine_patch();

private:
  amrex::MultiFab old_source_;
  std::vector<amrex::MultiFab> ks_source_;
  amrex::MultiFab crse_patch_, fine_patch_;
  friend std::vector<checkpoint_band_t> checkpoint_bands(level_t, gi_t);
};

struct group_buffers_t {
  std::unique_ptr<amrex::FluxRegister> freg, freg_scratch;
  std::optional<rk_bands_t> rk_bands;
};

struct level_buffers_t {
  std::vector<group_buffers_t> groups;
  std::array<std::unique_ptr<amrex::MultiFab>, dim> face_area;
  std::unique_ptr<amrex::MultiFab> cell_volume;
};

const amrex::FabArrayBase::FPinfo &
rk_fill_fpinfo(const amrex::FabArrayBase &fine,
               amrex::Interpolater *interpolator, const amrex::Geometry &fgeom,
               const amrex::Geometry &cgeom);

level_buffers_ptr make_level_buffers(level_t level, const amrex::BoxArray &ba,
                                     const amrex::DistributionMapping &dm);

level_buffers_t &buffers_of(const GHExt::PatchData::LevelData &leveldata);

group_buffers_t &
buffers_of(const GHExt::PatchData::LevelData &leveldata, gi_t gi,
           std::source_location site = std::source_location::current());

} // namespace CarpetX

#endif
