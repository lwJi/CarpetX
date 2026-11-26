#ifndef CARPETX_CARPETX_COMM_RK_BAND_PLACEMENT_INSIDE_HXX
#define CARPETX_CARPETX_COMM_RK_BAND_PLACEMENT_INSIDE_HXX

#include "internal.hxx"

#include "comm/prolongate_3d_rf2_inside.hxx"
#include <carpetx.hxx>

#include <AMReX_Box.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_FabArrayBase.H>
#include <AMReX_FabFactory.H>
#include <AMReX_IntVect.H>

#include <vector>

namespace CarpetX {

struct rk_band_layout_t {
  amrex::BoxArray crse, fine;
  amrex::DistributionMapping owner;
  const amrex::FabFactory<amrex::FArrayBox> &crse_factory, &fine_factory;
  std::vector<ghost_overlap_t> overlaps;
  amrex::BoxArray footprint;
};

class rk_band_cuts_t {
public:
  rk_band_layout_t layout_of(const amrex::FabArrayBase::FPinfo &fp,
                             const amrex::BoxArray &gba,
                             const amrex::DistributionMapping &dm,
                             const amrex::IntVect &nghost);

private:
  struct cut_t {
    amrex::BoxArray gba;
    amrex::DistributionMapping dm;
    amrex::IntVect nghost;
    amrex::Box domain;
    amrex::BoxArray fine;
    amrex::DistributionMapping owner;
    std::vector<ghost_overlap_t> overlaps;
  };
  std::vector<cut_t> cuts_;
};

} // namespace CarpetX

#endif
