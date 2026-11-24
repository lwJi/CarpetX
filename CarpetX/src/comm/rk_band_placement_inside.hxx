#ifndef CARPETX_CARPETX_COMM_RK_BAND_PLACEMENT_INSIDE_HXX
#define CARPETX_CARPETX_COMM_RK_BAND_PLACEMENT_INSIDE_HXX

#include "internal.hxx"

#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_FabArrayBase.H>
#include <AMReX_FabFactory.H>

namespace CarpetX {

struct rk_band_layout_t {
  amrex::BoxArray crse, fine;
  amrex::DistributionMapping owner;
  const amrex::FabFactory<amrex::FArrayBox> &crse_factory, &fine_factory;
};

rk_band_layout_t rk_band_layout_of(const amrex::FabArrayBase::FPinfo &fp);

} // namespace CarpetX

#endif
