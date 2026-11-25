#ifndef CARPETX_CARPETX_COMM_RK_BAND_PLACEMENT_INSIDE_HXX
#define CARPETX_CARPETX_COMM_RK_BAND_PLACEMENT_INSIDE_HXX

#include "internal.hxx"

#include <carpetx.hxx>

#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_FabArrayBase.H>
#include <AMReX_FabFactory.H>
#include <AMReX_IntVect.H>
#include <AMReX_Periodicity.H>

#include <cstdint>

namespace CarpetX {

struct rk_band_layout_t {
  amrex::BoxArray crse, fine;
  amrex::DistributionMapping owner;
  const amrex::FabFactory<amrex::FArrayBox> &crse_factory, &fine_factory;
  amrex::BoxArray footprint;
};

rk_band_layout_t rk_band_layout_of(const amrex::FabArrayBase::FPinfo &fp,
                                   const amrex::BoxArray &gba,
                                   const amrex::DistributionMapping &dm,
                                   const amrex::IntVect &nghost);

struct rk_band_locality_t {
  std::int64_t stored_local_bytes = 0, stored_remote_bytes = 0;
  rk_band_locality_t &operator+=(const rk_band_locality_t &other);
};

rk_band_locality_t
rk_band_locality_of(const rk_band_layout_t &layout, int numvars,
                    const amrex::BoxArray &parent_gba,
                    const amrex::DistributionMapping &parent_dm,
                    const amrex::Periodicity &period);

void report_rk_band_locality(level_t level, int iteration,
                             const rk_band_locality_t &total);

} // namespace CarpetX

#endif
