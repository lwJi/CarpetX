#ifndef CARPETX_CARPETX_COMM_RK_BANDS_INSIDE_HXX
#define CARPETX_CARPETX_COMM_RK_BANDS_INSIDE_HXX

#include "internal.hxx"

#include "config/stages.hxx"
#include "comm/prolongate_3d_rf2_inside.hxx"
#include "comm/rk_band_placement_inside.hxx"
#include <carpetx.hxx>

#include <AMReX_FabArrayBase.H>
#include <AMReX_Geometry.H>
#include <AMReX_Interpolater.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Periodicity.H>

#include <optional>
#include <source_location>
#include <span>
#include <vector>

namespace CarpetX {

struct fill_site_t {
  level_t level;
  gi_t gi;
};

class rk_sources_t {
public:
  rk_sources_t(const rk_band_layout_t &layout, int numvars, stage_range stages);

  struct store_t {
    rk_sources_t &bands;
    const amrex::MultiFab &src;
  };
  static void store_old(std::span<const store_t> stores,
                        const amrex::Periodicity &period, rat64 parent_step);
  static void store_stage(std::span<const store_t> stores, stage_t stage,
                          const amrex::Periodicity &period,
                          std::source_location site);

  struct sources_t {
    const amrex::MultiFab &old;
    std::span<const amrex::MultiFab> ks;
  };
  sources_t sources(rat64 parent_step, const fill_site_t &where) const;

  struct slots_t {
    amrex::MultiFab &old;
    std::span<amrex::MultiFab> ks;
  };
  std::optional<sources_t> stored(rat64 parent_step) const;
  slots_t restore(rat64 parent_step);

private:
  bool holds(rat64 parent_step) const;
  amrex::MultiFab &next_stage_band(stage_t stage, std::source_location site);

  amrex::MultiFab old_source_;
  std::vector<amrex::MultiFab> ks_source_;
  std::optional<rat64> parent_step_;
  int stages_stored_ = 0;
};

struct rk_fill_patches_t {
  rk_fill_patches_t(const rk_band_layout_t &layout, int numvars);

  amrex::MultiFab crse;
  ghost_overlaps_t overlaps;
  amrex::BoxArray footprint;
};

const amrex::FabArrayBase::FPinfo &
rk_fill_fpinfo(const amrex::FabArrayBase &fine,
               amrex::Interpolater *interpolator, const amrex::Geometry &fgeom,
               const amrex::Geometry &cgeom);

} // namespace CarpetX

#endif
