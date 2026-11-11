#ifndef CARPETX_CARPETX_COMM_SUBCYCLING_HXX
#define CARPETX_CARPETX_COMM_SUBCYCLING_HXX

#include "internal.hxx"

#include "hierarchy/clocks.hxx"
#include "hierarchy/driver.hxx"
#include <carpetx.hxx>

#include <cctk.h>

#include <AMReX_MultiFab.H>

#include <span>
#include <string>
#include <vector>

namespace CarpetX {

inline constexpr CCTK_REAL rk_substep_ratio = CCTK_REAL(1) / time_refinement;

inline constexpr tl_t rk_tl{0};

void StoreRKOldState(level_t level, std::span<const gi_t> var_groups, tl_t tl);

void StoreRKStage(level_t level, std::span<const gi_t> var_groups,
                  std::span<const gi_t> rhs_groups, stage_t stage);

void FillRKBoundary(level_t child, std::span<const gi_t> states,
                    stage_t completed);

void RecoverRKBoundary(window_t window);

struct checkpoint_band_t {
  std::string tag;
  amrex::MultiFab *band;
};

std::vector<checkpoint_band_t> checkpoint_bands(level_t level, gi_t gi,
                                                rat64 parent_step);

std::vector<checkpoint_band_t> restore_bands(level_t level, gi_t gi,
                                             rat64 parent_step);

} // namespace CarpetX

#endif
