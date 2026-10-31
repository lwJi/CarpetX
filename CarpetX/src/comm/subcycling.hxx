#ifndef CARPETX_CARPETX_COMM_SUBCYCLING_HXX
#define CARPETX_CARPETX_COMM_SUBCYCLING_HXX

#include "internal.hxx"

#include <carpetx.hxx>

#include <cctk.h>

#include <AMReX_MultiFab.H>

#include <span>
#include <string>
#include <vector>

namespace CarpetX {

inline constexpr CCTK_REAL rk_substep_ratio = 0.5;

void EnsureRKBuffers(level_t level, gi_t gi);

void StoreRKOldState(level_t level, std::span<const gi_t> var_groups, tl_t tl);

void StoreRKStage(level_t level, std::span<const gi_t> var_groups,
                  std::span<const gi_t> rhs_groups, int stage);

void FillRKBoundary(level_t level, std::span<const gi_t> var_groups, tl_t tl,
                    int stage, CCTK_REAL xsi, CCTK_REAL dtc);

void RecoverRKBoundary(window_t window, const cGH *cctkGH);

struct checkpoint_band_t {
  std::string tag;
  amrex::MultiFab *band;
};

std::vector<checkpoint_band_t> checkpoint_bands(level_t level, gi_t gi);

} // namespace CarpetX

#endif
