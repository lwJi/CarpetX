#ifndef CARPETX_CARPETX_COMM_SUBCYCLING_HXX
#define CARPETX_CARPETX_COMM_SUBCYCLING_HXX

#include "internal.hxx"

#include <carpetx.hxx>

#include <cctk.h>

#include <span>

namespace CarpetX {

inline constexpr CCTK_REAL rk_substep_ratio = 0.5;

void EnsureRKBuffers(level_t level, gi_t gi);

void StoreRKOldState(level_t level, std::span<const gi_t> var_groups, tl_t tl);

void StoreRKStage(level_t level, std::span<const gi_t> var_groups,
                  std::span<const gi_t> rhs_groups, int stage);

void FillRKBoundary(level_t level, std::span<const gi_t> var_groups, tl_t tl,
                    int stage, CCTK_REAL xsi, CCTK_REAL dtc);

void RecoverRKBoundary(window_t window, const cGH *cctkGH);

void EnsureFluxGeometry(level_t level);

void AccumulateFluxes(level_t level, int stage, CCTK_REAL weight);

void Reflux(const cGH *cctkGH, window_t window);

} // namespace CarpetX

#endif
