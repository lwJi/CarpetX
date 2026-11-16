#ifndef CARPETX_CARPETX_COMM_RK_GHOSTS_HXX
#define CARPETX_CARPETX_COMM_RK_GHOSTS_HXX

#include "internal.hxx"

#include "config/stages.hxx"
#include <carpetx.hxx>

#include <cctk.h>

#include <span>
#include <string>

namespace CarpetX {

inline constexpr CCTK_REAL rk_substep_ratio = CCTK_REAL(1) / time_refinement;

inline constexpr tl_t rk_tl{0};

void StoreRKOldState(level_t level, std::span<const gi_t> var_groups, tl_t tl);

void StoreRKStage(level_t level, std::span<const gi_t> var_groups,
                  std::span<const gi_t> rhs_groups, stage_t stage);

void FillRKBoundary(level_t child, std::span<const gi_t> states,
                    stage_t completed, const std::string &event);

void RecoverRKBoundary(window_t window);

} // namespace CarpetX

#endif
