#ifndef CARPETX_CARPETX_COMM_RK_GHOSTS_INSIDE_HXX
#define CARPETX_CARPETX_COMM_RK_GHOSTS_INSIDE_HXX

#include "internal.hxx"

#include "config/stages.hxx"

#include <cctk.h>

namespace CarpetX {

inline constexpr CCTK_REAL rk_substep_ratio = CCTK_REAL(1) / time_refinement;

} // namespace CarpetX

#endif
