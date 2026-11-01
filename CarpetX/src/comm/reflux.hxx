#ifndef CARPETX_CARPETX_COMM_REFLUX_HXX
#define CARPETX_CARPETX_COMM_REFLUX_HXX

#include "internal.hxx"

#include <carpetx.hxx>

#include <cctk.h>

namespace CarpetX {

void EnsureFluxGeometry(level_t level);

void AccumulateFluxes(level_t level, int stage, CCTK_REAL weight);

void Reflux(const cGH *cctkGH, window_t window);

} // namespace CarpetX

#endif
