#ifndef CARPETX_CARPETX_COMM_REFLUX_HXX
#define CARPETX_CARPETX_COMM_REFLUX_HXX

#include "internal.hxx"

#include "config/stages.hxx"
#include <carpetx.hxx>

#include <cctk.h>

namespace CarpetX {

bool group_has_flux_register(gi_t gi);
void warn_inert_flux_tags();

void EnsureFluxGeometry(level_t level);

void AccumulateFluxes(level_t level, stage_t stage, CCTK_REAL weight);

void Reflux(window_t window);

} // namespace CarpetX

#endif
