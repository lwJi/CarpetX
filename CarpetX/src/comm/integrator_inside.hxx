#ifndef CARPETX_CARPETX_COMM_INTEGRATOR_INSIDE_HXX
#define CARPETX_CARPETX_COMM_INTEGRATOR_INSIDE_HXX

#include "internal.hxx"

#include "hierarchy/model.hxx"
#include <carpetx.hxx>

namespace CarpetX {

bool group_is_integrated(gi_t gi);
stage_range registered_stages();
rk_dense_output_t registered_dense_output();

bool integrator_owns_coarse_fine_ghosts(
    const GHExt::PatchData::LevelData &leveldata, gi_t gi);

} // namespace CarpetX

#endif
