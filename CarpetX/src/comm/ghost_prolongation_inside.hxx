#ifndef CARPETX_CARPETX_COMM_GHOST_PROLONGATION_INSIDE_HXX
#define CARPETX_CARPETX_COMM_GHOST_PROLONGATION_INSIDE_HXX

#include "internal.hxx"

#include "hierarchy/model.hxx"
#include <carpetx.hxx>

namespace CarpetX {

void error_if_parent_invalid(const GHExt::PatchData::LevelData &leveldata,
                             gi_t gi, tl_t tl, const char *label);

} // namespace CarpetX

#endif
