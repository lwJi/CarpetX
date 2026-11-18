#ifndef CARPETX_CARPETX_COMM_GHOST_PROLONGATION_HXX
#define CARPETX_CARPETX_COMM_GHOST_PROLONGATION_HXX

#include "internal.hxx"

#include "hierarchy/model.hxx"
#include <carpetx.hxx>

#include <span>

namespace CarpetX {

struct timelevels_t;

void ProlongateGhosts(window_t window, std::span<const gi_t> groups,
                      timelevels_t tls);

void ProlongateAligned(window_t window, std::span<const gi_t> groups,
                       timelevels_t tls);

void ProlongateAfterRestriction(window_t window);

void error_if_parent_invalid(const GHExt::PatchData::LevelData &leveldata,
                             gi_t gi, tl_t tl, const char *label);

} // namespace CarpetX

#endif
