#ifndef CARPETX_CARPETX_COMM_GHOST_PROLONGATION_HXX
#define CARPETX_CARPETX_COMM_GHOST_PROLONGATION_HXX

#include "internal.hxx"

#include <carpetx.hxx>

#include <span>

namespace CarpetX {

struct timelevels_t;

void ProlongateGhosts(window_t window, std::span<const gi_t> groups,
                      timelevels_t tls);

void ProlongateAligned(window_t window, std::span<const gi_t> groups,
                       timelevels_t tls);

void ProlongateAfterRestriction(window_t window);

} // namespace CarpetX

#endif
