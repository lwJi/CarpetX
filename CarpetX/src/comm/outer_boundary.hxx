#ifndef CARPETX_CARPETX_COMM_OUTER_BOUNDARY_HXX
#define CARPETX_CARPETX_COMM_OUTER_BOUNDARY_HXX

#include "internal.hxx"

#include <carpetx.hxx>

#include <span>

namespace CarpetX {

struct timelevels_t;

void ApplyOuterBoundary(window_t window, std::span<const gi_t> groups,
                        timelevels_t tls);

} // namespace CarpetX

#endif
