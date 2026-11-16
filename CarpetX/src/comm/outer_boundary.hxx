#ifndef CARPETX_CARPETX_COMM_OUTER_BOUNDARY_HXX
#define CARPETX_CARPETX_COMM_OUTER_BOUNDARY_HXX

#include "internal.hxx"

#include "comm/timelevels.hxx"
#include <carpetx.hxx>

#include <source_location>
#include <span>

namespace CarpetX {

void ApplyOuterBoundary(window_t window, std::span<const gi_t> groups,
                        timelevels_t tls);

void error_if_sync_nested(
    std::source_location site = std::source_location::current());

void fill_outer_boundary(window_t window, std::span<const gi_t> groups,
                         timelevels_t tls, const char *event);

} // namespace CarpetX

#endif
