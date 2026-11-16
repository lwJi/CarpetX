#ifndef CARPETX_CARPETX_COMM_OUTER_BOUNDARY_HXX
#define CARPETX_CARPETX_COMM_OUTER_BOUNDARY_HXX

#include "internal.hxx"

#include "comm/sync_guard.hxx"
#include "comm/timelevels.hxx"
#include <carpetx.hxx>

#include <span>

namespace CarpetX {

void ApplyOuterBoundary(window_t window, std::span<const gi_t> groups,
                        timelevels_t tls);

void apply_outer_boundary(const sync_guard_t &sync, window_t window,
                          std::span<const gi_t> groups, timelevels_t tls,
                          const char *event);

} // namespace CarpetX

#endif
