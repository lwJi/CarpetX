#ifndef CARPETX_CARPETX_COMM_SYNC_HXX
#define CARPETX_CARPETX_COMM_SYNC_HXX

#include "internal.hxx"

#include "comm/timelevels.hxx"

#include <carpetx.hxx>

#include <cctk.h>

#include <span>
#include <vector>

namespace CarpetX {

std::vector<gi_t> collect_restrictable_groups();

void Restrict(window_t window, std::span<const gi_t> groups);
void ProlongateAfterRestriction(window_t window);

void RestrictNoPoison(window_t window, std::span<const gi_t> groups);

void SyncGroups(window_t window, std::span<const gi_t> groups);

int SyncGroupsOnActiveLevels(const cGH *restrict cctkGH, int numgroups,
                             const int *groups, const int *directions);

void ProlongateGhosts(window_t window, std::span<const gi_t> groups,
                      timelevels_t tls);

void ProlongateAligned(window_t window, std::span<const gi_t> groups,
                       timelevels_t tls);

void SyncGhostsOnly(window_t window, std::span<const gi_t> groups,
                    timelevels_t tls);

void ApplyOuterBoundary(window_t window, std::span<const gi_t> groups,
                        timelevels_t tls);

} // namespace CarpetX

#endif
