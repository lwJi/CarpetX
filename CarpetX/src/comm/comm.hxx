#ifndef CARPETX_CARPETX_COMM_COMM_HXX
#define CARPETX_CARPETX_COMM_COMM_HXX

#include "internal.hxx"

#include "hierarchy/driver.hxx"

#include <carpetx.hxx>

#include <cctk.h>

#include <optional>
#include <source_location>
#include <span>
#include <vector>

namespace CarpetX {

std::vector<gi_t> collect_restrictable_groups();

struct timelevels_t {
  static constexpr timelevels_t one(const tl_t tl) { return timelevels_t{tl}; }
  static constexpr timelevels_t all_but_oldest() {
    return timelevels_t{std::nullopt};
  }
  tl_range
  range(const GHExt::GroupRecord &record,
        std::source_location site = std::source_location::current()) const;

private:
  constexpr explicit timelevels_t(const std::optional<tl_t> tl) : tl_(tl) {}
  std::optional<tl_t> tl_;
};

void Restrict(const cGH *cctkGH, window_t window, std::span<const gi_t> groups);
void ProlongateAfterRestriction(window_t window, const cGH *cctkGH);

void RestrictNoPoison(const cGH *cctkGH, window_t window,
                      std::span<const gi_t> groups);

int SyncGroupsByDirI(window_t window, const cGH *restrict cctkGH,
                     std::span<const gi_t> groups);

int SyncGroupsByDirISubcycling(window_t window, const cGH *restrict cctkGH,
                               std::span<const gi_t> groups);

int SyncGroups(window_t window, const cGH *restrict cctkGH,
               std::span<const gi_t> groups);

int SyncGroupsOnActiveLevels(const cGH *restrict cctkGH, int numgroups,
                             const int *groups, const int *directions);

int SyncGroupsByDirIProlongateOnly(window_t window, const cGH *restrict cctkGH,
                                   std::span<const gi_t> groups,
                                   timelevels_t tls);

int SyncGroupsByDirIProlongateOnlyAligned(window_t window,
                                          const cGH *restrict cctkGH,
                                          std::span<const gi_t> groups,
                                          timelevels_t tls);

int SyncGroupsByDirIGhostOnly(window_t window, const cGH *restrict cctkGH,
                              std::span<const gi_t> groups, timelevels_t tls);

} // namespace CarpetX

#endif
