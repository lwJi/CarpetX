#ifndef CARPETX_CARPETX_COMM_SYNC_GUARD_HXX
#define CARPETX_CARPETX_COMM_SYNC_GUARD_HXX

#include "internal.hxx"

#include "comm/timelevels.hxx"
#include <carpetx.hxx>

#include <source_location>
#include <span>

namespace CarpetX {

class sync_guard_t {
  explicit sync_guard_t(
      std::source_location site = std::source_location::current());
  friend void SyncGroups(window_t window, std::span<const gi_t> groups);
  friend void SyncGhostsOnly(window_t window, std::span<const gi_t> groups,
                             timelevels_t tls);
  friend void ProlongateGhosts(window_t window, std::span<const gi_t> groups,
                               timelevels_t tls);
  friend void ProlongateAligned(window_t window, std::span<const gi_t> groups,
                                timelevels_t tls);
  friend void ProlongateAfterRestriction(window_t window);
  friend void ApplyOuterBoundary(window_t window, std::span<const gi_t> groups,
                                 timelevels_t tls);

public:
  ~sync_guard_t();
  sync_guard_t(const sync_guard_t &) = delete;
  sync_guard_t &operator=(const sync_guard_t &) = delete;
};

} // namespace CarpetX

#endif
