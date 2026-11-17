#ifndef CARPETX_CARPETX_COMM_SYNC_HXX
#define CARPETX_CARPETX_COMM_SYNC_HXX

#include "internal.hxx"

#include "comm/timelevels.hxx"

#include <carpetx.hxx>

#include <cctk.h>

#include <span>
#include <vector>

namespace CarpetX {

class requested_sync_t {
  requested_sync_t() = default;
  friend void SyncRequested(window_t window, std::span<const gi_t> groups);

public:
  requested_sync_t(const requested_sync_t &) = delete;
  requested_sync_t &operator=(const requested_sync_t &) = delete;
};

class public_ghost_only_t {
  public_ghost_only_t() = default;
  friend void sync_ghosts_only(window_t window, std::span<const gi_t> groups);
  friend void sync_ghosts_only(window_t window, std::span<const gi_t> groups,
                               tl_t tl);

public:
  public_ghost_only_t(const public_ghost_only_t &) = delete;
  public_ghost_only_t &operator=(const public_ghost_only_t &) = delete;
};

void SyncGroups(window_t window, std::span<const gi_t> groups,
                const requested_sync_t &requested);

void SyncGhostsOnly(window_t window, std::span<const gi_t> groups,
                    timelevels_t tls, const public_ghost_only_t &asked);

} // namespace CarpetX

#endif
