#include "comm/sync_guard.hxx"

#include "util/refuse.hxx"

#include <source_location>

namespace CarpetX {

namespace {
bool sync_active = false;
} // namespace

sync_guard_t::sync_guard_t(const std::source_location site) {
  if (sync_active)
    refuse(site, "a sync is already open: a sync cannot be started inside "
                 "one. A routine that a sync calls, such as a thorn's "
                 "MultiPatch_Interpolate, must not sync grid functions");
  sync_active = true;
}

sync_guard_t::~sync_guard_t() { sync_active = false; }

} // namespace CarpetX
