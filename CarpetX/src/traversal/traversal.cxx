#include "traversal/traversal.hxx"
#include "hierarchy/driver.hxx"

#include <cassert>
#include <optional>

namespace CarpetX {

namespace {
std::optional<window_t> active_levels;
const char *tail_bin = nullptr;
bool recovery_traversal = false;
const cFunctionData *last_routine = nullptr;
} // namespace

traversal_scope_t::traversal_scope_t(const window_t window,
                                     const std::source_location site) {
  if (active_levels) {
    const auto [min_level, max_level] = active_levels->level_bounds();
    refuse(site,
           "a traversal is already open on levels [%d, %d): a window scope "
           "cannot be opened inside one",
           min_level, max_level);
  }
  active_levels = window;
}

traversal_scope_t::~traversal_scope_t() { active_levels.reset(); }

tail_bin_scope_t::tail_bin_scope_t(const char *const bin,
                                   const std::source_location site) {
  assert(bin);
  if (tail_bin)
    refuse(site,
           "a tail-bin traversal of %s is already open: a tail-bin scope "
           "cannot be opened inside one",
           tail_bin);
  tail_bin = bin;
}

tail_bin_scope_t::~tail_bin_scope_t() { tail_bin = nullptr; }

recovery_scope_t::recovery_scope_t(const std::source_location site) {
  if (recovery_traversal)
    refuse(site, "a recovery traversal is already open: a recovery scope "
                 "cannot be opened inside one");
  recovery_traversal = true;
}

recovery_scope_t::~recovery_scope_t() { recovery_traversal = false; }

routine_scope_t::routine_scope_t(const cFunctionData *const routine)
    : routine(routine) {
  last_routine = routine;
}

routine_scope_t::~routine_scope_t() { last_routine = routine; }

window_t scheduled_window(const char *const caller,
                          const std::source_location site) {
  if (!active_levels)
    refuse(site, "%s called outside a traversal: no levels are active", caller);
  return *active_levels;
}

const char *scheduled_tail_bin() { return tail_bin; }

bool in_recovery_traversal() { return recovery_traversal; }

const cFunctionData *scheduled_routine() { return last_routine; }

} // namespace CarpetX
