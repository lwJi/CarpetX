#include "traversal/traversal.hxx"
#include "util/refuse.hxx"

#include <cassert>
#include <optional>

namespace CarpetX {

namespace {
std::optional<window_t> active_levels;
const char *tail_bin = nullptr;
const cFunctionData *last_routine = nullptr;
int routine_depth = 0;
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

routine_scope_t::routine_scope_t(const cFunctionData *const routine)
    : routine(routine), depth(++routine_depth) {
  last_routine = routine;
}

routine_scope_t::~routine_scope_t() {
  assert(routine_depth == depth);
  --routine_depth;
  last_routine = routine;
}

bool routine_scope_t::outermost() const { return depth == 1; }

window_t scheduled_window(const char *const caller,
                          const std::source_location site) {
  if (!active_levels)
    refuse(site, "%s called outside a traversal: no levels are active", caller);
  return *active_levels;
}

const char *scheduled_tail_bin() { return tail_bin; }

const cFunctionData *scheduled_routine() { return last_routine; }

} // namespace CarpetX
