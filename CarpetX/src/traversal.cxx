#include "traversal.hxx"
#include "driver.hxx"

#include <optional>

namespace CarpetX {

namespace {
std::optional<window_t> active_levels;
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

window_t scheduled_window(const char *const caller,
                          const std::source_location site) {
  if (!active_levels)
    refuse(site, "%s called outside a traversal: no levels are active", caller);
  return *active_levels;
}

} // namespace CarpetX
