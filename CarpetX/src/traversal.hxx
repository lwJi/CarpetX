#ifndef CARPETX_CARPETX_TRAVERSAL_HXX
#define CARPETX_CARPETX_TRAVERSAL_HXX

#include "internal.hxx"

#include <carpetx.hxx>

#include <source_location>

namespace CarpetX {

class traversal_scope_t {
public:
  explicit traversal_scope_t(
      window_t window,
      std::source_location site = std::source_location::current());
  ~traversal_scope_t();
  traversal_scope_t(const traversal_scope_t &) = delete;
  traversal_scope_t &operator=(const traversal_scope_t &) = delete;
};

window_t
scheduled_window(const char *caller,
                 std::source_location site = std::source_location::current());

} // namespace CarpetX

#endif
