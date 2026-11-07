#ifndef CARPETX_CARPETX_UTIL_REFUSE_HXX
#define CARPETX_CARPETX_UTIL_REFUSE_HXX

#include "internal.hxx"

#include <cctk.h>

#include <source_location>

namespace CarpetX {

[[noreturn]] void refuse(std::source_location site, const char *fmt, ...)
    CCTK_ATTRIBUTE_FORMAT(printf, 2, 3);

class operation_scope_t {
public:
  explicit operation_scope_t(const char *name);
  ~operation_scope_t();
  operation_scope_t(const operation_scope_t &) = delete;
  operation_scope_t &operator=(const operation_scope_t &) = delete;

private:
  const char *const previous;
};

const char *current_operation();

} // namespace CarpetX

#endif
