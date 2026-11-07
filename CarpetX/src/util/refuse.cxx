#include "util/refuse.hxx"

#include <cctk.h>

#include <cassert>
#include <cstdarg>
#include <cstdio>
#include <vector>

namespace CarpetX {

namespace {
thread_local const char *operation_name = nullptr;
}

const char *current_operation() { return operation_name; }

operation_scope_t::operation_scope_t(const char *const name)
    : previous(operation_name) {
  operation_name = name;
}

operation_scope_t::~operation_scope_t() { operation_name = previous; }

void refuse(const std::source_location site, const char *const fmt, ...) {
  va_list ap, ap2;
  va_start(ap, fmt);
  va_copy(ap2, ap);
  const int length = std::vsnprintf(nullptr, 0, fmt, ap);
  va_end(ap);
  assert(length >= 0);
  std::vector<char> message(length + 1);
  std::vsnprintf(message.data(), message.size(), fmt, ap2);
  va_end(ap2);
  if (operation_name)
    CCTK_VError(site.line(), site.file_name(), CCTK_THORNSTRING, "%s: %s",
                operation_name, message.data());
  CCTK_VError(site.line(), site.file_name(), CCTK_THORNSTRING, "%s",
              message.data());
}

} // namespace CarpetX
