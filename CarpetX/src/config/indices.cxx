#include "config/indices.hxx"

#include <sstream>
#include <string>

namespace CarpetX {

std::string describe(const window_t window) {
  const auto levels = window.level_bounds();
  const auto patches = window.patch_bounds();
  std::ostringstream buf;
  buf << "patches [" << patches.min << ", " << patches.max
      << "), refinement levels [" << levels.min << ", " << levels.max << ")";
  return buf.str();
}

} // namespace CarpetX
