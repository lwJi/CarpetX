#ifndef CARPETX_CARPETX_UTIL_VAR_TYPE_HXX
#define CARPETX_CARPETX_UTIL_VAR_TYPE_HXX

#include "internal.hxx"

#include <cstddef>
#include <optional>

namespace CarpetX {

enum class var_type { real, integer, complex };

#pragma GCC diagnostic error "-Wswitch"

std::optional<var_type> var_type_from_cctk(int cctk_vartype);
int cctk_code(var_type type);
std::size_t size_of(var_type type);

} // namespace CarpetX

#endif
