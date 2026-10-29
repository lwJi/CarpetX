#include "util/var_type.hxx"

#include <cctk.h>

#include <cstddef>
#include <optional>
#include <type_traits>

namespace CarpetX {

std::optional<var_type> var_type_from_cctk(const int cctk_vartype) {
  switch (cctk_vartype) {
  case CCTK_VARIABLE_REAL:
    return var_type::real;
  case CCTK_VARIABLE_INT:
    return var_type::integer;
  case CCTK_VARIABLE_COMPLEX:
    return var_type::complex;
  default:
    return std::nullopt;
  }
}

int cctk_code(const var_type type) {
  return visit(type, [](const auto tag) {
    using T = typename decltype(tag)::type;
    if constexpr (std::is_same_v<T, CCTK_REAL>)
      return CCTK_VARIABLE_REAL;
    else if constexpr (std::is_same_v<T, CCTK_INT>)
      return CCTK_VARIABLE_INT;
    else
      return CCTK_VARIABLE_COMPLEX;
  });
}

std::size_t size_of(const var_type type) {
  return visit(type, [](const auto tag) {
    return sizeof(typename decltype(tag)::type);
  });
}

} // namespace CarpetX
