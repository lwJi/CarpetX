#include "util/var_type.hxx"

#include <cctk.h>

#include <cassert>
#include <cstddef>
#include <optional>

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
  switch (type) {
  case var_type::real:
    return CCTK_VARIABLE_REAL;
  case var_type::integer:
    return CCTK_VARIABLE_INT;
  case var_type::complex:
    return CCTK_VARIABLE_COMPLEX;
  }
  CCTK_VERROR("CarpetX: unknown var_type value %d", int(type));
}

std::size_t size_of(const var_type type) {
  const int size = CCTK_VarTypeSize(cctk_code(type));
  assert(size > 0);
  return size;
}

} // namespace CarpetX
