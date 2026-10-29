#ifndef CARPETX_CARPETX_UTIL_VAR_TYPE_HXX
#define CARPETX_CARPETX_UTIL_VAR_TYPE_HXX

#include "internal.hxx"

#include <cctk.h>

#include <cstddef>
#include <optional>
#include <utility>

namespace CarpetX {

enum class var_type { real, integer, complex };

template <typename T> struct var_type_tag {
  using type = T;
};

template <typename F> decltype(auto) visit(const var_type type, F &&f) {
  switch (type) {
  case var_type::real:
    return std::forward<F>(f)(var_type_tag<CCTK_REAL>{});
  case var_type::integer:
    return std::forward<F>(f)(var_type_tag<CCTK_INT>{});
  case var_type::complex:
    return std::forward<F>(f)(var_type_tag<CCTK_COMPLEX>{});
  }
  CCTK_VERROR("CarpetX: unknown var_type value %d", int(type));
}

std::optional<var_type> var_type_from_cctk(int cctk_vartype);
int cctk_code(var_type type);
std::size_t size_of(var_type type);

} // namespace CarpetX

#endif
