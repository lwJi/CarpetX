#include "rk_methods.hxx"
#include "rk_tableau.hxx"

#include <carpetx_rk.hxx>

#include <type_traits>
#include <variant>

namespace ODESolvers {
namespace {

constexpr tableau_t tableau_of(const char *const name) {
  for (const rk_method_t &rk : rk_methods)
    if (rk_methods_detail::names_equal(rk.name, name))
      return std::get<explicit_rk_t>(rk.scheme).tableau;
  return tableau_t{};
}

constexpr tableau_t rk3_stages_ssprk3_weights = [] {
  tableau_t tableau = tableau_of("RK3");
  tableau.b = tableau_of("SSPRK3").b;
  return tableau;
}();

static_assert(is_explicit(rk3_stages_ssprk3_weights) &&
                  rows_sum_to_c(rk3_stages_ssprk3_weights) &&
                  weights_sum_to_one(rk3_stages_ssprk3_weights) &&
                  !has_declared_order(rk3_stages_ssprk3_weights),
              "rk_rejected: RK3's stages with SSPRK3's weights pass the "
              "row-sum and weight-sum rules and must be refused by the "
              "order rule");

constexpr tableau_t rk4_one_coefficient_off = [] {
  tableau_t tableau = tableau_of("RK4");
  tableau.a[2][1] = 1.0 / 3;
  return tableau;
}();

static_assert(is_explicit(rk4_one_coefficient_off) &&
                  !rows_sum_to_c(rk4_one_coefficient_off),
              "rk_rejected: RK4 with one mistyped a must be refused by the "
              "row-sum rule");

static_assert(
    !std::is_constructible_v<explicit_rk_t, CarpetX::rk_dense_output_t,
                             tableau_t>,
    "rk_rejected: a row is a tableau or a CarpetX dense-output method; a "
    "dense output written beside a tableau of the row's own must not compile");

static_assert(std::is_const_v<decltype(explicit_rk_t::dense_output)> &&
                  std::is_const_v<decltype(explicit_rk_t::tableau)>,
              "rk_rejected: a finished row's dense output and tableau are "
              "fixed by its constructor and must not be writable afterwards");

} // namespace
} // namespace ODESolvers
