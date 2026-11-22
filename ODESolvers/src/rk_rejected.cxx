#include "rk_methods.hxx"
#include "rk_program.hxx"
#include "rk_tableau.hxx"

#include <carpetx_rk.hxx>

#include <type_traits>
#include <utility>
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

static_assert(
    std::is_const_v<decltype(explicit_rk_t::dense_output)> &&
        std::is_const_v<decltype(explicit_rk_t::tableau)> &&
        std::is_const_v<decltype(explicit_rk_t::program)> &&
        !std::is_constructible_v<explicit_rk_t, tableau_t, program_t>,
    "rk_rejected: a finished row's dense output, tableau and program are "
    "fixed by its constructor and must not be writable afterwards, and its "
    "program is planned from its tableau and must not be supplied beside it");

constexpr program_t program_of(const char *const name) {
  for (const rk_method_t &rk : rk_methods)
    if (rk_methods_detail::names_equal(rk.name, name))
      return std::get<explicit_rk_t>(rk.scheme).program;
  return program_t{};
}

constexpr program_check_t refused_only_by(bool program_check_t::*const rule) {
  program_check_t check = every_rule_holds;
  check.*rule = false;
  return check;
}

constexpr program_t ssprk3_feeds_before_state = [] {
  program_t program = program_of("SSPRK3");
  std::swap(program.stage[1].assign[0], program.stage[1].assign[1]);
  return program;
}();

static_assert(check_program(tableau_of("SSPRK3"), ssprk3_feeds_before_state) ==
                  refused_only_by(&program_check_t::order_free),
              "rk_rejected: SSPRK3 with stage 1 feeding the weighted sum "
              "before the state reads it reaches its rows when the stage is "
              "evaluated at once and must be refused by the listed-order "
              "rule, and by no other");

constexpr program_t ssprk3_shu_osher_stage = [] {
  program_t program = program_of("SSPRK3");
  assign_t &state = program.stage[1].assign[0];
  state.coeff = {};
  state.coeff[reg_var] = 1.0 / 4;
  state.coeff[reg_old] = 3.0 / 4;
  state.coeff[reg_rhs] = 1.0 / 4;
  return program;
}();

static_assert(check_program(tableau_of("SSPRK3"), ssprk3_shu_osher_stage) ==
                  refused_only_by(&program_check_t::reads_var_at_stage_0_only),
              "rk_rejected: SSPRK3 with stage 1 in Shu-Osher form, a quarter "
              "of the state plus three quarters of the old state, reaches "
              "row 2 and must be refused by the rule that the state is read "
              "only at stage 0, and by no other");

constexpr program_t dp87_reads_released_copy = [] {
  program_t program = program_of("DP87");
  program.stage[3].frees[reg_k0 + 2] = false;
  program.stage[2].frees[reg_k0 + 2] = true;
  return program;
}();

static_assert(check_program(tableau_of("DP87"), dp87_reads_released_copy) ==
                  refused_only_by(&program_check_t::reads_live),
              "rk_rejected: DP87 with the copy of stage 2 freed one stage "
              "early, before stage 3 reads it, must be refused by the rule "
              "that a register is read only while it holds a value, and by "
              "no other");

constexpr program_t rk4_rate_reads_old = [] {
  program_t program = program_of("RK4");
  stage_t &stage = program.stage[1];
  assign_t &copy = rk_program_detail::next_assign(stage, reg_k0 + 1);
  copy.coeff[reg_old] = 1;
  copy.coeff[reg_rhs] = 1;
  stage.frees[reg_k0 + 1] = true;
  return program;
}();

static_assert(check_program(tableau_of("RK4"), rk4_rate_reads_old) ==
                  refused_only_by(&program_check_t::rates_read_rates),
              "rk_rejected: RK4 with a stage copy that adds the old state to "
              "the right-hand side, read by nothing and freed in the same "
              "stage, must be refused by the rule that a register of "
              "right-hand sides does not read a state, and by no other");

} // namespace
} // namespace ODESolvers
