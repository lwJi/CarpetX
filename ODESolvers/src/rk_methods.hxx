#ifndef CARPETX_ODESOLVERS_RK_METHODS_HXX
#define CARPETX_ODESOLVERS_RK_METHODS_HXX

// The single definition of each ODESolvers::method's stage count, subcycling
// support, reflux support and final-update weights. Read by
// ODESolvers_InitConstants (which publishes the stage count to the driver at
// WRAGH), by ODESolvers_CheckMethod at PARAMCHECK, and by both solvers'
// calcrhs, which hand b_s * dt to the driver's flux registers
// (CarpetX::AccumulateFluxes) so that after a full step a register holds
// exactly the flux combination the state received.
//
// The hand-written stage-update sequences in odesolvers_solve.cxx and
// odesolvers_solve_subcycling.cxx remain the authority for the values here:
// a row's b must be the effective weight of each stage's RHS in that
// sequence's final update, y1 = y0 + dt * sum_s b_s f(Y_s). A row with a
// nonzero b is valid only with a multi-level conservation test that pins it
// (FluxWaveToyX/test/standing_subcycling*.par: SSPRK3 with two levels, RK4
// with three; standing_2levels*.par: RK4 with two levels, no subcycling); a
// mismatch between b and the update sequence is not caught by any other
// check and shows up only as a conservation drift.

#include <carpetx_rk.hxx>

#include <cctk.h>

#include <array>
#include <cstddef>
#include <initializer_list>

namespace ODESolvers {

struct rk_method_t {
  const char *name;
  int nstages;
  CarpetX::rk_dense_output_t dense_output;
  bool reflux_ok;
  bool explicit_b; // b below is meaningful (false: IMEX rows, and the
                   // RKF78/DP87 rows whose b lives in their local tableau
                   // in odesolvers_solve.cxx)
  std::array<CCTK_REAL, CarpetX::max_num_rk_stages> b; // final-update weights
};

inline constexpr std::array<rk_method_t, 10> rk_methods{{
    {"constant", 4, CarpetX::rk_dense_output_t::rk4, true, true, {0, 0, 0, 0}},
    {"Euler", 1, CarpetX::rk_dense_output_t::none, true, true, {1, 0, 0, 0}},
    {"RK2", 2, CarpetX::rk_dense_output_t::none, true, true, {0, 1, 0, 0}},
    {"RK3",
     3,
     CarpetX::rk_dense_output_t::none,
     true,
     true,
     {1.0 / 6, 2.0 / 3, 1.0 / 6, 0}},
    {"SSPRK3",
     3,
     CarpetX::rk_dense_output_t::ssprk3,
     true,
     true,
     {1.0 / 6, 1.0 / 6, 2.0 / 3, 0}},
    {"RK4",
     4,
     CarpetX::rk_dense_output_t::rk4,
     true,
     true,
     {1.0 / 6, 1.0 / 3, 1.0 / 3, 1.0 / 6}},
    {"RKF78", 11, CarpetX::rk_dense_output_t::none, true, false, {0, 0, 0, 0}},
    {"DP87", 13, CarpetX::rk_dense_output_t::none, true, false, {0, 0, 0, 0}},
    {"IMEX122",
     2,
     CarpetX::rk_dense_output_t::none,
     false,
     false,
     {0, 0, 0, 0}},
    {"Implicit Euler",
     2,
     CarpetX::rk_dense_output_t::none,
     false,
     false,
     {0, 0, 0, 0}},
}};

namespace rk_methods_detail {

constexpr bool row_consistent(const rk_method_t &rk) {
  if (rk.nstages < 1)
    return false;
  const bool has_dense_output =
      rk.dense_output != CarpetX::rk_dense_output_t::none;
  if (has_dense_output &&
      rk.nstages != CarpetX::rk_dense_output_stages(rk.dense_output))
    return false;
  if ((rk.explicit_b || has_dense_output) && !rk.reflux_ok)
    return false;
  CCTK_REAL sum = 0;
  bool all_zero = true;
  for (std::size_t s = 0; s < rk.b.size(); ++s) {
    sum += rk.b[s];
    all_zero = all_zero && rk.b[s] == 0;
    // No weight beyond the stages that are run
    if (int(s) >= rk.nstages && rk.b[s] != 0)
      return false;
  }
  if (!rk.explicit_b)
    return all_zero;
  // Consistency: sum b = 1, except for the method that never updates
  const CCTK_REAL err = sum - 1 < 0 ? 1 - sum : sum - 1;
  return all_zero || err <= 1.0e-15;
}

constexpr bool table_consistent() {
  for (const auto &rk : rk_methods)
    if (!row_consistent(rk))
      return false;
  return true;
}

constexpr char lower(const char c) {
  return c >= 'A' && c <= 'Z' ? char(c - 'A' + 'a') : c;
}

constexpr bool names_equal(const char *a, const char *b) {
  while (*a && lower(*a) == lower(*b)) {
    ++a;
    ++b;
  }
  return lower(*a) == lower(*b);
}

constexpr bool names_distinct() {
  for (std::size_t i = 0; i < rk_methods.size(); ++i)
    for (std::size_t j = 0; j < i; ++j)
      if (names_equal(rk_methods[i].name, rk_methods[j].name))
        return false;
  return true;
}

constexpr bool explicit_b_rows_fit() {
  for (const auto &rk : rk_methods)
    if (rk.explicit_b && rk.nstages > CarpetX::max_num_rk_stages)
      return false;
  return true;
}

constexpr bool
dense_output_rows_are(const std::initializer_list<const char *> methods) {
  std::size_t listed_rows = 0;
  for (const auto &rk : rk_methods) {
    bool listed = false;
    for (const char *const method : methods)
      listed = listed || names_equal(rk.name, method);
    if (listed != (rk.dense_output != CarpetX::rk_dense_output_t::none))
      return false;
    listed_rows += listed;
  }
  return listed_rows == methods.size();
}

} // namespace rk_methods_detail

static_assert(rk_methods_detail::table_consistent(),
              "rk_methods: a row's b must sum to 1 (or be all zero), vanish "
              "beyond nstages, and be zero when explicit_b is false; a row "
              "with a dense output must have that method's stage count "
              "(CarpetX::rk_dense_output_stages); an explicit_b row or a row "
              "with a dense output must be reflux_ok");
static_assert(rk_methods_detail::names_distinct(),
              "rk_methods: rk_method finds a row by its name, compared "
              "case-insensitively, so no two rows may share a name");
static_assert(rk_methods_detail::explicit_b_rows_fit(),
              "rk_methods: an explicit_b row must fit its b into "
              "max_num_rk_stages; a longer method keeps b in its own tableau");

// Look up the row for the given ODESolvers::method (case-insensitive, like
// CCTK_EQUALS). Every keyword value has a row; a missing one is a hard error.
inline const rk_method_t &rk_method(const char *const method) {
  for (const auto &rk : rk_methods)
    if (CCTK_Equals(method, rk.name))
      return rk;
  CCTK_VERROR("ODESolvers::method = \"%s\" has no row in the RK method table "
              "(ODESolvers/src/rk_methods.hxx)",
              method);
}

} // namespace ODESolvers

#endif // #ifndef CARPETX_ODESOLVERS_RK_METHODS_HXX
