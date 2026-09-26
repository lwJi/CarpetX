#ifndef CARPETX_ODESOLVERS_RK_METHODS_HXX
#define CARPETX_ODESOLVERS_RK_METHODS_HXX

// The single definition of each ODESolvers::method's stage count, subcycling
// support and final-update weights. Read by ODESolvers_InitConstants (which
// publishes the stage count to the driver at WRAGH), by ODESolvers_CheckMethod
// at PARAMCHECK, and by the solvers' calcrhs, which hand b_s * dt to the
// driver's flux registers (CarpetX::AccumulateFluxes) so that after a full
// step a register holds exactly the flux combination the state received.
//
// The hand-written stage-update sequences in odesolvers_solve.cxx and
// odesolvers_solve_subcycling.cxx remain the authority for the values here:
// a row's b must be the effective weight of each stage's RHS in that
// sequence's final update, y1 = y0 + dt * sum_s b_s f(Y_s). A row with a
// nonzero b is valid only with a multi-level conservation test that pins it
// (FluxWaveToyX/test/standing_subcycling*.par: SSPRK3 with two levels, RK4
// with three); a mismatch between b and the update sequence is not caught by
// any other check and shows up only as a conservation drift.

// TODO: Don't include files from other thorns; create a proper interface
#include "../../CarpetX/src/driver.hxx"

#include <cctk.h>

#include <array>
#include <cstddef>

namespace ODESolvers {

struct rk_method_t {
  const char *name;
  int nstages;        // stages run per step; published as ghext->num_rk_stages
  bool subcycling_ok; // implemented by ODESolvers_Solve_Subcycling
  bool explicit_b;    // b below is meaningful (false: IMEX rows, and the
                      // RKF78/DP87 rows whose b lives in their local tableau
                      // in odesolvers_solve.cxx)
  std::array<CCTK_REAL, CarpetX::max_num_rk_stages> b; // final-update weights
};

// One row per value of the ODESolvers::method keyword (param.ccl).
//
// "constant" keeps 4 stages so that the subcycling band machinery sizes its
// buffers as for RK4; its weights are zero because the state never changes.
// RKF78 runs the 11-stage tableau in odesolvers_solve.cxx (the two embedded
// error-estimate stages are commented out there), DP87 the 13-stage one; both
// exceed max_num_rk_stages, so their b stays in the local tableau and they
// are not available under subcycling.
inline constexpr std::array<rk_method_t, 10> rk_methods{{
    {"constant", 4, true, true, {0, 0, 0, 0}},
    {"Euler", 1, false, true, {1, 0, 0, 0}},
    {"RK2", 2, false, true, {0, 1, 0, 0}},
    {"RK3", 3, false, true, {1.0 / 6, 2.0 / 3, 1.0 / 6, 0}},
    {"SSPRK3", 3, true, true, {1.0 / 6, 1.0 / 6, 2.0 / 3, 0}},
    {"RK4", 4, true, true, {1.0 / 6, 1.0 / 3, 1.0 / 3, 1.0 / 6}},
    {"RKF78", 11, false, false, {0, 0, 0, 0}},
    {"DP87", 13, false, false, {0, 0, 0, 0}},
    {"IMEX122", 2, false, false, {0, 0, 0, 0}},
    {"Implicit Euler", 2, false, false, {0, 0, 0, 0}},
}};

namespace rk_methods_detail {

constexpr bool row_consistent(const rk_method_t &rk) {
  if (rk.nstages < 1)
    return false;
  // Everything the subcycling solver runs must fit into the band arrays
  if (rk.subcycling_ok && rk.nstages > CarpetX::max_num_rk_stages)
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

} // namespace rk_methods_detail

static_assert(rk_methods_detail::table_consistent(),
              "rk_methods: a row's b must sum to 1 (or be all zero), vanish "
              "beyond nstages, and be zero when explicit_b is false; a "
              "subcycling_ok row must fit into max_num_rk_stages");

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
