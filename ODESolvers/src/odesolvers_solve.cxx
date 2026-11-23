#include "integrated_groups.hxx"
#include "rk_methods.hxx"
#include "rk_step.hxx"
#include "state.hxx"

#include <carpetx_rk.hxx>

#include <cctk.h>
#include <cctk_Arguments.h>
#include <cctk_Parameters.h>

#include <optional>
#include <variant>
#include <vector>

namespace ODESolvers {

extern "C" void ODESolvers_InitConstants(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTS_ODESolvers_InitConstants;

  *do_substeps = 0;
}

extern "C" void ODESolvers_RegisterIntegrator(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTS_ODESolvers_RegisterIntegrator;
  DECLARE_CCTK_PARAMETERS;

  std::visit(
      overloaded{
          [](const frozen_t &) {},
          [](const explicit_rk_t &scheme) {
            const std::vector<CarpetX::rk_pair_t> &pairs =
                integrated_groups().pairs;
            CarpetX::register_integrator(
                scheme.dense_output == CarpetX::rk_dense_output_t::none
                    ? CarpetX::rk_config_t::with_weights(
                          std::vector<CCTK_REAL>(scheme.tableau.b.begin(),
                                                 scheme.tableau.b.begin() +
                                                     scheme.tableau.nstages),
                          pairs)
                    : CarpetX::rk_config_t::with_dense_output(
                          scheme.dense_output, pairs));
          }},
      rk_method(method).scheme);
}

extern "C" void ODESolvers_Solve(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTS_ODESolvers_Solve;
  DECLARE_CCTK_PARAMETERS;

  static bool did_output = false;
  if (verbose || !did_output)
    CCTK_VINFO("ODE integrator is %s", method);
  did_output = true;

  static CarpetX::Timer timer("ODESolvers::Solve");
  CarpetX::Interval interval(timer);

  static CarpetX::Timer timer_setup("ODESolvers::Solve::setup");
  std::optional<CarpetX::Interval> interval_setup(timer_setup);

  const CarpetX::window_t window = CarpetX::active_window();
  const integrated_groups_t &integrated = integrated_groups();

  grid_state_t var, rhs;
  window.loop_serially([&](const CarpetX::level_t level) {
    for (const CarpetX::rk_pair_t &pair : integrated.pairs) {
      var.groups.push_back(CarpetX::group_t{level, pair.state});
      rhs.groups.push_back(CarpetX::group_t{level, pair.rhs});
    }
  });
  if (verbose)
    CCTK_VINFO("  Integrating %d variables", integrated.nvars);
  if (integrated.nvars == 0)
    CCTK_VWARN(CCTK_WARN_ALERT, "Integrating %d variables", integrated.nvars);

  interval_setup.reset();

  step(cctkGH, rk_method(method), window, var, rhs,
       integrated.invalidated_by_update);
}

} // namespace ODESolvers
