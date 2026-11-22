#include "rk_methods.hxx"
#include "rk_step.hxx"
#include "solve.hxx"

#include <carpetx_rk.hxx>

namespace ODESolvers {
using namespace std;

namespace {

void register_explicit(const explicit_rk_t &scheme) {
  CarpetX::rk_config_t integrator;
  integrator.dense_output = scheme.dense_output;
  integrator.b.assign(scheme.tableau.b.begin(),
                      scheme.tableau.b.begin() + scheme.tableau.nstages);
  for (int gi0 = 0; gi0 < CCTK_NumGroups(); ++gi0) {
    if (CCTK_GroupTypeI(gi0) != CCTK_GF)
      continue;
    const CarpetX::gi_t gi{gi0};
    const std::optional<CarpetX::gi_t> rhs_gi = get_group_rhs(gi);
    if (rhs_gi)
      integrator.integrated.push_back(CarpetX::rk_pair_t{gi, *rhs_gi});
  }
  CarpetX::register_integrator(integrator);
}

} // namespace

extern "C" void ODESolvers_InitConstants(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTS_ODESolvers_InitConstants;
  DECLARE_CCTK_PARAMETERS;

  *do_substeps = 0;

  std::visit(overloaded{[](const frozen_t &) {},
                        [](const explicit_rk_t &scheme) {
                          register_explicit(scheme);
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

  statecomp_t var, rhs;
  std::vector<CarpetX::gi_t> var_groups, rhs_groups, dep_groups;
  int nvars = 0;
  bool do_accumulate_nvars = true;
  window.loop_serially([&](const CarpetX::level_t level) {
    for (int gi0 = 0; gi0 < CCTK_NumGroups(); ++gi0) {
      if (CCTK_GroupTypeI(gi0) != CCTK_GF)
        continue;
      const CarpetX::gi_t gi{gi0};
      const std::optional<CarpetX::gi_t> rhs_gi = get_group_rhs(gi);
      if (rhs_gi) {
        assert(*rhs_gi != gi);
        const CarpetX::group_t group{level, gi};
        const CarpetX::group_t rhs_group{level, *rhs_gi};
        assert(CarpetX::numvars(rhs_group) == CarpetX::numvars(group));
        var.entries.push_back({group, std::nullopt});
        rhs.entries.push_back({rhs_group, std::nullopt});
        if (do_accumulate_nvars) {
          nvars += CarpetX::numvars(group);
          var_groups.push_back(gi);
          rhs_groups.push_back(*rhs_gi);
          const auto &dependents = get_group_dependents(gi);
          dep_groups.insert(dep_groups.end(), dependents.begin(),
                            dependents.end());
        }
      }
    }
    do_accumulate_nvars = false;
  });
  if (verbose)
    CCTK_VINFO("  Integrating %d variables", nvars);
  if (nvars == 0)
    CCTK_VWARN(CCTK_WARN_ALERT, "Integrating %d variables", nvars);

  {
    std::sort(var_groups.begin(), var_groups.end());
    const auto last = std::unique(var_groups.begin(), var_groups.end());
    assert(last == var_groups.end());
  }

  {
    std::sort(rhs_groups.begin(), rhs_groups.end());
    const auto last = std::unique(rhs_groups.begin(), rhs_groups.end());
    assert(last == rhs_groups.end());
  }

  // Add RHS variables to dependent variables
  dep_groups.insert(dep_groups.end(), rhs_groups.begin(), rhs_groups.end());

  {
    std::sort(dep_groups.begin(), dep_groups.end());
    const auto last = std::unique(dep_groups.begin(), dep_groups.end());
    dep_groups.erase(last, dep_groups.end());
  }

  for (const CarpetX::gi_t gi : var_groups)
    assert(std::find(dep_groups.begin(), dep_groups.end(), gi) ==
           dep_groups.end());
  for (const CarpetX::gi_t gi : rhs_groups)
    assert(std::find(var_groups.begin(), var_groups.end(), gi) ==
           var_groups.end());

  interval_setup.reset();

  step(cctkGH, rk_method(method), window, var, rhs, dep_groups);
}

} // namespace ODESolvers
