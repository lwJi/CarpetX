#include "rk_methods.hxx"
#include "rk_step.hxx"
#include "solve.hxx"

#include <carpetx.hxx>
#include <carpetx_rk.hxx>

#include <vector>

namespace ODESolvers {

namespace {

struct solve_setup_t {
  statecomp_t var, rhs;
  std::vector<CarpetX::gi_t> var_groups, rhs_groups, dep_groups;
  int nvars = 0;
};

solve_setup_t collect_solve_setup(const CarpetX::window_t window) {
  solve_setup_t s;
  s.var.timelevel = CarpetX::tl_t{0};
  s.rhs.timelevel = CarpetX::tl_t{0};
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
        s.var.entries.push_back({group, std::nullopt});
        s.rhs.entries.push_back({rhs_group, std::nullopt});

        if (do_accumulate_nvars) {
          s.nvars += CarpetX::numvars(group);
          s.var_groups.push_back(gi);
          s.rhs_groups.push_back(*rhs_gi);
          const auto &dependents = get_group_dependents(gi);
          s.dep_groups.insert(s.dep_groups.end(), dependents.begin(),
                              dependents.end());
        }
      }
    }
    do_accumulate_nvars = false;
  });

  {
    std::sort(s.var_groups.begin(), s.var_groups.end());
    const auto last = std::unique(s.var_groups.begin(), s.var_groups.end());
    assert(last == s.var_groups.end());
  }

  {
    std::sort(s.rhs_groups.begin(), s.rhs_groups.end());
    const auto last = std::unique(s.rhs_groups.begin(), s.rhs_groups.end());
    assert(last == s.rhs_groups.end());
  }

  // Add RHS variables to dependent variables
  s.dep_groups.insert(s.dep_groups.end(), s.rhs_groups.begin(),
                      s.rhs_groups.end());

  {
    std::sort(s.dep_groups.begin(), s.dep_groups.end());
    const auto last = std::unique(s.dep_groups.begin(), s.dep_groups.end());
    s.dep_groups.erase(last, s.dep_groups.end());
  }

  for (const CarpetX::gi_t gi : s.var_groups)
    assert(std::find(s.dep_groups.begin(), s.dep_groups.end(), gi) ==
           s.dep_groups.end());
  for (const CarpetX::gi_t gi : s.rhs_groups)
    assert(std::find(s.var_groups.begin(), s.var_groups.end(), gi) ==
           s.var_groups.end());

  return s;
}

} // namespace

extern "C" void ODESolvers_Solve_Subcycling(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTS_ODESolvers_Solve_Subcycling;
  DECLARE_CCTK_PARAMETERS;

  static bool did_output = false;
  if (verbose || !did_output)
    CCTK_VINFO("Integrator is %s", method);
  did_output = true;

  static CarpetX::Timer timer("ODESolvers::Solve");
  CarpetX::Interval interval(timer);

  static CarpetX::Timer timer_setup("ODESolvers::Solve::setup");
  std::optional<CarpetX::Interval> interval_setup(timer_setup);

  const CarpetX::window_t window = CarpetX::active_window();

  auto setup = collect_solve_setup(window);
  auto &var = setup.var;
  auto &rhs = setup.rhs;
  auto &dep_groups = setup.dep_groups;
  const int nvars = setup.nvars;
  if (verbose)
    CCTK_VINFO("  Integrating %d variables", nvars);
  if (nvars == 0)
    CCTK_VWARN(CCTK_WARN_ALERT, "Integrating %d variables", nvars);

  interval_setup.reset();

  step(cctkGH, rk_method(method), window, var, rhs, dep_groups);
}

extern "C" void ODESolvers_CheckTimelevels(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTS_ODESolvers_CheckTimelevels;
  DECLARE_CCTK_PARAMETERS;

  for (int gi = 0; gi < CCTK_NumGroups(); ++gi) {
    if (!get_group_rhs(CarpetX::gi_t{gi}))
      continue;
    const int ntls = CCTK_ActiveTimeLevelsGI(cctkGH, gi);
    if (ntls >= 2)
      CCTK_VERROR("ODESolvers subcycling requires evolved groups to have a "
                  "single timelevel, but group \"%s\" has %d active "
                  "timelevels. Subcycling does not support timelevels >= 2 "
                  "for evolution variables.",
                  CCTK_FullGroupName(gi), ntls);
  }
}

extern "C" void ODESolvers_CheckMethod(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTS_ODESolvers_CheckMethod;
  DECLARE_CCTK_PARAMETERS;

  const cParamData *const method_param =
      CCTK_ParameterData("method", "ODESolvers");
  assert(method_param);
  for (const t_range *keyword = method_param->range; keyword;
       keyword = keyword->next) {
    if (!CCTK_Equals(keyword->origin, "ODESolvers"))
      continue;
    const bool has_row = std::any_of(
        rk_methods.begin(), rk_methods.end(), [&](const rk_method_t &row) {
          return CCTK_Equals(keyword->range, row.name);
        });
    if (!has_row)
      CCTK_VERROR("ODESolvers::method keyword \"%s\" (ODESolvers/param.ccl) "
                  "has no row in the RK method table "
                  "(ODESolvers/src/rk_methods.hxx)",
                  keyword->range);
  }
  for (const rk_method_t &row : rk_methods) {
    bool has_keyword = false;
    for (const t_range *keyword = method_param->range; keyword;
         keyword = keyword->next)
      if (CCTK_Equals(keyword->origin, "ODESolvers") &&
          CCTK_Equals(keyword->range, row.name))
        has_keyword = true;
    if (!has_keyword)
      CCTK_VERROR("The RK method table (ODESolvers/src/rk_methods.hxx) has a "
                  "row \"%s\" that is not a keyword of ODESolvers::method "
                  "(ODESolvers/param.ccl)",
                  row.name);
  }
}

} // namespace ODESolvers
