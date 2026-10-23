#include "rk_methods.hxx"
#include "solve.hxx"

#include <carpetx.hxx>
#include <carpetx_rk.hxx>

#include <array>
#include <cstddef>
#include <vector>

namespace ODESolvers {

namespace {

struct solve_setup_t {
  statecomp_t var, rhs;
  std::vector<int> var_groups, rhs_groups, dep_groups;
  int nvars = 0;
};

solve_setup_t collect_solve_setup(const CarpetX::window_t window) {
  solve_setup_t s;
  s.var.timelevel = 0;
  s.rhs.timelevel = 0;
  bool do_accumulate_nvars = true;
  window.loop_serially([&](const CarpetX::level_t level) {
    for (int gi = 0; gi < CCTK_NumGroups(); ++gi) {
      if (CCTK_GroupTypeI(gi) != CCTK_GF)
        continue;
      const int rhs_gi = get_group_rhs(gi);
      if (rhs_gi >= 0) {
        assert(rhs_gi != gi);
        const CarpetX::group_t group{level, gi};
        const CarpetX::group_t rhs_group{level, rhs_gi};
        assert(CarpetX::numvars(rhs_group) == CarpetX::numvars(group));
        s.var.entries.push_back({group, std::nullopt});
        s.rhs.entries.push_back({rhs_group, std::nullopt});

        if (do_accumulate_nvars) {
          s.nvars += CarpetX::numvars(group);
          s.var_groups.push_back(gi);
          s.rhs_groups.push_back(rhs_gi);
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

  for (const int gi : s.var_groups)
    assert(std::find(s.dep_groups.begin(), s.dep_groups.end(), gi) ==
           s.dep_groups.end());
  for (const int gi : s.rhs_groups)
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

  const CCTK_REAL dt = CCTK_DELTA_TIME;

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

  const CCTK_REAL old_time = cctkGH->cctk_time - dt;
  CCTK_REAL t = old_time;

  static CarpetX::Timer timer_lincomb("ODESolvers::Solve::lincomb");
  static CarpetX::Timer timer_rhs("ODESolvers::Solve::rhs");
  static CarpetX::Timer timer_poststep("ODESolvers::Solve::poststep");

  const rk_method_t &rk = rk_method(method);
  assert(rk.dense_output != CarpetX::rk_dense_output_t::none);

  std::optional<CarpetX::rk_step_t> step;

  const auto open_step = [&]() {
    step.emplace(CarpetX::open_rk_step(window));
  };
  const auto calcrhs = [&](const int n) {
    CarpetX::Interval interval_rhs(timer_rhs);
    if (verbose)
      CCTK_VINFO("Calculating RHS #%d at t=%g", n, double(t));
    CarpetX::run_schedule_group_at(cctkGH, "ODESolvers_RHS", t,
                                   cctkGH->cctk_delta_time);
    rhs.require_valid(CarpetX::make_valid_int(),
                      "ODESolvers after calling ODESolvers_RHS");
    step.value().rhs_computed(cctkGH->cctk_delta_time);
  };
  const auto calcupdate = [&](const int n, const CCTK_REAL c,
                              const CCTK_REAL a0, const auto &as,
                              const auto &vars) {
    CarpetX::Interval interval_lincomb(timer_lincomb);
    if (verbose)
      CCTK_VINFO("Calculated new state #%d at t=%g", n, double(t));
    statecomp_t::lincomb(var, a0, as, vars, CarpetX::make_valid_int());
    var.require_valid(CarpetX::make_valid_int(),
                      "ODESolvers after defining new state vector");
    CarpetX::invalidate(window, dep_groups, 0,
                        "ODESolvers updated the state vector");
    t = old_time + c;
  };
  const auto calcpoststep = [&]() {
    CarpetX::Interval interval_poststep(timer_poststep);
    CarpetX::run_schedule_group_at(cctkGH, "ODESolvers_PostStep", t,
                                   cctkGH->cctk_delta_time);
  };
  const auto calcys_rmbnd = [&]() {
    if (verbose)
      CCTK_VINFO("Fill refinement boundary ghost zones using Ys at t=%g",
                 double(t));
    step.value().state_updated();
  };

  static_assert(
      rk_methods_detail::dense_output_rows_are({"constant", "RK4", "SSPRK3"}),
      "rk_methods: the rows with a dense output must be the methods "
      "ODESolvers_Solve_Subcycling has a branch for");

  if (CCTK_EQUALS(method, "constant")) {

    // y1 = y0

    // do nothing

  } else if (CCTK_EQUALS(method, "RK4")) {

    // k1 = f(y0)
    // k2 = f(y0 + h/2 k1)
    // k3 = f(y0 + h/2 k2)
    // k4 = f(y0 + h k3)
    // y1 = y0 + h/6 k1 + h/3 k2 + h/3 k3 + h/6 k4

    // The table row must describe the sequence below
    assert(rk.nstages == 4);

    const auto old = var.copy();

    open_step();

    calcrhs(1);
    auto kaccum = rhs.copy();
    calcupdate(1, dt / 2, 1.0, reals<1>{dt / 2}, states<1>{&rhs});
    calcys_rmbnd();
    calcpoststep();

    calcrhs(2);
    statecomp_t::lincomb(kaccum, 1.0, reals<1>{2.0}, states<1>{&rhs},
                         CarpetX::make_valid_int());
    calcupdate(2, dt / 2, 0.0, reals<2>{1.0, dt / 2}, states<2>{&old, &rhs});
    calcys_rmbnd();
    calcpoststep();

    calcrhs(3);
    statecomp_t::lincomb(kaccum, 1.0, reals<1>{2.0}, states<1>{&rhs},
                         CarpetX::make_valid_int());
    calcupdate(3, dt, 0.0, reals<2>{1.0, dt}, states<2>{&old, &rhs});
    calcys_rmbnd();
    calcpoststep();

    calcrhs(4);
    calcupdate(4, dt, 0.0, reals<3>{1.0, dt / 6, dt / 6},
               states<3>{&old, &kaccum, &rhs});
    calcys_rmbnd();
    calcpoststep();

  } else if (CCTK_EQUALS(method, "SSPRK3")) {

    // k1 = f(y0)
    // k2 = f(y0 + h k1)
    // k3 = f(y0 + h/4 k1 + h/4 k2)
    // y1 = y0 + h/6 k1 + h/6 k2 + 2/3 h k3

    // The table row must describe the sequence below
    assert(rk.nstages == 3);

    const auto old = var.copy();

    open_step();

    calcrhs(1);
    const auto k1 = rhs.copy();
    calcupdate(1, dt, 1.0, reals<1>{dt}, states<1>{&rhs}); // var = y0 + dt*k1
    calcys_rmbnd();
    calcpoststep();

    calcrhs(2);
    const auto k2 = rhs.copy();
    calcupdate(2, dt / 2, 0.0, reals<3>{1.0, dt / 4, dt / 4},
               states<3>{&old, &k1, &k2});
    calcys_rmbnd();
    calcpoststep();

    calcrhs(3);
    calcupdate(3, dt, 0.0, reals<4>{1.0, dt / 6, dt / 6, 2 * dt / 3},
               states<4>{&old, &k1, &k2, &rhs});
    calcys_rmbnd();
    calcpoststep();

  } else {
    CCTK_VERROR("ODESolvers::method = \"%s\" has a dense output in "
                "rk_methods.hxx but no branch in the subcycling solver",
                method);
  }
}

extern "C" void ODESolvers_CheckTimelevels(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTS_ODESolvers_CheckTimelevels;
  DECLARE_CCTK_PARAMETERS;

  for (int gi = 0; gi < CCTK_NumGroups(); ++gi) {
    if (get_group_rhs(gi) < 0)
      continue; // not an ODE-evolved group
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
}

} // namespace ODESolvers
