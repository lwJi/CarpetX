#include "rk_methods.hxx"
#include "solve.hxx"

#include <carpetx_rk.hxx>

namespace ODESolvers {
using namespace std;

extern "C" void ODESolvers_InitConstants(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTS_ODESolvers_InitConstants;
  DECLARE_CCTK_PARAMETERS;

  *do_substeps = 0;

  const rk_method_t &rk = rk_method(method);
  CarpetX::rk_config_t integrator;
  integrator.dense_output = registered_dense_output(rk);
  std::visit(overloaded{[&](const frozen_t &) {
                          integrator.b.assign(CarpetX::rk_dense_output_stages(
                                                  integrator.dense_output),
                                              0);
                        },
                        [&](const explicit_rk_t &scheme) {
                          integrator.b.assign(scheme.tableau.b.begin(),
                                              scheme.tableau.b.begin() +
                                                  scheme.tableau.nstages);
                        }},
             rk.scheme);
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

extern "C" void ODESolvers_Solve(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTS_ODESolvers_Solve;
  DECLARE_CCTK_PARAMETERS;

  static bool did_output = false;
  if (verbose || !did_output)
    CCTK_VINFO("ODE integrator is %s", method);
  did_output = true;

  static CarpetX::Timer timer("ODESolvers::Solve");
  CarpetX::Interval interval(timer);

  const CCTK_REAL dt = cctk_delta_time;

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

  const CCTK_REAL old_time = cctkGH->cctk_time - dt;
  CCTK_REAL t = old_time;

  static CarpetX::Timer timer_lincomb("ODESolvers::Solve::lincomb");
  static CarpetX::Timer timer_rhs("ODESolvers::Solve::rhs");
  static CarpetX::Timer timer_poststep("ODESolvers::Solve::poststep");

  const rk_method_t &rk = rk_method(method);

  const auto copy_state = [](const statecomp_t &state) { return state.copy(); };
  std::optional<CarpetX::rk_step_t> step;
  const auto open_step = [&]() {
    step.emplace(CarpetX::open_rk_step(window));
  };
  const auto calcrhs = [&](const int n) {
    CarpetX::Interval interval_rhs(timer_rhs);
    if (verbose)
      CCTK_VINFO("Calculating RHS #%d at t=%g", n, double(t));
    CarpetX::run_schedule_group_at(cctkGH, "ODESolvers_RHS", t, dt);
    step.value().rhs_computed();
  };
  const auto calcupdate = [&](const int n, const CCTK_REAL c,
                              const CCTK_REAL a0, const auto &as,
                              const auto &vars) {
    {
      CarpetX::Interval interval_lincomb(timer_lincomb);
      statecomp_t::lincomb(var, a0, as, vars, CarpetX::make_valid_int());
      CarpetX::invalidate(window, dep_groups, CarpetX::tl_t{0},
                          "ODESolvers updated the state vector");
    }
    step.value().state_updated();
    {
      CarpetX::Interval interval_poststep(timer_poststep);
      t = old_time + c;
      CarpetX::run_schedule_group_at(cctkGH, "ODESolvers_PostStep", t, dt);
      if (verbose)
        CCTK_VINFO("Calculated new state #%d at t=%g", n, double(t));
    }
  };

  if (CCTK_EQUALS(method, "constant")) {

    // y1 = y0

    // do nothing

  } else if (CCTK_EQUALS(method, "Euler")) {

    // k1 = f(y0)
    // y1 = y0 + h k1

    assert(std::get<explicit_rk_t>(rk.scheme).tableau.nstages == 1);

    open_step();

    calcrhs(1);
    calcupdate(1, dt, 1.0, reals<1>{dt}, states<1>{&rhs});

  } else if (CCTK_EQUALS(method, "RK2")) {

    // k1 = f(y0)
    // k2 = f(y0 + h/2 k1)
    // y1 = y0 + h k2

    assert(std::get<explicit_rk_t>(rk.scheme).tableau.nstages == 2);

    const auto old = copy_state(var);

    open_step();

    calcrhs(1);
    calcupdate(1, dt / 2, 1.0, reals<1>{dt / 2}, states<1>{&rhs});

    calcrhs(2);
    calcupdate(2, dt, 0.0, reals<2>{1.0, dt}, states<2>{&old, &rhs});

  } else if (CCTK_EQUALS(method, "RK3")) {

    // k1 = f(y0)
    // k2 = f(y0 + h/2 k1)
    // k3 = f(y0 - h k1 + 2 h k2)
    // y1 = y0 + h/6 k1 + 2/3 h k2 + h/6 k3

    assert(std::get<explicit_rk_t>(rk.scheme).tableau.nstages == 3);

    const auto old = copy_state(var);

    open_step();

    calcrhs(1);
    const auto k1 = copy_state(rhs);
    calcupdate(1, dt / 2, 1.0, reals<1>{dt / 2}, states<1>{&k1});

    calcrhs(2);
    const auto k2 = copy_state(rhs);
    calcupdate(2, dt, 0.0, reals<3>{1.0, -dt, 2 * dt},
               states<3>{&old, &k1, &k2});

    calcrhs(3);
    calcupdate(3, dt, 0.0, reals<4>{1.0, dt / 6, 2 * dt / 3, dt / 6},
               states<4>{&old, &k1, &k2, &rhs});

  } else if (CCTK_EQUALS(method, "SSPRK3")) {

    // doi:10.1016/0021-9991(88)90177-5

    // Low-storage Shu-Osher form (Shu & Osher 1988):
    //   u1 = u0 + h L(u0)
    //   u2 = 3/4 u0 + 1/4 u1 + 1/4 h L(u1)
    //   u3 = 1/3 u0 + 2/3 u2 + 2/3 h L(u2)
    // i.e. y1 = y0 + h/6 L(u0) + h/6 L(u1) + 2/3 h L(u2)

    assert(std::get<explicit_rk_t>(rk.scheme).tableau.nstages == 3);

    const auto old = copy_state(var);

    open_step();

    calcrhs(1);
    calcupdate(1, dt, 1.0, reals<1>{dt}, states<1>{&rhs});

    calcrhs(2);
    calcupdate(2, dt / 2, 1.0 / 4.0, reals<2>{3.0 / 4.0, dt / 4},
               states<2>{&old, &rhs});

    calcrhs(3);
    calcupdate(3, dt, 2.0 / 3.0, reals<2>{1.0 / 3.0, 2 * dt / 3},
               states<2>{&old, &rhs});

  } else if (CCTK_EQUALS(method, "RK4")) {

    // k1 = f(y0)
    // k2 = f(y0 + h/2 k1)
    // k3 = f(y0 + h/2 k2)
    // k4 = f(y0 + h k3)
    // y1 = y0 + h/6 k1 + h/3 k2 + h/3 k3 + h/6 k4

    assert(std::get<explicit_rk_t>(rk.scheme).tableau.nstages == 4);

    const auto old = copy_state(var);

    open_step();

    calcrhs(1);
    auto kaccum = copy_state(rhs);
    calcupdate(1, dt / 2, 1.0, reals<1>{dt / 2}, states<1>{&kaccum});

    calcrhs(2);
    {
      CarpetX::Interval interval_lincomb(timer_lincomb);
      statecomp_t::lincomb(kaccum, 1.0, reals<1>{2.0}, states<1>{&rhs},
                           CarpetX::make_valid_int());
    }
    calcupdate(2, dt / 2, 0.0, reals<2>{1.0, dt / 2}, states<2>{&old, &rhs});

    calcrhs(3);
    {
      CarpetX::Interval interval_lincomb(timer_lincomb);
      statecomp_t::lincomb(kaccum, 1.0, reals<1>{2.0}, states<1>{&rhs},
                           CarpetX::make_valid_int());
    }
    calcupdate(3, dt, 0.0, reals<2>{1.0, dt}, states<2>{&old, &rhs});

    calcrhs(4);
    calcupdate(4, dt, 0.0, reals<3>{1.0, dt / 6, dt / 6},
               states<3>{&old, &kaccum, &rhs});

  } else if (CCTK_EQUALS(method, "RKF78")) {

    const tableau_t &tab = std::get<explicit_rk_t>(rk.scheme).tableau;
    const size_t nsteps = tab.nstages;

    const auto old = copy_state(var);

    open_step();

    vector<statecomp_t> ks;
    ks.reserve(nsteps);
    for (size_t step = 0; step < nsteps; ++step) {
      if (step > 0) {
        const auto &c = tab.c[step];
        const auto &as = tab.a[step];

        // Add scaled RHS to state vector
        vector<CCTK_REAL> factors;
        vector<const statecomp_t *> srcs;
        factors.reserve(as.size() + 1);
        srcs.reserve(as.size() + 1);
        factors.push_back(1.0);
        srcs.push_back(&old);
        for (size_t i = 0; i < as.size(); ++i) {
          if (as.at(i) != 0) {
            factors.push_back(as.at(i) * dt);
            srcs.push_back(&ks.at(i));
          }
        }
        calcupdate(step, c * dt, 0.0, factors, srcs);
        // TODO: Deallocate ks that are not needed any more
      }

      calcrhs(step + 1);
      ks.push_back(copy_state(rhs));
    }

    vector<CCTK_REAL> factors;
    vector<const statecomp_t *> srcs;
    factors.reserve(nsteps + 1);
    srcs.reserve(nsteps + 1);
    factors.push_back(1);
    srcs.push_back(&old);
    for (size_t i = 0; i < nsteps; ++i) {
      if (tab.b.at(i) != 0) {
        factors.push_back(tab.b.at(i) * dt);
        srcs.push_back(&ks.at(i));
      }
    }
    calcupdate(nsteps, dt, 0.0, factors, srcs);

  } else if (CCTK_EQUALS(method, "DP87")) {

    const tableau_t &tab = std::get<explicit_rk_t>(rk.scheme).tableau;
    const size_t nsteps = tab.nstages;

    const auto old = copy_state(var);

    open_step();

    vector<statecomp_t> ks;
    ks.reserve(nsteps);
    for (size_t step = 0; step < nsteps; ++step) {
      if (step > 0) {
        const auto &c = tab.c[step];
        const auto &as = tab.a[step];

        // Add scaled RHS to state vector
        vector<CCTK_REAL> factors;
        vector<const statecomp_t *> srcs;
        factors.reserve(as.size() + 1);
        srcs.reserve(as.size() + 1);
        factors.push_back(1.0);
        srcs.push_back(&old);
        for (size_t i = 0; i < as.size(); ++i) {
          if (as.at(i) != 0) {
            factors.push_back(as.at(i) * dt);
            srcs.push_back(&ks.at(i));
          }
        }
        calcupdate(step, c * dt, 0.0, factors, srcs);
        // TODO: Deallocate ks that are not needed any more
      }

      calcrhs(step + 1);
      ks.push_back(copy_state(rhs));
    }

    vector<CCTK_REAL> factors;
    vector<const statecomp_t *> srcs;
    factors.reserve(nsteps + 1);
    srcs.reserve(nsteps + 1);
    factors.push_back(1);
    srcs.push_back(&old);
    for (size_t i = 0; i < nsteps; ++i) {
      if (tab.b.at(i) != 0) {
        factors.push_back(tab.b.at(i) * dt);
        srcs.push_back(&ks.at(i));
      }
    }
    calcupdate(nsteps, dt, 0.0, factors, srcs);

  } else {
    assert(0);
  }
}

} // namespace ODESolvers
