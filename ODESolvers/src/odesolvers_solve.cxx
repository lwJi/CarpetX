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
  integrator.dense_output = rk.dense_output;
  if (!rk.implicit)
    integrator.b.assign(rk.b.begin(), rk.b.begin() + rk.nstages);
  for (int gi = 0; gi < CCTK_NumGroups(); ++gi) {
    if (CCTK_GroupTypeI(gi) != CCTK_GF)
      continue;
    const int rhs_gi = get_group_rhs(gi);
    if (rhs_gi >= 0)
      integrator.integrated.push_back(CarpetX::rk_pair_t{gi, rhs_gi});
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
  std::vector<int> var_groups, rhs_groups, dep_groups;
  int nvars = 0;
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
        var.entries.push_back({group, std::nullopt});
        rhs.entries.push_back({rhs_group, std::nullopt});
        if (do_accumulate_nvars) {
          nvars += CarpetX::numvars(group);
          var_groups.push_back(gi);
          rhs_groups.push_back(rhs_gi);
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

  for (const int gi : var_groups)
    assert(std::find(dep_groups.begin(), dep_groups.end(), gi) ==
           dep_groups.end());
  for (const int gi : rhs_groups)
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
    rhs.require_valid(CarpetX::make_valid_int(),
                      "ODESolvers after calling ODESolvers_RHS");
    step.value().rhs_computed();
  };
  const auto calcupdate = [&](const int n, const CCTK_REAL c,
                              const CCTK_REAL a0, const auto &as,
                              const auto &vars) {
    {
      CarpetX::Interval interval_lincomb(timer_lincomb);
      statecomp_t::lincomb(var, a0, as, vars, CarpetX::make_valid_int());
      var.require_valid(CarpetX::make_valid_int(),
                        "ODESolvers after defining new state vector");
      CarpetX::invalidate(window, dep_groups, 0,
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

    assert(rk.nstages == 1);

    open_step();

    calcrhs(1);
    calcupdate(1, dt, 1.0, reals<1>{dt}, states<1>{&rhs});

  } else if (CCTK_EQUALS(method, "RK2")) {

    // k1 = f(y0)
    // k2 = f(y0 + h/2 k1)
    // y1 = y0 + h k2

    assert(rk.nstages == 2);

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

    assert(rk.nstages == 3);

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

    assert(rk.nstages == 3);

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

    assert(rk.nstages == 4);

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

    typedef CCTK_REAL T;
    const auto R = [](T x, T y) { return x / y; };
    const vector<tuple<T, vector<T> > > tableau{
        {/* 1 */ 0, {}},                                           //
        {/* 2 */ R(2, 27), {R(2, 27)}},                            //
        {/* 3 */ R(1, 9), {R(1, 36), R(3, 36)}},                   //
        {/* 4 */ R(1, 6), {R(1, 24), 0, R(3, 24)}},                //
        {/* 5 */ R(5, 12), {R(20, 48), 0, R(-75, 48), R(75, 48)}}, //
        {/* 6 */ R(1, 2), {R(1, 20), 0, 0, R(5, 20), R(4, 20)}},   //
        {/* 7 */ R(5, 6),
         {R(-25, 108), 0, 0, R(125, 108), R(-260, 108), R(250, 108)}}, //
        {/* 8 */ R(1, 6),
         {R(31, 300), 0, 0, 0, R(61, 225), R(-2, 9), R(13, 900)}}, //
        {/* 9 */ R(2, 3),
         {2, 0, 0, R(-53, 6), R(704, 45), R(-107, 9), R(67, 90), 3}}, //
        {/* 10 */ R(1, 3),
         {R(-91, 108), 0, 0, R(23, 108), R(-976, 135), R(311, 54),
          R(-19, 60), R(17, 6), R(-1, 12)}}, //
        {/* 11 */ 1,
         {R(2383, 4100), 0, 0, R(-341, 164), R(4496, 1025), R(-301, 82),
          R(2133, 4100), R(45, 82), R(45, 164), R(18, 41)}}, //
    };

    const size_t nsteps = tableau.size();
    assert(int(nsteps) == rk.nstages);
    for (size_t step = 0; step < nsteps; ++step) {
      assert(get<1>(tableau.at(step)).size() == step);
      const auto &c = get<0>(tableau.at(step));
      const auto &as = get<1>(tableau.at(step));
      T x = 0;
      for (const auto &a : as)
        x += a;
      assert(fabs(x - c) <= 10 * numeric_limits<T>::epsilon());
    }

    const auto old = copy_state(var);

    open_step();

    vector<statecomp_t> ks;
    ks.reserve(nsteps);
    for (size_t step = 0; step < nsteps; ++step) {
      if (step > 0) {
        const auto &c = get<0>(tableau.at(step));
        const auto &as = get<1>(tableau.at(step));

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
      if (rk.b.at(i) != 0) {
        factors.push_back(rk.b.at(i) * dt);
        srcs.push_back(&ks.at(i));
      }
    }
    calcupdate(nsteps, dt, 0.0, factors, srcs);

  } else if (CCTK_EQUALS(method, "DP87")) {

    typedef CCTK_REAL T;
    const auto R = [](T x, T y) { return x / y; };
    // doi:10.1016/0771-050X(81)90010-3
    const vector<vector<T> > tableau{
        {
            {/*1*/},                                    //
            {/*2*/ R(1, 18)},                           //
            {/*3*/ R(1, 48), R(1, 16)},                 //
            {/*4*/ R(1, 32), 0, R(3, 32)},              //
            {/*5*/ R(5, 16), 0, -R(75, 64), R(75, 64)}, //
            {/*6*/ R(3, 80), 0, 0, R(3, 16), R(3, 20)}, //
            {/*7*/ R(29443841, 614563906), 0, 0, R(77736538, 692538347),
             -R(28693883, 1125000000), R(23124283, 1800000000)}, //
            {/*8*/ R(16016141, 946692911), 0, 0, R(61564180, 158732637),
             R(22789713, 633445777), R(545815736, 2771057229),
             -R(180193667, 1043307555)}, //
            {/*9*/ R(39632708, 573591083), 0, 0, -R(433636366, 683701615),
             -R(421739975, 2616292301), R(100302831, 723423059),
             R(790204164, 839813087), R(800635310, 3783071287)}, //
            {/*10*/ R(246121993, 1340847787), 0, 0,
             -R(37695042795, 15268766246), -R(309121744, 1061227803),
             -R(12992083, 490766935), R(6005943493, 2108947869),
             R(393006217, 1396673457), R(123872331, 1001029789)}, //
            {/*11*/ -R(1028468189, 846180014), 0, 0, R(8478235783, 508512852),
             R(1311729495, 1432422823), -R(10304129995, 1701304382),
             -R(48777925059, 3047939560), R(15336726248, 1032824649),
             -R(45442868181, 3398467696), R(3065993473, 597172653)}, //
            {/*12*/ R(185892177, 718116043), 0, 0, -R(3185094517, 667107341),
             -R(477755414, 1098053517), -R(703635378, 230739211),
             R(5731566787, 1027545527), R(5232866602, 850066563),
             -R(4093664535, 808688257), R(3962137247, 1805957418),
             R(65686358, 487910083)}, //
            {/*13*/ R(403863854, 491063109), 0, 0, -R(5068492393, 434740067),
             -R(411421997, 543043805), R(652783627, 914296604),
             R(11173962825, 925320556), -R(13158990841, 6184727034),
             R(3936647629, 1978049680), -R(160528059, 685178525),
             R(248638103, 1413531060), 0}, //
        },
    };

    const size_t nsteps = tableau.size();
    assert(int(nsteps) == rk.nstages);
    for (size_t step = 0; step < nsteps; ++step)
      assert(tableau.at(step).size() == step);

    const auto old = copy_state(var);

    open_step();

    vector<statecomp_t> ks;
    ks.reserve(nsteps);
    for (size_t step = 0; step < nsteps; ++step) {
      if (step > 0) {
        const auto &as = tableau.at(step);
        T c = 0;
        for (const auto &a : as)
          c += a;

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
      if (rk.b.at(i) != 0) {
        factors.push_back(rk.b.at(i) * dt);
        srcs.push_back(&ks.at(i));
      }
    }
    calcupdate(nsteps, dt, 0.0, factors, srcs);

  } else if (CCTK_EQUALS(method, "IMEX122") ||
             CCTK_EQUALS(method, "Implicit Euler")) {
    // Implicit definition:
    //   y1 = y0 + h/2 f(y0) + h/2 g(y1)
    //   y2 = y0 + h f(y1) + h g(y1)

    // Implicit RHS:
    //   u1 = G(u0, h)   where   u1 = u0 + h g(u1)

    // Explicit definition:
    //   k1 = f(y0)
    //   y1 = G(y0 + h/2 k1, h/2)
    //   k'2 = (y1 - y0 - h/2 k1) / (h/2)
    //   k2 = f(y1)
    //   y2 = y0 + h k2 + h k'2

    const auto y0 = var.copy();

    if (verbose)
      CCTK_VINFO("Calculating RHS #1 at t=%g", double(old_time));
    CarpetX::run_schedule_group_at(cctkGH, "ODESolvers_RHS", old_time, dt);
    const auto k1 = rhs.copy();

    statecomp_t::lincomb(var, 1, make_array(dt / 2), make_array(&rhs),
                         CarpetX::make_valid_int());
    var.require_valid(CarpetX::make_valid_int(),
                      "ODESolvers after defining new state vector");
    CarpetX::invalidate(window, dep_groups, 0,
                        "ODESolvers updated the state vector");
    CarpetX::run_schedule_group_at(cctkGH, "ODESolvers_PostStep",
                                   old_time + dt / 2, dt);

    if (verbose)
      CCTK_VINFO("Taking implicit step #1 at t=%g with dt=%g",
                 double(old_time + dt / 2), double(dt / 2));
    CarpetX::run_schedule_group_at(cctkGH, "ODESolvers_ImplicitStep",
                                   old_time + dt / 2, dt / 2);

    CarpetX::run_schedule_group_at(cctkGH, "ODESolvers_PostStep", old_time + dt,
                                   dt);
    const auto y1 = var.copy();

    statecomp_t kprime2;
    statecomp_t::lincomb(kprime2, 0,
                         make_array(-CCTK_REAL(1), +CCTK_REAL(1), -dt / 2),
                         make_array(&y0, &y1, &k1), CarpetX::make_valid_int());

    if (verbose)
      CCTK_VINFO("Calculating RHS #2 at t=%g", double(old_time + dt));
    CarpetX::run_schedule_group_at(cctkGH, "ODESolvers_RHS", old_time + dt, dt);
    const auto k2 = rhs.copy();

    statecomp_t::lincomb(var, 0, make_array(CCTK_REAL(1), dt, dt),
                         make_array(&y0, &k2, &kprime2),
                         CarpetX::make_valid_int());
    var.require_valid(CarpetX::make_valid_int(),
                      "ODESolvers after defining new state vector");
    CarpetX::invalidate(window, dep_groups, 0,
                        "ODESolvers updated the state vector");

  } else {
    assert(0);
  }
}

} // namespace ODESolvers
