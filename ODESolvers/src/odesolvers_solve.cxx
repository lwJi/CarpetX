#include "rk_methods.hxx"
#include "solve.hxx"

#include "../../CarpetX/src/subcycling.hxx"

namespace ODESolvers {
using namespace std;

extern "C" void ODESolvers_InitConstants(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTS_ODESolvers_InitConstants;
  DECLARE_CCTK_PARAMETERS;

  *do_substeps = 0;

  CarpetX::ghext->num_rk_stages = rk_method(method).nstages;

  std::vector<bool> &integrated = CarpetX::ghext->rk_integrated_group;
  integrated.assign(CCTK_NumGroups(), false);
  for (int gi = 0; gi < CCTK_NumGroups(); ++gi)
    integrated.at(gi) =
        CCTK_GroupTypeI(gi) == CCTK_GF && get_group_rhs(gi) >= 0;
}

extern "C" void ODESolvers_CheckTags(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTS_ODESolvers_CheckTags;

  for (int gi = 0; gi < CCTK_NumGroups(); ++gi) {
    if (!CarpetX::group_is_integrated(gi))
      continue;
    if (!CarpetX::get_group_checkpoint_flag(gi))
      CCTK_VERROR("Group \"%s\" is integrated by ODESolvers (rhs= tag) but "
                  "not checkpointed (checkpoint=\"no\"). Integrated state "
                  "must be checkpointed: a checkpoint would omit its state "
                  "(and, under subcycling, its RK source bands), so recovery "
                  "could not restore it. Remove checkpoint=\"no\".",
                  CCTK_FullGroupName(gi));
    if (!CarpetX::get_group_evolve_flag(gi))
      CCTK_VERROR("Group \"%s\" is integrated by ODESolvers (rhs= tag) but "
                  "declared evolve=\"no\". Regrid would not reliably refill "
                  "its state. Remove evolve=\"no\".",
                  CCTK_FullGroupName(gi));
  }
}

extern "C" void ODESolvers_Solve(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTS_ODESolvers_Solve;
  DECLARE_CCTK_PARAMETERS;

  static bool did_output = false;
  if (verbose || !did_output)
    CCTK_VINFO("ODE integrator is %s", method);
  did_output = true;

  static Timer timer("ODESolvers::Solve");
  Interval interval(timer);

  const CCTK_REAL dt = cctk_delta_time;
  const int tl = 0;

  static Timer timer_setup("ODESolvers::Solve::setup");
  std::optional<Interval> interval_setup(timer_setup);

  statecomp_t var, rhs;
  std::vector<int> var_groups, rhs_groups, dep_groups;
  int nvars = 0;
  bool do_accumulate_nvars = true;
  assert(CarpetX::active_levels);
  CarpetX::active_levels->loop_serially([&](const auto &leveldata) {
    for (const auto &groupdataptr : leveldata.groupdata) {
      if (groupdataptr == nullptr)
        continue;

      auto &groupdata = *groupdataptr;
      const int rhs_gi = get_group_rhs(groupdata.groupindex);
      if (rhs_gi >= 0) {
        assert(rhs_gi != groupdata.groupindex);
        auto &rhs_groupdata = *leveldata.groupdata.at(rhs_gi);
        assert(rhs_groupdata.numvars == groupdata.numvars);
        var.groupdatas.push_back(&groupdata);
        var.mfabs.push_back(groupdata.mfab.at(tl).get());
        rhs.groupdatas.push_back(&rhs_groupdata);
        rhs.mfabs.push_back(rhs_groupdata.mfab.at(tl).get());
        if (do_accumulate_nvars) {
          nvars += groupdata.numvars;
          var_groups.push_back(groupdata.groupindex);
          rhs_groups.push_back(rhs_gi);
          const auto &dependents = get_group_dependents(groupdata.groupindex);
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

  {
    static Timer timer_alloc_temps("ODESolvers::Solve::alloc_temps");
    Interval interval_alloc_temps(timer_alloc_temps);
    statecomp_t::init_tmp_mfabs();
  }

  const CCTK_REAL saved_time = cctkGH->cctk_time;
  const CCTK_REAL old_time = cctkGH->cctk_time - dt;

  static Timer timer_lincomb("ODESolvers::Solve::lincomb");
  static Timer timer_rhs("ODESolvers::Solve::rhs");
  static Timer timer_poststep("ODESolvers::Solve::poststep");

  const rk_method_t &rk = rk_method(method);

  const auto copy_state = [](const auto &var, const valid_t where) {
    return var.copy(where);
  };
  const auto calcrhs = [&](const int n, const CCTK_REAL b_n) {
    Interval interval_rhs(timer_rhs);
    if (verbose)
      CCTK_VINFO("Calculating RHS #%d at t=%g", n, double(cctkGH->cctk_time));
    CallScheduleGroup(cctkGH, "ODESolvers_RHS");
    rhs.check_valid(make_valid_int(),
                    "ODESolvers after calling ODESolvers_RHS");
    active_levels->loop_coarse_to_fine([&](const auto &restrict leveldata) {
      CarpetX::AccumulateFluxes(leveldata.patch, leveldata.level, n, b_n * dt);
    });
    synchronize();
  };
  const auto calcupdate = [&](const int n, const CCTK_REAL c,
                              const CCTK_REAL a0, const auto &as,
                              const auto &vars) {
    {
      Interval interval_lincomb(timer_lincomb);
      statecomp_t::lincomb(var, a0, as, vars, make_valid_int());
      var.check_valid(make_valid_int(),
                      "ODESolvers after defining new state vector");
      mark_invalid(dep_groups);
    }
    if (n == rk.nstages) {
      static Timer timer_reflux("ODESolvers::Solve::reflux");
      Interval interval_reflux(timer_reflux);
      CarpetX::Reflux(cctkGH, 0, CarpetX::ghext->num_levels());
    }
    {
      Interval interval_poststep(timer_poststep);
      *const_cast<CCTK_REAL *>(&cctkGH->cctk_time) = old_time + c;
      CallScheduleGroup(cctkGH, "ODESolvers_PostStep");
      if (verbose)
        CCTK_VINFO("Calculated new state #%d at t=%g", n,
                   double(cctkGH->cctk_time));
    }
  };

  *const_cast<CCTK_REAL *>(&cctkGH->cctk_time) = old_time;

  if (CCTK_EQUALS(method, "constant")) {

  } else if (CCTK_EQUALS(method, "Euler")) {

    assert(rk.nstages == 1);

    calcrhs(1, rk.b.at(0));
    calcupdate(1, dt, 1.0, reals<1>{dt}, states<1>{&rhs});

  } else if (CCTK_EQUALS(method, "RK2")) {

    assert(rk.nstages == 2);

    const auto old = copy_state(var, make_valid_all());

    calcrhs(1, rk.b.at(0));
    calcupdate(1, dt / 2, 1.0, reals<1>{dt / 2}, states<1>{&rhs});

    calcrhs(2, rk.b.at(1));
    calcupdate(2, dt, 0.0, reals<2>{1.0, dt}, states<2>{&old, &rhs});

  } else if (CCTK_EQUALS(method, "RK3")) {

    assert(rk.nstages == 3);

    const auto old = copy_state(var, make_valid_all());

    calcrhs(1, rk.b.at(0));
    const auto k1 = copy_state(rhs, make_valid_int());
    calcupdate(1, dt / 2, 1.0, reals<1>{dt / 2}, states<1>{&k1});

    calcrhs(2, rk.b.at(1));
    const auto k2 = copy_state(rhs, make_valid_int());
    calcupdate(2, dt, 0.0, reals<3>{1.0, -dt, 2 * dt},
               states<3>{&old, &k1, &k2});

    calcrhs(3, rk.b.at(2));
    calcupdate(3, dt, 0.0, reals<4>{1.0, dt / 6, 2 * dt / 3, dt / 6},
               states<4>{&old, &k1, &k2, &rhs});

  } else if (CCTK_EQUALS(method, "SSPRK3")) {

    // doi:10.1016/0021-9991(88)90177-5

    assert(rk.nstages == 3);

    const auto old = copy_state(var, make_valid_all());

    calcrhs(1, rk.b.at(0));
    calcupdate(1, dt, 1.0, reals<1>{dt}, states<1>{&rhs});

    calcrhs(2, rk.b.at(1));
    calcupdate(2, dt / 2, 1.0 / 4.0, reals<2>{3.0 / 4.0, dt / 4},
               states<2>{&old, &rhs});

    calcrhs(3, rk.b.at(2));
    calcupdate(3, dt, 2.0 / 3.0, reals<2>{1.0 / 3.0, 2 * dt / 3},
               states<2>{&old, &rhs});

  } else if (CCTK_EQUALS(method, "RK4")) {

    assert(rk.nstages == 4);

    const auto old = copy_state(var, make_valid_all());

    calcrhs(1, rk.b.at(0));
    const auto kaccum = copy_state(rhs, make_valid_int());
    calcupdate(1, dt / 2, 1.0, reals<1>{dt / 2}, states<1>{&kaccum});

    calcrhs(2, rk.b.at(1));
    {
      Interval interval_lincomb(timer_lincomb);
      statecomp_t::lincomb(kaccum, 1.0, reals<1>{2.0}, states<1>{&rhs},
                           make_valid_int());
    }
    calcupdate(2, dt / 2, 0.0, reals<2>{1.0, dt / 2}, states<2>{&old, &rhs});

    calcrhs(3, rk.b.at(2));
    {
      Interval interval_lincomb(timer_lincomb);
      statecomp_t::lincomb(kaccum, 1.0, reals<1>{2.0}, states<1>{&rhs},
                           make_valid_int());
    }
    calcupdate(3, dt, 0.0, reals<2>{1.0, dt}, states<2>{&old, &rhs});

    calcrhs(4, rk.b.at(3));
    calcupdate(4, dt, 0.0, reals<3>{1.0, dt / 6, dt / 6},
               states<3>{&old, &kaccum, &rhs});

  } else if (CCTK_EQUALS(method, "RKF78")) {

    typedef CCTK_REAL T;
    const auto R = [](T x, T y) { return x / y; };
    const tuple<vector<tuple<T, vector<T> > >, vector<T> > tableau{
        {
            {0, {}},                                           //
            {R(2, 27), {R(2, 27)}},                            //
            {R(1, 9), {R(1, 36), R(3, 36)}},                   //
            {R(1, 6), {R(1, 24), 0, R(3, 24)}},                //
            {R(5, 12), {R(20, 48), 0, R(-75, 48), R(75, 48)}}, //
            {R(1, 2), {R(1, 20), 0, 0, R(5, 20), R(4, 20)}},   //
            {R(5, 6),
             {R(-25, 108), 0, 0, R(125, 108), R(-260, 108), R(250, 108)}}, //
            {R(1, 6),
             {R(31, 300), 0, 0, 0, R(61, 225), R(-2, 9), R(13, 900)}}, //
            {R(2, 3),
             {2, 0, 0, R(-53, 6), R(704, 45), R(-107, 9), R(67, 90), 3}}, //
            {R(1, 3),
             {R(-91, 108), 0, 0, R(23, 108), R(-976, 135), R(311, 54),
              R(-19, 60), R(17, 6), R(-1, 12)}}, //
            {1,
             {R(2383, 4100), 0, 0, R(-341, 164), R(4496, 1025), R(-301, 82),
              R(2133, 4100), R(45, 82), R(45, 164), R(18, 41)}}, //
        },
        {
            R(41, 840),
            0,
            0,
            0,
            0,
            R(34, 105),
            R(9, 35),
            R(9, 35),
            R(9, 280),
            R(9, 280),
            R(41, 840),
        }};

    const size_t nsteps = get<0>(tableau).size();
    assert(int(nsteps) == rk_method(method).nstages);
    {
      for (size_t step = 0; step < nsteps; ++step) {
        assert(get<1>(get<0>(tableau).at(step)).size() == step);
        const auto &c = get<0>(get<0>(tableau).at(step));
        const auto &as = get<1>(get<0>(tableau).at(step));
        T x = 0;
        for (const auto &a : as)
          x += a;
        assert(fabs(x - c) <= 10 * numeric_limits<T>::epsilon());
      }
      assert(get<1>(tableau).size() == nsteps);
      const auto &bs = get<1>(tableau);
      T x = 0;
      for (const auto &b : bs)
        x += b;
      assert(fabs(x - 1) <= 10 * numeric_limits<T>::epsilon());
    }

    const auto old = copy_state(var, make_valid_all());

    const auto &bs = get<1>(tableau);

    vector<statecomp_t> ks;
    ks.reserve(nsteps);
    for (size_t step = 0; step < nsteps; ++step) {
      if (step > 0) {
        const auto &c = get<0>(get<0>(tableau).at(step));
        const auto &as = get<1>(get<0>(tableau).at(step));

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
      }

      calcrhs(step + 1, bs.at(step));
      ks.push_back(copy_state(rhs, make_valid_int()));
    }

    vector<CCTK_REAL> factors;
    vector<const statecomp_t *> srcs;
    factors.reserve(bs.size() + 1);
    srcs.reserve(bs.size() + 1);
    factors.push_back(1);
    srcs.push_back(&old);
    for (size_t i = 0; i < bs.size(); ++i) {
      if (bs.at(i) != 0) {
        factors.push_back(bs.at(i) * dt);
        srcs.push_back(&ks.at(i));
      }
    }
    calcupdate(nsteps, dt, 0.0, factors, srcs);

  } else if (CCTK_EQUALS(method, "DP87")) {

    typedef CCTK_REAL T;
    const auto R = [](T x, T y) { return x / y; };
    // doi:10.1016/0771-050X(81)90010-3
    const tuple<vector<vector<T> >, vector<T> > tableau{
        {
            {},                                    //
            {R(1, 18)},                           //
            {R(1, 48), R(1, 16)},                 //
            {R(1, 32), 0, R(3, 32)},              //
            {R(5, 16), 0, -R(75, 64), R(75, 64)}, //
            {R(3, 80), 0, 0, R(3, 16), R(3, 20)}, //
            {R(29443841, 614563906), 0, 0, R(77736538, 692538347),
             -R(28693883, 1125000000), R(23124283, 1800000000)}, //
            {R(16016141, 946692911), 0, 0, R(61564180, 158732637),
             R(22789713, 633445777), R(545815736, 2771057229),
             -R(180193667, 1043307555)}, //
            {R(39632708, 573591083), 0, 0, -R(433636366, 683701615),
             -R(421739975, 2616292301), R(100302831, 723423059),
             R(790204164, 839813087), R(800635310, 3783071287)}, //
            {R(246121993, 1340847787), 0, 0,
             -R(37695042795, 15268766246), -R(309121744, 1061227803),
             -R(12992083, 490766935), R(6005943493, 2108947869),
             R(393006217, 1396673457), R(123872331, 1001029789)}, //
            {-R(1028468189, 846180014), 0, 0, R(8478235783, 508512852),
             R(1311729495, 1432422823), -R(10304129995, 1701304382),
             -R(48777925059, 3047939560), R(15336726248, 1032824649),
             -R(45442868181, 3398467696), R(3065993473, 597172653)}, //
            {R(185892177, 718116043), 0, 0, -R(3185094517, 667107341),
             -R(477755414, 1098053517), -R(703635378, 230739211),
             R(5731566787, 1027545527), R(5232866602, 850066563),
             -R(4093664535, 808688257), R(3962137247, 1805957418),
             R(65686358, 487910083)}, //
            {R(403863854, 491063109), 0, 0, -R(5068492393, 434740067),
             -R(411421997, 543043805), R(652783627, 914296604),
             R(11173962825, 925320556), -R(13158990841, 6184727034),
             R(3936647629, 1978049680), -R(160528059, 685178525),
             R(248638103, 1413531060), 0}, //
        },
        {R(14005451, 335480064), 0, 0, 0, 0, -R(59238493, 1068277825),
         R(181606767, 758867731), R(561292985, 797845732),
         -R(1041891430, 1371343529), R(760417239, 1151165299),
         R(118820643, 751138087), -R(528747749, 2220607170), R(1, 4)}};

    const size_t nsteps = get<0>(tableau).size();
    assert(int(nsteps) == rk_method(method).nstages);
    {
      for (size_t step = 0; step < nsteps; ++step)
        assert(get<0>(tableau).at(step).size() == step);
      assert(get<1>(tableau).size() == nsteps);
      const auto &bs = get<1>(tableau);
      T x = 0;
      for (const auto &b : bs)
        x += b;
      assert(fabs(x - 1) <= 10 * numeric_limits<T>::epsilon());
    }

    const auto old = copy_state(var, make_valid_all());

    const auto &bs = get<1>(tableau);

    vector<statecomp_t> ks;
    ks.reserve(nsteps);
    for (size_t step = 0; step < nsteps; ++step) {
      if (step > 0) {
        const auto &as = get<0>(tableau).at(step);
        T c = 0;
        for (const auto &a : as)
          c += a;

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
      }

      calcrhs(step + 1, bs.at(step));
      ks.push_back(copy_state(rhs, make_valid_int()));
    }

    vector<CCTK_REAL> factors;
    vector<const statecomp_t *> srcs;
    factors.reserve(bs.size() + 1);
    srcs.reserve(bs.size() + 1);
    factors.push_back(1);
    srcs.push_back(&old);
    for (size_t i = 0; i < bs.size(); ++i) {
      if (bs.at(i) != 0) {
        factors.push_back(bs.at(i) * dt);
        srcs.push_back(&ks.at(i));
      }
    }
    calcupdate(nsteps, dt, 0.0, factors, srcs);

  } else if (CCTK_EQUALS(method, "IMEX122") ||
             CCTK_EQUALS(method, "Implicit Euler")) {
    const auto y0 = var.copy(make_valid_int());

    *const_cast<CCTK_REAL *>(&cctkGH->cctk_time) = old_time;
    if (verbose)
      CCTK_VINFO("Calculating RHS #1 at t=%g", double(cctkGH->cctk_time));
    CallScheduleGroup(cctkGH, "ODESolvers_RHS");
    const auto k1 = rhs.copy(make_valid_int());

    *const_cast<CCTK_REAL *>(&cctkGH->cctk_time) = old_time + dt / 2;
    statecomp_t::lincomb(var, 1, make_array(dt / 2), make_array(&rhs),
                         make_valid_int());
    var.check_valid(make_valid_int(),
                    "ODESolvers after defining new state vector");
    mark_invalid(dep_groups);
    CallScheduleGroup(cctkGH, "ODESolvers_PostStep");

    *const_cast<CCTK_REAL *>(&cctkGH->cctk_time) = old_time + dt / 2;
    *const_cast<CCTK_REAL *>(&cctkGH->cctk_delta_time) = dt / 2;
    if (verbose)
      CCTK_VINFO("Taking implicit step #1 at t=%g with dt=%g",
                 double(cctkGH->cctk_time), double(cctkGH->cctk_delta_time));
    CallScheduleGroup(cctkGH, "ODESolvers_ImplicitStep");
    *const_cast<CCTK_REAL *>(&cctkGH->cctk_delta_time) = dt;

    *const_cast<CCTK_REAL *>(&cctkGH->cctk_time) = old_time + dt;
    CallScheduleGroup(cctkGH, "ODESolvers_PostStep");
    const auto y1 = var.copy(make_valid_int());

    statecomp_t kprime2;
    statecomp_t::lincomb(kprime2, 0,
                         make_array(-CCTK_REAL(1), +CCTK_REAL(1), -dt / 2),
                         make_array(&y0, &y1, &k1), make_valid_int());

    *const_cast<CCTK_REAL *>(&cctkGH->cctk_time) = old_time + dt;
    if (verbose)
      CCTK_VINFO("Calculating RHS #2 at t=%g", double(cctkGH->cctk_time));
    CallScheduleGroup(cctkGH, "ODESolvers_RHS");
    const auto k2 = rhs.copy(make_valid_int());

    statecomp_t::lincomb(var, 0, make_array(CCTK_REAL(1), dt, dt),
                         make_array(&y0, &k2, &kprime2), make_valid_int());
    var.check_valid(make_valid_int(),
                    "ODESolvers after defining new state vector");
    mark_invalid(dep_groups);

  } else {
    assert(0);
  }

  {
    static Timer timer_free_temps("ODESolvers::Solve::free_temps");
    Interval interval_free_temps(timer_free_temps);
    statecomp_t::free_tmp_mfabs();
  }

  *const_cast<CCTK_REAL *>(&cctkGH->cctk_time) = saved_time;
}

} // namespace ODESolvers
