#include "solve.hxx"
#include <subcycling.hxx>

namespace ODESolvers {
using namespace std;

constexpr int rkstages = 4;

std::vector<int> OldGroups, VarGroups, RhsGroups;
array<std::vector<int>, rkstages> KsGroups;

extern "C" void ODESolvers_Solve_Subcycling_Setup(CCTK_ARGUMENTS) {
  const auto &patchdata0 = ghext->patchdata.at(0);
  const auto &leveldata0 = patchdata0.leveldata.at(0);
  bool do_accumulate_nvars = true;
  for (const auto &groupdataptr : leveldata0.groupdata) {
    if (groupdataptr == nullptr)
      continue;

    auto &groupdata = *groupdataptr;
    const int rhs_gi = get_group_rhs(groupdata.groupindex);
    const int old_gi = get_group_old(groupdata.groupindex);
    const auto &ks_gi = get_group_ks<int, rkstages>(groupdata.groupindex);
    if (rhs_gi >= 0) {
      assert(rhs_gi != groupdata.groupindex);
      if (do_accumulate_nvars) {
        VarGroups.push_back(groupdata.groupindex);
        RhsGroups.push_back(rhs_gi);
        OldGroups.push_back(old_gi);
        for (int i = 0; i < rkstages; i++) {
          KsGroups[i].push_back(ks_gi[i]);
        }
      }
    }
  }
  do_accumulate_nvars = false;
}

extern "C" void ODESolvers_Solve_CalcYfFromKcs1(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTS_ODESolvers_Solve_CalcYfFromKcs1;
  const CCTK_REAL xsi = (cctk_iteration % 2) ? 0.0 : 0.5;
  const CCTK_REAL dt = CCTK_DELTA_TIME;
  Subcycling::CalcYfFromKcs<rkstages>(CCTK_PASS_CTOC, VarGroups, OldGroups,
                                      KsGroups, dt * 2, xsi, 1);
}

extern "C" void ODESolvers_Solve_CalcYfFromKcs2(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTS_ODESolvers_Solve_CalcYfFromKcs2;
  const CCTK_REAL xsi = (cctk_iteration % 2) ? 0.0 : 0.5;
  const CCTK_REAL dt = CCTK_DELTA_TIME;
  Subcycling::CalcYfFromKcs<rkstages>(CCTK_PASS_CTOC, VarGroups, OldGroups,
                                      KsGroups, dt * 2, xsi, 2);
}

extern "C" void ODESolvers_Solve_CalcYfFromKcs3(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTS_ODESolvers_Solve_CalcYfFromKcs3;
  const CCTK_REAL xsi = (cctk_iteration % 2) ? 0.0 : 0.5;
  const CCTK_REAL dt = CCTK_DELTA_TIME;
  Subcycling::CalcYfFromKcs<rkstages>(CCTK_PASS_CTOC, VarGroups, OldGroups,
                                      KsGroups, dt * 2, xsi, 3);
}

extern "C" void ODESolvers_Solve_CalcYfFromKcs4(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTS_ODESolvers_Solve_CalcYfFromKcs4;
  const CCTK_REAL xsi = (cctk_iteration % 2) ? 0.0 : 0.5;
  const CCTK_REAL dt = CCTK_DELTA_TIME;
  Subcycling::CalcYfFromKcs<rkstages>(CCTK_PASS_CTOC, VarGroups, OldGroups,
                                      KsGroups, dt * 2, xsi, 4);
}

extern "C" void ODESolvers_Solve_SetK1(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTS_ODESolvers_Solve_SetK1;
  Subcycling::SetK<rkstages>(CCTK_PASS_CTOC, KsGroups, RhsGroups, 1);
}
extern "C" void ODESolvers_Solve_SetK2(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTS_ODESolvers_Solve_SetK2;
  Subcycling::SetK<rkstages>(CCTK_PASS_CTOC, KsGroups, RhsGroups, 2);
}
extern "C" void ODESolvers_Solve_SetK3(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTS_ODESolvers_Solve_SetK3;
  Subcycling::SetK<rkstages>(CCTK_PASS_CTOC, KsGroups, RhsGroups, 3);
}
extern "C" void ODESolvers_Solve_SetK4(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTS_ODESolvers_Solve_SetK4;
  Subcycling::SetK<rkstages>(CCTK_PASS_CTOC, KsGroups, RhsGroups, 4);
}

extern "C" void ODESolvers_Solve_Subcycling(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTS_ODESolvers_Solve_Subcycling;
  DECLARE_CCTK_PARAMETERS;

  static bool did_output = false;
  if (verbose || !did_output)
    CCTK_VINFO("Integrator is %s", method);
  did_output = true;

  static Timer timer("ODESolvers::Solve");
  Interval interval(timer);

  const CCTK_REAL dt = CCTK_DELTA_TIME;
  const int tl = 0;

  static Timer timer_setup("ODESolvers::Solve::setup");
  std::optional<Interval> interval_setup(timer_setup);

  statecomp_t var, rhs, old;
  array<statecomp_t, rkstages> ks;
  std::vector<int> var_groups, rhs_groups, dep_groups, old_groups;
  array<std::vector<int>, rkstages> ks_groups;
  int nvars = 0;
  bool do_accumulate_nvars = true;
  assert(CarpetX::active_levels);
  CarpetX::active_levels->loop_serially([&](const auto &leveldata) {
    for (const auto &groupdataptr : leveldata.groupdata) {
      // TODO: add support for evolving grid scalars
      if (groupdataptr == nullptr)
        continue;

      auto &groupdata = *groupdataptr;
      const int rhs_gi = get_group_rhs(groupdata.groupindex);
      const int old_gi = get_group_old(groupdata.groupindex);
      const auto &ks_gi = get_group_ks<int, rkstages>(groupdata.groupindex);
      if (rhs_gi >= 0) {
        assert(rhs_gi != groupdata.groupindex);
        auto &rhs_groupdata = *leveldata.groupdata.at(rhs_gi);
        assert(rhs_groupdata.numvars == groupdata.numvars);
        var.groupdatas.push_back(&groupdata);
        var.mfabs.push_back(groupdata.mfab.at(tl).get());
        rhs.groupdatas.push_back(&rhs_groupdata);
        rhs.mfabs.push_back(rhs_groupdata.mfab.at(tl).get());
        auto &old_groupdata = *leveldata.groupdata.at(old_gi);
        old.groupdatas.push_back(&old_groupdata);
        old.mfabs.push_back(old_groupdata.mfab.at(tl).get());
        for (int i = 0; i < rkstages; i++) {
          auto &ki_groupdata = *leveldata.groupdata.at(ks_gi[i]);
          ks[i].groupdatas.push_back(&ki_groupdata);
          ks[i].mfabs.push_back(ki_groupdata.mfab.at(tl).get());
        }

        if (do_accumulate_nvars) {
          nvars += groupdata.numvars;
          var_groups.push_back(groupdata.groupindex);
          rhs_groups.push_back(rhs_gi);
          old_groups.push_back(old_gi);
          for (int i = 0; i < rkstages; i++) {
            ks_groups[i].push_back(ks_gi[i]);
          }
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

  {
    std::sort(old_groups.begin(), old_groups.end());
    const auto last = std::unique(old_groups.begin(), old_groups.end());
    assert(last == old_groups.end());
  }

  for (int i = 0; i < rkstages; i++) {
    std::sort(ks_groups[i].begin(), ks_groups[i].end());
    const auto last = std::unique(ks_groups[i].begin(), ks_groups[i].end());
    assert(last == ks_groups[i].end());
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

  const auto calcrhs = [&](const int n) {
    Interval interval_rhs(timer_rhs);
    if (verbose)
      CCTK_VINFO("Calculating RHS #%d at t=%g", n, double(cctkGH->cctk_time));
    CallScheduleGroup(cctkGH, "ODESolvers_RHS");
    rhs.check_valid(make_valid_int(),
                    "ODESolvers after calling ODESolvers_RHS");
  };
  // t = t_0 + c
  // var = a_0 * var + \Sum_i a_i * var_i
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

    // y1 = y0

    // do nothing

  } else if (CCTK_EQUALS(method, "RK4")) {

    // k1 = f(y0)
    // k2 = f(y0 + h/2 k1)
    // k3 = f(y0 + h/2 k2)
    // k4 = f(y0 + h k3)
    // y1 = y0 + h/6 k1 + h/3 k2 + h/3 k3 + h/6 k4

    // Initialize Ks
    if (cctk_iteration == 1) {
      for (int s = 0; s < rkstages; s++) {
        statecomp_t::lincomb(ks[s], 0, make_array(CCTK_REAL(1)),
                             make_array(&rhs), make_valid_int());
      }
    }

    // Set OldState:
    {
      Interval interval_lincomb(timer_lincomb);
      statecomp_t::lincomb(old, 0, make_array(CCTK_REAL(1)), make_array(&var),
                           make_valid_int());
    }

    // Sync OldState and Ks:
    CallScheduleGroup(cctkGH, "ODESolvers_SyncKsOld");

    // k1 = f(Y1)
    CallScheduleGroup(cctkGH,
                      "ODESolvers_CalcYfFromKcs1"); // refinement boundary only
    calcrhs(1);
    CallScheduleGroup(cctkGH, "ODESolvers_SetK1"); // interior only
    calcupdate(1, dt / 2, 1.0, reals<1>{dt / 2}, states<1>{&rhs});

    // k2 = f(Y2)
    CallScheduleGroup(cctkGH,
                      "ODESolvers_CalcYfFromKcs2"); // refinement boundary only
    calcrhs(2);
    CallScheduleGroup(cctkGH, "ODESolvers_SetK2"); // interior only
    calcupdate(2, dt / 2, 0.0, reals<2>{1.0, dt / 2}, states<2>{&old, &rhs});

    // k3 = f(Y3)
    CallScheduleGroup(cctkGH,
                      "ODESolvers_CalcYfFromKcs3"); // refinement boundary only
    calcrhs(3);
    CallScheduleGroup(cctkGH, "ODESolvers_SetK3"); // interior only
    calcupdate(3, dt, 0.0, reals<2>{1.0, dt}, states<2>{&old, &rhs});

    // k4 = f(Y4)
    CallScheduleGroup(cctkGH,
                      "ODESolvers_CalcYfFromKcs4"); // refinement boundary only
    calcrhs(4);
    CallScheduleGroup(cctkGH, "ODESolvers_SetK4"); // interior only
    //{
    //  Interval interval_lincomb(timer_lincomb);
    //  statecomp_t::lincomb(ks[3], 0.0, reals<1>{1.0}, states<1>{&rhs},
    //                       make_valid_int());
    //}

    // y1 = y0 + h/6 k1 + h/3 k2 + h/3 k3 + h/6 k4
    calcupdate(4, dt, 0.0, reals<5>{1.0, dt / 6, dt / 3, dt / 3, dt / 6},
               states<5>{&old, &ks[0], &ks[1], &ks[2], &ks[3]});

  } else {
    assert(0);
  }

  {
    static Timer timer_free_temps("ODESolvers::Solve::free_temps");
    Interval interval_free_temps(timer_free_temps);
    statecomp_t::free_tmp_mfabs();
  }

  // Reset current time
  *const_cast<CCTK_REAL *>(&cctkGH->cctk_time) = saved_time;

  // TODO: Update time here, and not during time level cycling in the driver
}

} // namespace ODESolvers
