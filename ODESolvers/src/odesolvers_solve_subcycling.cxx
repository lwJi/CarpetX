#include "solve.hxx"
#include <subcycling.hxx>

namespace ODESolvers {

constexpr int rkstages = 4;

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
  std::array<statecomp_t, rkstages> ks;
  std::vector<int> var_groups, rhs_groups, dep_groups, old_groups;
  std::array<std::vector<int>, rkstages> ks_groups;
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
      if (sync_interprocess_ghost_only_on_update) {
        SyncGroupsByDirIGhostOnly(cctkGH, var_groups.size(), var_groups.data(),
                                  nullptr);
        CallScheduleGroup(cctkGH, "ODESolvers_PostSubStep");
      } else {
        CallScheduleGroup(cctkGH, "ODESolvers_PostStep");
      }
      if (verbose)
        CCTK_VINFO("Calculated new state #%d at t=%g", n,
                   double(cctkGH->cctk_time));
    }
  };
  // calculate Ys from ks and old on the mesh refinement boundary
  const auto calcys_rmbnd = [&](const int stage) {
    active_levels->loop_parallel([&](int patch, int level, int index,
                                     int component, const cGH *local_cctkGH) {
      if (level == 0)
        return;
      const auto &patchdata = ghext->patchdata.at(patch);
      const CCTK_REAL xsi = (patchdata.leveldata.at(level).iteration ==
                             patchdata.leveldata.at(level - 1).iteration)
                                ? 0.5
                                : 0.0;
      update_cctkGH(const_cast<cGH *>(local_cctkGH), cctkGH);
      Subcycling::CalcYfFromKcs<rkstages>(const_cast<cGH *>(local_cctkGH),
                                          var_groups, old_groups, ks_groups,
                                          dt * 2, xsi, stage);
    });
    synchronize();
    var.set_valid(make_valid_all());
  };
  // calculate yn from ks and old on the mesh refinement boundary
  const auto calcyn_rmbnd = [&]() {
    active_levels->loop_parallel([&](int patch, int level, int index,
                                     int component, const cGH *local_cctkGH) {
      if (level == 0)
        return;
      const auto &patchdata = ghext->patchdata.at(patch);
      const CCTK_REAL xsi = (patchdata.leveldata.at(level).iteration ==
                             patchdata.leveldata.at(level - 1).iteration)
                                ? 1.0
                                : 0.5;
      update_cctkGH(const_cast<cGH *>(local_cctkGH), cctkGH);
      Subcycling::CalcYfFromKcs<rkstages>(const_cast<cGH *>(local_cctkGH),
                                          var_groups, old_groups, ks_groups,
                                          dt * 2, xsi, 1);
    });
    synchronize();
    var.set_valid(make_valid_all());
  };
  // set ks in the interior which will be used for prolongation later
  const auto setks = [&](const int stage) {
    active_levels->loop_parallel([&](int patch, int level, int index,
                                     int component, const cGH *local_cctkGH) {
      update_cctkGH(const_cast<cGH *>(local_cctkGH), cctkGH);
      Subcycling::SetK<rkstages>(const_cast<cGH *>(local_cctkGH), ks_groups,
                                 rhs_groups, stage);
    });
    synchronize();
  };
  // set old in the interior which will be used for prolongation later
  const auto setold = [&]() {
    active_levels->loop_parallel([&](int patch, int level, int index,
                                     int component, const cGH *local_cctkGH) {
      update_cctkGH(const_cast<cGH *>(local_cctkGH), cctkGH);
      Subcycling::SetOld(const_cast<cGH *>(local_cctkGH), old_groups,
                         var_groups);
    });
    synchronize();
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

    // Sync OldState and Ks: prolongate old and ks from parent level which are
    // set in previous steps.
    // CallScheduleGroup(cctkGH, "ODESolvers_SyncKsOld");
    if (old_groups.size() > 0) {
      old.set_valid(make_valid_int()); // mark interior valid to work around
                                       // poison mechanism
      SyncGroupsByDirIProlongateOnly(cctkGH, old_groups.size(),
                                     old_groups.data(), nullptr);
      for (int s = 0; s < rkstages; ++s) {
        ks[s].set_valid(make_valid_int()); // mark interior valid to work around
                                           // poison mechanism
        SyncGroupsByDirIProlongateOnly(cctkGH, ks_groups[s].size(),
                                       ks_groups[s].data(), nullptr);
      }
    }

    // Grid functions used to fill the refinement boundary substate.
    // Temporary variables cannot be used for old values here because
    // they need to be accessed in subsequent CallScheduleGroup functions,
    // which do not yet support access to temporary variables.
    setold();

    // k1 = f(Y1)
    calcys_rmbnd(1); // refinement boundary only
    calcrhs(1);
    setks(1); // interior only
    calcupdate(1, dt / 2, 1.0, reals<1>{dt / 2}, states<1>{&rhs});

    // k2 = f(Y2)
    calcys_rmbnd(2); // refinement boundary only
    calcrhs(2);
    setks(2); // interior only
    calcupdate(2, dt / 2, 0.0, reals<2>{1.0, dt / 2}, states<2>{&old, &rhs});

    // k3 = f(Y3)
    calcys_rmbnd(3); // refinement boundary only
    calcrhs(3);
    setks(3); // interior only
    calcupdate(3, dt, 0.0, reals<2>{1.0, dt}, states<2>{&old, &rhs});

    // k4 = f(Y4)
    calcys_rmbnd(4); // refinement boundary only
    calcrhs(4);
    setks(4); // interior only
    calcupdate(4, dt, 0.0, reals<5>{1.0, dt / 6, dt / 3, dt / 3, dt / 6},
               states<5>{&old, &ks[0], &ks[1], &ks[2], &ks[3]});

    if (sync_interprocess_ghost_only_on_update) {
      calcyn_rmbnd();
    }

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
