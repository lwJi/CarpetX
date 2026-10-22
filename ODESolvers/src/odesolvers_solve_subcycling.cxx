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

bool aligned_with_parent(const CarpetX::level_t level) {
  return CarpetX::iteration(level) ==
         CarpetX::iteration(CarpetX::parent(level));
}

CCTK_REAL parent_step(const cGH *cctkGH, const CarpetX::level_t level) {
  return cctkGH->cctk_delta_time *
         CCTK_REAL(CarpetX::delta_iteration(CarpetX::parent(level)));
}

CCTK_REAL substep_start(const CarpetX::level_t level) {
  return aligned_with_parent(level) ? 0.5 : 0.0;
}

// Collect evolved groups into statecomp_t bundles. The old-state anchor is now
// a scratch copy of var(tl=0) made by the solver (no extra timelevel); the RK
// k-stages live as coarse-fine bands on the child level's GroupData. Operates
// on CarpetX::active_levels; no cGH needed.
solve_setup_t collect_solve_setup() {
  solve_setup_t s;
  s.var.timelevel = 0;
  s.rhs.timelevel = 0;
  bool do_accumulate_nvars = true;
  assert(CarpetX::active_levels);
  CarpetX::active_levels->loop_serially([&](const auto &leveldata) {
    for (const auto &groupdataptr : leveldata.groupdata) {
      // TODO: add support for evolving grid scalars
      if (groupdataptr == nullptr)
        continue;

      auto &groupdata = *groupdataptr;
      const int rhs_gi = get_group_rhs(groupdata.groupindex);
      if (rhs_gi >= 0) {
        assert(rhs_gi != groupdata.groupindex);
        auto &rhs_groupdata = *leveldata.groupdata.at(rhs_gi);
        assert(rhs_groupdata.numvars == groupdata.numvars);
        s.var.groupdatas.push_back(&groupdata);
        s.var.mfabs.push_back(groupdata.mfab.at(0).get());
        s.rhs.groupdatas.push_back(&rhs_groupdata);
        s.rhs.mfabs.push_back(rhs_groupdata.mfab.at(0).get());

        if (do_accumulate_nvars) {
          s.nvars += groupdata.numvars;
          s.var_groups.push_back(groupdata.groupindex);
          s.rhs_groups.push_back(rhs_gi);
          const auto &dependents = get_group_dependents(groupdata.groupindex);
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

  static Timer timer("ODESolvers::Solve");
  Interval interval(timer);

  const CCTK_REAL dt = CCTK_DELTA_TIME;

  static Timer timer_setup("ODESolvers::Solve::setup");
  std::optional<Interval> interval_setup(timer_setup);

  auto setup = collect_solve_setup();
  auto &var = setup.var;
  auto &rhs = setup.rhs;
  auto &dep_groups = setup.dep_groups;
  const int nvars = setup.nvars;
  if (verbose)
    CCTK_VINFO("  Integrating %d variables", nvars);
  if (nvars == 0)
    CCTK_VWARN(CCTK_WARN_ALERT, "Integrating %d variables", nvars);

  interval_setup.reset();

  {
    static Timer timer_alloc_temps("ODESolvers::Solve::alloc_temps");
    Interval interval_alloc_temps(timer_alloc_temps);
    statecomp_t::init_tmp_mfabs();
  }

  const CCTK_REAL old_time = cctkGH->cctk_time - dt;
  CCTK_REAL t = old_time;

  static Timer timer_lincomb("ODESolvers::Solve::lincomb");
  static Timer timer_rhs("ODESolvers::Solve::rhs");
  static Timer timer_poststep("ODESolvers::Solve::poststep");

  const rk_method_t &rk = rk_method(method);
  assert(rk.subcycling_ok);

  const CarpetX::window_t window = CarpetX::active_window();
  std::vector<CarpetX::level_t> levels;
  std::vector<CarpetX::rk_step_t> steps;
  std::vector<CarpetX::rk_stage_t> stages;

  const auto open_steps = [&]() {
    window.loop_coarse_to_fine([&](const CarpetX::level_t level) {
      levels.push_back(level);
      steps.push_back(CarpetX::open_rk_step(level));
    });
    synchronize();
  };
  const auto calcrhs = [&](const int n) {
    Interval interval_rhs(timer_rhs);
    if (verbose)
      CCTK_VINFO("Calculating RHS #%d at t=%g", n, double(t));
    CarpetX::run_schedule_group_at(cctkGH, "ODESolvers_RHS", t,
                                   cctkGH->cctk_delta_time);
    rhs.check_valid(make_valid_int(),
                    "ODESolvers after calling ODESolvers_RHS");
    assert(stages.empty());
    stages.reserve(steps.size());
    for (CarpetX::rk_step_t &step : steps)
      stages.push_back(step.stage(n));
    for (CarpetX::rk_stage_t &stage : stages)
      stage.rhs_computed(rk.b.at(n - 1) * dt);
    synchronize();
  };
  const auto calcupdate = [&](const int n, const CCTK_REAL c,
                              const CCTK_REAL a0, const auto &as,
                              const auto &vars) {
    Interval interval_lincomb(timer_lincomb);
    if (verbose)
      CCTK_VINFO("Calculated new state #%d at t=%g", n, double(t));
    statecomp_t::lincomb(var, a0, as, vars, make_valid_int());
    var.check_valid(make_valid_int(),
                    "ODESolvers after defining new state vector");
    mark_invalid(dep_groups);
    t = old_time + c;
  };
  const auto calcpoststep = [&]() {
    Interval interval_poststep(timer_poststep);
    CarpetX::run_schedule_group_at(cctkGH, "ODESolvers_PostStep", t,
                                   cctkGH->cctk_delta_time);
  };
  const auto calcys_rmbnd = [&]() {
    if (verbose)
      CCTK_VINFO("Fill refinement boundary ghost zones using Ys at t=%g",
                 double(t));
    assert(stages.size() == levels.size());
    for (std::size_t i = 0; i < stages.size(); ++i) {
      const CarpetX::level_t level = levels.at(i);
      if (level.level == 0)
        continue;
      assert(parent_step(cctkGH, level) == dt * 2);
      stages.at(i).fill_next_boundary(substep_start(level),
                                      parent_step(cctkGH, level));
    }
    stages.clear();
    synchronize();
  };

  static_assert(
      rk_methods_detail::subcycling_ok_rows_are({"constant", "RK4", "SSPRK3"}),
      "rk_methods: the subcycling_ok rows must be the methods "
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

    // Scratch copy of u(t_n) = var(tl=0), the RK4 interior anchor y0. At one
    // timelevel var(tl=0) holds the previous step's result, so no init copy is
    // needed (mirrors the non-subcycling solver).
    const auto old = var.copy(make_valid_all());

    open_steps();

    calcrhs(1);
    const auto kaccum = rhs.copy(make_valid_int());
    calcupdate(1, dt / 2, 1.0, reals<1>{dt / 2}, states<1>{&rhs});
    calcys_rmbnd();
    calcpoststep();

    calcrhs(2);
    statecomp_t::lincomb(kaccum, 1.0, reals<1>{2.0}, states<1>{&rhs},
                         make_valid_int());
    calcupdate(2, dt / 2, 0.0, reals<2>{1.0, dt / 2}, states<2>{&old, &rhs});
    calcys_rmbnd();
    calcpoststep();

    calcrhs(3);
    statecomp_t::lincomb(kaccum, 1.0, reals<1>{2.0}, states<1>{&rhs},
                         make_valid_int());
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

    // Scratch copy of u(t_n) = var(tl=0), the SSPRK3 interior anchor y0.
    const auto old = var.copy(make_valid_all());

    open_steps();

    calcrhs(1);
    const auto k1 = rhs.copy(make_valid_int());
    calcupdate(1, dt, 1.0, reals<1>{dt}, states<1>{&rhs}); // var = y0 + dt*k1
    calcys_rmbnd();
    calcpoststep();

    calcrhs(2);
    const auto k2 = rhs.copy(make_valid_int());
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
    // Unreachable: rk.subcycling_ok is asserted above, and the table marks
    // only the methods implemented here as subcycling_ok
    CCTK_VERROR("ODESolvers::method = \"%s\" is marked subcycling_ok in "
                "rk_methods.hxx but has no branch in the subcycling solver",
                method);
  }

  {
    static Timer timer_free_temps("ODESolvers::Solve::free_temps");
    Interval interval_free_temps(timer_free_temps);
    statecomp_t::free_tmp_mfabs();
  }

  // TODO: Update time here, and not during time level cycling in the driver
}

extern "C" void ODESolvers_Solve_Subcycling_Recovery(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTS_ODESolvers_Solve_Subcycling_Recovery;
  DECLARE_CCTK_PARAMETERS;

  if (verbose)
    CCTK_VINFO("Subcycling recovery: refilling refinement-boundary ghosts "
               "(spatial prolongation on time-aligned levels of a restricted "
               "window, dense output from the level's restored source bands "
               "otherwise)");

  static Timer timer("ODESolvers::Solve_Subcycling_Recovery");
  Interval interval(timer);

  auto setup = collect_solve_setup();
  auto &var = setup.var;
  auto &var_groups = setup.var_groups;
  if (setup.nvars == 0)
    return;

  if (var_groups.size() > 0) {
    var.check_valid(make_valid_int(),
                    "ODESolvers_Solve_Subcycling_Recovery requires the tl=0 "
                    "interior to be populated by checkpoint recovery");

    const CarpetX::window_t window = CarpetX::active_window();
    const bool restricted = CarpetX::window_is_restricted(window);

    if (restricted)
      CarpetX::prolongate_aligned(window, var_groups, 0);

    window.loop_coarse_to_fine([&](const CarpetX::level_t level) {
      if (level.level == 0)
        return;
      assert(aligned_with_parent(level) == (level.level > window.min_level) &&
             "recovery traverses POST_RECOVER_VARIABLES once per clock "
             "group, so only the group's coarsest level lags its parent");
      if (restricted && aligned_with_parent(level))
        return;
      CarpetX::rk_recover_boundary(level, substep_start(level),
                                   parent_step(cctkGH, level));
    });
    synchronize();
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
