#include "solve.hxx"

// Driver primitives for the state-based refinement-boundary fill
// (StoreRKOldState, StoreRKStage, FillRKBoundary).
// TODO: Don't include files from other thorns; create a proper interface.
#include "../../CarpetX/src/subcycling.hxx"

#include <AMReX_MultiFab.H>

#include <array>

namespace ODESolvers {

namespace {

struct solve_setup_t {
  statecomp_t var, rhs;
  std::vector<int> var_groups, rhs_groups, dep_groups;
  int nvars = 0;
};

// Where within the parent's step the dense-output polynomial is evaluated for
// a fine level's refinement-boundary fill. This is the only place the
// evaluation-point convention lives.
struct dense_output_point_t {
  CCTK_REAL xsi;
  int stage;
};

// Because Evolve advances the fine counter before CCTK_EVOL, a fine level is
// aligned with its parent exactly during its second substep (it lands on the
// parent's time), hence xsi = 1/2; during the first substep it is behind,
// hence xsi = 0. At the virtual end-of-substep stage (num_rk_stages + 1) the
// polynomial is evaluated at xsi + 1/2 with the pure-U (stage 1) combination,
// which is what a normal RK step would need at its next stage 1.
dense_output_point_t
dense_output_point(const CarpetX::GHExt::PatchData::LevelData &leveldata,
                   const int stage) {
  assert(leveldata.level > 0);
  const auto &patchdata = CarpetX::ghext->patchdata.at(leveldata.patch);
  const auto &prev_leveldata = patchdata.leveldata.at(leveldata.level - 1);
  const int virtual_end = CarpetX::ghext->num_rk_stages + 1;
  assert(stage >= 1 && stage <= virtual_end);
  const CCTK_REAL xsi =
      (leveldata.iteration == prev_leveldata.iteration) ? 0.5 : 0.0;
  if (stage == virtual_end)
    return {xsi + 0.5, 1};
  return {xsi, stage};
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
  auto &var_groups = setup.var_groups;
  auto &rhs_groups = setup.rhs_groups;
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

  const CCTK_REAL saved_time = cctkGH->cctk_time;
  const CCTK_REAL old_time = cctkGH->cctk_time - dt;

  static Timer timer_lincomb("ODESolvers::Solve::lincomb");
  static Timer timer_rhs("ODESolvers::Solve::rhs");
  static Timer timer_poststep("ODESolvers::Solve::poststep");

  // Effective weight b_s of each RK stage in the final update,
  //   y1 = y0 + dt * sum_s b_s f(Y_s),
  // handed to the driver's flux register so that after a full step it holds
  // exactly the flux combination the state received. One row per method the
  // subcycling solver implements below; a method without a row is a hard
  // error, so that adding a method cannot silently produce a
  // non-conservative reflux.
  const std::array<CCTK_REAL, CarpetX::max_num_rk_stages> stage_weights =
      [&]() -> std::array<CCTK_REAL, CarpetX::max_num_rk_stages> {
    if (CCTK_EQUALS(method, "constant"))
      return {0, 0, 0, 0};
    if (CCTK_EQUALS(method, "RK4"))
      return {1.0 / 6, 1.0 / 3, 1.0 / 3, 1.0 / 6};
    if (CCTK_EQUALS(method, "SSPRK3"))
      return {1.0 / 6, 1.0 / 6, 2.0 / 3, 0};
    CCTK_VERROR("ODESolvers::method = \"%s\" is not supported by the "
                "subcycling solver (no flux-register stage weights)",
                method);
  }();

  const auto calcrhs = [&](const int n) {
    Interval interval_rhs(timer_rhs);
    if (verbose)
      CCTK_VINFO("Calculating RHS #%d at t=%g", n, double(cctkGH->cctk_time));
    CallScheduleGroup(cctkGH, "ODESolvers_RHS");
    rhs.check_valid(make_valid_int(),
                    "ODESolvers after calling ODESolvers_RHS");
    // Feed the driver's flux registers with this stage's flux, weighted by
    // the stage's effective weight and this batch's step, now: the flux
    // groups hold exactly the flux the update below consumes, and calcupdate
    // marks them invalid afterwards (they are dependents of the state).
    active_levels->loop_coarse_to_fine([&](const auto &restrict leveldata) {
      CarpetX::AccumulateFluxes(leveldata.patch, leveldata.level, n,
                                stage_weights.at(n - 1) * dt);
    });
    synchronize();
  };
  // t = t_0 + c
  // var = a_0 * var + \Sum_i a_i * var_i
  const auto calcupdate = [&](const int n, const CCTK_REAL c,
                              const CCTK_REAL a0, const auto &as,
                              const auto &vars) {
    {
      Interval interval_lincomb(timer_lincomb);
      if (verbose)
        CCTK_VINFO("Calculated new state #%d at t=%g", n,
                   double(cctkGH->cctk_time));
      statecomp_t::lincomb(var, a0, as, vars, make_valid_int());
      var.check_valid(make_valid_int(),
                      "ODESolvers after defining new state vector");
      mark_invalid(dep_groups);
    }
    {
      Interval interval_poststep(timer_poststep);
      *const_cast<CCTK_REAL *>(&cctkGH->cctk_time) = old_time + c;
    }
  };
  // calling ODESolvers_PostStep Group
  const auto calcpoststep = [&]() {
    CallScheduleGroup(cctkGH, "ODESolvers_PostStep");
  };
  // Fill the refinement-boundary ghosts of var(tl=0) on every fine level for
  // the given stage: the driver evaluates the dense-output polynomial on the
  // level's own source bands (the parent's old state + k-stages) at
  // (stage0, xsi) and prolongates that single coarse state in space.
  // dtc = dt*2 is the parent's step under 2:1 time refinement.
  const auto calcys_rmbnd = [&](const int stage) {
    if (verbose)
      CCTK_VINFO(
          "Fill refinement boundary ghost zones using Ys for stage #%d at t=%g",
          stage, double(cctkGH->cctk_time));

    active_levels->loop_coarse_to_fine([&](auto &leveldata) {
      if (leveldata.level == 0)
        return;
      const auto [xsi, stage0] = dense_output_point(leveldata, stage);
      CarpetX::FillRKBoundary(leveldata.patch, leveldata.level, var_groups,
                              /*tl=*/0, stage0, xsi, dt * 2);
    });
    synchronize();
    var.set_valid(make_valid_all());
  };
  // Store the interior RHS of this stage of each level into the k-stage source
  // band of its child level, which owns the bands (levels with children only),
  // to be combined into the children's refinement-boundary fills by
  // calcys_rmbnd.
  const auto setks = [&](const int stage) {
    if (verbose)
      CCTK_VINFO(
          "Set interior Ks for stage #%d at t=%g, to be prolongated later",
          stage, double(cctkGH->cctk_time));
    active_levels->loop_coarse_to_fine([&](const auto &restrict leveldata) {
      CarpetX::StoreRKStage(leveldata.patch, leveldata.level, var_groups,
                            rhs_groups, stage);
    });
    synchronize();
  };
  // Capture u(t_n) = var(tl=0) of each level into the old_source_band of its
  // child level, which owns the bands (interior only, levels with children
  // only), once per step before the RK stages overwrite var. This is also
  // where the child's RK buffers are (lazily) allocated: that reads the
  // next-finer level, so it must run once all levels exist, and it warms an
  // AMReX cache and opens its own MFIter/OpenMP region, so it must run
  // single-threaded (loop_serially).
  const auto store_old = [&]() {
    active_levels->loop_serially([&](const auto &restrict leveldata) {
      CarpetX::StoreRKOldState(leveldata.patch, leveldata.level, var_groups,
                               /*tl=*/0);
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

    // Scratch copy of u(t_n) = var(tl=0), the RK4 interior anchor y0. At one
    // timelevel var(tl=0) holds the previous step's result, so no init copy is
    // needed (mirrors the non-subcycling solver).
    const auto old = var.copy(make_valid_all());

    // Capture u(t_n) into the old source bands (and allocate the bands) before
    // the RK stages overwrite var.
    if (var_groups.size() > 0)
      store_old();

    // k1 = f(Y1)
    calcrhs(1);
    setks(1); // interior only
    const auto kaccum = rhs.copy(make_valid_int());
    calcupdate(1, dt / 2, 1.0, reals<1>{dt / 2}, states<1>{&rhs});
    calcys_rmbnd(2); // refinement boundary only
    calcpoststep();

    // k2 = f(Y2)
    calcrhs(2);
    setks(2); // interior only
    statecomp_t::lincomb(kaccum, 1.0, reals<1>{2.0}, states<1>{&rhs},
                         make_valid_int());
    calcupdate(2, dt / 2, 0.0, reals<2>{1.0, dt / 2}, states<2>{&old, &rhs});
    calcys_rmbnd(3); // refinement boundary only
    calcpoststep();

    // k3 = f(Y3)
    calcrhs(3);
    setks(3); // interior only
    statecomp_t::lincomb(kaccum, 1.0, reals<1>{2.0}, states<1>{&rhs},
                         make_valid_int());
    calcupdate(3, dt, 0.0, reals<2>{1.0, dt}, states<2>{&old, &rhs});
    calcys_rmbnd(4); // refinement boundary only
    calcpoststep();

    // k4 = f(Y4)
    calcrhs(4);
    setks(4); // interior only
    calcupdate(4, dt, 0.0, reals<3>{1.0, dt / 6, dt / 6},
               states<3>{&old, &kaccum, &rhs});
    calcys_rmbnd(5); // refinement boundary only
    calcpoststep();

    // No calcys_rmbnd(1) here: refinement-boundary ghosts are kept aligned by
    // subcycling-aware POSTRESTRICT SYNCs. The post-recovery case is handled
    // by ODESolvers_Solve_Subcycling_Recovery at CCTK_CPINITIAL.

  } else if (CCTK_EQUALS(method, "SSPRK3")) {

    // k1 = f(y0)
    // k2 = f(y0 + h k1)
    // k3 = f(y0 + h/4 k1 + h/4 k2)
    // y1 = y0 + h/6 k1 + h/6 k2 + 2/3 h k3

    assert(ghext->num_rk_stages == 3);

    // Scratch copy of u(t_n) = var(tl=0), the SSPRK3 interior anchor y0.
    const auto old = var.copy(make_valid_all());

    // Capture u(t_n) into the old source bands (and allocate the bands) before
    // the RK stages overwrite var.
    if (var_groups.size() > 0)
      store_old();

    // k1 = f(Y1)
    calcrhs(1);
    setks(1); // interior only
    const auto k1 = rhs.copy(make_valid_int());
    calcupdate(1, dt, 1.0, reals<1>{dt}, states<1>{&rhs}); // var = y0 + dt*k1
    calcys_rmbnd(2); // refinement boundary only
    calcpoststep();

    // k2 = f(Y2)
    calcrhs(2);
    setks(2); // interior only
    const auto k2 = rhs.copy(make_valid_int());
    calcupdate(2, dt / 2, 0.0, reals<3>{1.0, dt / 4, dt / 4},
               states<3>{&old, &k1, &k2});
    calcys_rmbnd(3); // refinement boundary only
    calcpoststep();

    // k3 = f(Y3)
    calcrhs(3);
    setks(3); // interior only
    calcupdate(3, dt, 0.0, reals<4>{1.0, dt / 6, dt / 6, 2 * dt / 3},
               states<4>{&old, &k1, &k2, &rhs});
    calcys_rmbnd(4); // virtual end-of-step (num_rk_stages + 1)
    calcpoststep();

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

extern "C" void ODESolvers_Solve_Subcycling_Recovery(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTS_ODESolvers_Solve_Subcycling_Recovery;
  DECLARE_CCTK_PARAMETERS;

  // Skip on fresh initialization; cctk_iteration > 0 only on recovery.
  if (cctk_iteration <= 0)
    return;

  if (verbose)
    CCTK_VINFO("Subcycling recovery: refilling refinement-boundary ghosts "
               "(spatial prolongation on time-aligned levels, dense output "
               "from the level's restored source bands otherwise)");

  static Timer timer("ODESolvers::Solve_Subcycling_Recovery");
  Interval interval(timer);

  const CCTK_REAL dt = CCTK_DELTA_TIME;

  auto setup = collect_solve_setup();
  auto &var = setup.var;
  auto &var_groups = setup.var_groups;
  if (setup.nvars == 0)
    return;

  // Refill each recovered fine level's refinement-boundary (cf) ghosts.
  // Time-aligned levels use spatial tl=0 prolongation. A level that is behind
  // its parent is mid-cycle: its restored source bands hold the parent's
  // in-progress coarse step, and the same driver fill the uninterrupted run
  // made at the previous fine substep's virtual end-of-step reconstructs the
  // cf-ghosts it last wrote. The checkpoint reader has already refused a
  // mid-cycle checkpoint that lacks those bands.
  if (var_groups.size() > 0) {
    var.check_valid(make_valid_int(),
                    "ODESolvers_Solve_Subcycling_Recovery requires the tl=0 "
                    "interior to be populated by checkpoint recovery");

    // Spatial prolongation fills every fine level's cf-ghosts; the
    // unsynchronized levels below are then overwritten with dense output.
    SyncGroupsByDirIProlongateOnly(cctkGH, var_groups.size(), var_groups.data(),
                                   nullptr, /*tl=*/0);

    active_levels->loop_coarse_to_fine([&](auto &restrict leveldata) {
      const int level = leveldata.level;
      if (level == 0)
        return;
      const auto &patchdata = ghext->patchdata.at(leveldata.patch);
      const auto &prev_leveldata = patchdata.leveldata.at(level - 1);
      // Time-aligned with the parent: spatial prolongation above is correct.
      if (leveldata.iteration == prev_leveldata.iteration)
        return;
      // Mirror the previous fine substep's calcys_rmbnd at the virtual
      // end-of-step: base offset 0.0 plus the +0.5 give xsi = 0.5, stage0 = 1,
      // dtc = dt*2 (the parent's step).
      CarpetX::FillRKBoundary(leveldata.patch, level, var_groups, /*tl=*/0,
                              /*stage=*/1, /*xsi=*/0.5, dt * 2);
    });
    synchronize();
    var.set_valid(make_valid_all());
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

} // namespace ODESolvers
