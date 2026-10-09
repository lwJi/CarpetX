#include "rk_methods.hxx"
#include "solve.hxx"

// Driver primitives for the state-based refinement-boundary fill
// (StoreRKOldState, StoreRKStage, FillRKBoundary, window_is_restricted).
// TODO: Don't include files from other thorns; create a proper interface.
#include "../../CarpetX/src/subcycling.hxx"

#include <AMReX_MultiFab.H>

#include <array>

namespace ODESolvers {

namespace {

using LevelData = CarpetX::GHExt::PatchData::LevelData;

struct solve_setup_t {
  statecomp_t var, rhs;
  std::vector<int> var_groups, rhs_groups, dep_groups;
  int nvars = 0;
};

// The parent level of a refined level
const LevelData &parent_leveldata(const LevelData &leveldata) {
  assert(leveldata.level > 0);
  return CarpetX::ghext->patchdata.at(leveldata.patch)
      .leveldata.at(leveldata.level - 1);
}

// Because Evolve advances the fine counter before CCTK_EVOL, a fine level is
// aligned with its parent (its clock equals the parent's) exactly during its
// second substep, which lands on the parent's time. During its first substep
// it lags its parent.
bool aligned_with_parent(const LevelData &leveldata) {
  return leveldata.iteration == parent_leveldata(leveldata).iteration;
}

// The parent's step, taken from the clocks: cctk_delta_time is the coarse
// step, and the parent's delta_iteration is its fraction of a coarse step.
// Inside an evolution batch (the window is one level L, with cctk_timefac =
// 2^L) this equals CCTK_DELTA_TIME * 2 bit for bit, since both scale
// cctk_delta_time by the same power of two. Unlike CCTK_DELTA_TIME * 2, it
// does not depend on cctk_timefac, so it is also the parent's step in a
// window that spans several levels.
CCTK_REAL parent_step(const cGH *cctkGH, const LevelData &leveldata) {
  return cctkGH->cctk_delta_time *
         CCTK_REAL(parent_leveldata(leveldata).delta_iteration);
}

// Where within the parent's step the dense-output polynomial is evaluated for
// a fine level's refinement-boundary fill. This is the only place the
// evaluation-point convention lives.
struct dense_output_point_t {
  CCTK_REAL xsi;
  int stage;
};

// During the first substep (lagging the parent) the stages are evaluated at
// xsi = 0, during the second (aligned) at xsi = 1/2. At the virtual
// end-of-substep stage (num_rk_stages + 1) the polynomial is evaluated half a
// fine step later, at xsi + 1/2, with the pure-U (stage 1) combination, which
// is what the next substep needs at its stage 1. For an aligned level that is
// the end of the parent's step, xsi = 1, where the polynomial is the parent's
// RK update. The parent's tl=0 is that update after ODESolvers_PostStep, and
// after reflux and restriction once its window is restricted, so where those
// change the state the two differ by more than round-off.
dense_output_point_t dense_output_point(const LevelData &leveldata,
                                        const int stage) {
  const int virtual_end = CarpetX::ghext->num_rk_stages + 1;
  assert(stage >= 1 && stage <= virtual_end);
  const CCTK_REAL xsi = aligned_with_parent(leveldata) ? 0.5 : 0.0;
  if (stage == virtual_end)
    return {xsi + 0.5, 1};
  return {xsi, stage};
}

// Fill the refinement-boundary (coarse-fine) ghosts of var(tl=0) of `groups`
// on a level > 0 for the given stage of its current substep: the driver
// evaluates the dense-output polynomial on the level's own source bands (the
// parent's old state + k-stages) at dense_output_point and prolongates that
// single coarse state in space, with the parent's step as dtc. The bands are
// all it reads, so even at the end of the parent's step it needs neither the
// parent's tl=0 nor the communication of a spatial prolongation from it.
// Evolution (calcys_rmbnd) calls it at every stage, the virtual end-of-step
// included, and recovery (ODESolvers_Solve_Subcycling_Recovery) at the
// virtual end-of-step, so recovery reproduces that fill by construction.
// Validity is left to the caller.
void fill_rk_boundary(const cGH *cctkGH, const LevelData &leveldata,
                      const std::vector<int> &groups, const int stage) {
  const auto [xsi, stage0] = dense_output_point(leveldata, stage);
  CarpetX::FillRKBoundary(leveldata.patch, leveldata.level, groups,
                          /*tl=*/0, stage0, xsi,
                          parent_step(cctkGH, leveldata));
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

  // The method's row in the RK method table (rk_methods.hxx): its stage
  // count and the effective weight b_s of each stage in the final update,
  //   y1 = y0 + dt * sum_s b_s f(Y_s),
  // handed to the driver's flux register so that after a full step it holds
  // exactly the flux combination the state received. ODESolvers_CheckMethod
  // rejected at PARAMCHECK any method this solver does not implement.
  const rk_method_t &rk = rk_method(method);
  assert(rk.subcycling_ok);

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
                                rk.b.at(n - 1) * dt);
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
  // the given stage, the virtual end-of-step (num_rk_stages + 1) included,
  // by dense output from the level's own source bands (fill_rk_boundary).
  const auto calcys_rmbnd = [&](const int stage) {
    if (verbose)
      CCTK_VINFO(
          "Fill refinement boundary ghost zones using Ys for stage #%d at t=%g",
          stage, double(cctkGH->cctk_time));

    active_levels->loop_coarse_to_fine([&](auto &leveldata) {
      if (leveldata.level == 0)
        return;
      // In an evolution batch the window is this one level, so the parent's
      // step is also twice this level's dt
      assert(parent_step(cctkGH, leveldata) == dt * 2);
      fill_rk_boundary(cctkGH, leveldata, var_groups, stage);
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

    // The table row must describe the sequence below
    assert(rk.nstages == 4);

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
    // by ODESolvers_Solve_Subcycling_Recovery at
    // CCTK_POST_RECOVER_VARIABLES.

  } else if (CCTK_EQUALS(method, "SSPRK3")) {

    // k1 = f(y0)
    // k2 = f(y0 + h k1)
    // k3 = f(y0 + h/4 k1 + h/4 k2)
    // y1 = y0 + h/6 k1 + h/6 k2 + 2/3 h k3

    // The table row must describe the sequence below
    assert(rk.nstages == 3);

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

  // Reset current time
  *const_cast<CCTK_REAL *>(&cctkGH->cctk_time) = saved_time;

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

  // Refill each recovered fine level's refinement-boundary (cf) ghosts the
  // way the uninterrupted run last wrote them. Recovery traverses
  // CCTK_POST_RECOVER_VARIABLES once per clock group, and this routine runs
  // first in each traversal, so the window is one clock group, the window in
  // which its levels last ran their end of step. Its levels other than its
  // coarsest are aligned with their parents. Its coarsest level, unless it is
  // level 0, lags its parent: it is mid-cycle. The ghosts were last written
  //  - at the virtual end of the level's last substep (calcys_rmbnd) by dense
  //    output from its source bands, which hold the parent's last step. The
  //    checkpoint restored them; its reader has already refused a mid-cycle
  //    checkpoint that lacks bands still needed here. fill_rk_boundary at the
  //    same point, with the parent's step taken from the clocks, reproduces
  //    these ghosts;
  //  - or, on an aligned level of a window the driver restricted at that end
  //    of step (CarpetX::window_is_restricted: the finest clock group), after
  //    that by spatial prolongation from the restricted parent
  //    (ProlongateRestrictedGFs), for every evolved group, restricted or not.
  //    The parent's tl=0 still holds that state, and
  //    SyncGroupsByDirIProlongateOnlyAligned repeats the prolongation. These
  //    are exactly the levels that need no bands: a time-aligned checkpoint
  //    carries none.
  // The ODESolvers_PostStep SYNC that follows then only exchanges ghosts
  // between boxes of the same level.
  if (var_groups.size() > 0) {
    var.check_valid(make_valid_int(),
                    "ODESolvers_Solve_Subcycling_Recovery requires the tl=0 "
                    "interior to be populated by checkpoint recovery");

    const bool restricted = CarpetX::window_is_restricted(
        active_levels->min_level, active_levels->max_level);

    // Skips level 0 and every level that lags its parent
    if (restricted)
      CarpetX::SyncGroupsByDirIProlongateOnlyAligned(
          cctkGH, var_groups.size(), var_groups.data(), nullptr, /*tl=*/0);

    const int virtual_end = CarpetX::ghext->num_rk_stages + 1;
    active_levels->loop_coarse_to_fine([&](const auto &leveldata) {
      if (leveldata.level == 0)
        return;
      // Prolongated in space above
      if (restricted && aligned_with_parent(leveldata))
        return;
      fill_rk_boundary(cctkGH, leveldata, var_groups, virtual_end);
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

// Reject at PARAMCHECK, instead of at the first CCTK_EVOL, a method the
// configuration cannot run (rk_methods.hxx):
//  - under subcycling, one the subcycling solver does not implement; only
//    the implemented ones have a stage count that fits the driver's band
//    arrays and a b row the flux registers can be fed with;
//  - in a run that may refine (CarpetX::max_num_levels > 1), an implicit
//    (IMEX) method when some group gets a flux register: the driver would
//    allocate the register on every refined level, and only an explicit
//    method hands every stage's flux to it with a known weight
//    (ODESolvers_Solve's calcrhs). A unigrid run owns no register (they
//    live on levels > 0), so an implicit method is fine there even with
//    CarpetX::do_reflux = yes.
// Runs after WRAGH, where ODESolvers_InitConstants publishes
// rk_integrated_group, so CarpetX::group_has_flux_register (the predicate
// that register allocation uses) already gives its final answer.
extern "C" void ODESolvers_CheckMethod(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTS_ODESolvers_CheckMethod;
  DECLARE_CCTK_PARAMETERS;

  const rk_method_t &rk = rk_method(method);

  if (CarpetX::ghext->use_subcycling && !rk.subcycling_ok) {
    std::string supported;
    for (const auto &row : rk_methods)
      if (row.subcycling_ok)
        supported +=
            (supported.empty() ? "\"" : ", \"") + std::string(row.name) + "\"";
    CCTK_VERROR("ODESolvers::method = \"%s\" is not supported by the "
                "subcycling solver (CarpetX::use_subcycling = yes); supported "
                "methods: %s",
                method, supported.c_str());
  }

  // CarpetX::max_num_levels is private to the driver; read it by name
  int max_num_levels_type;
  const void *const max_num_levels_p =
      CCTK_ParameterGet("max_num_levels", "CarpetX", &max_num_levels_type);
  assert(max_num_levels_p);
  assert(max_num_levels_type == PARAMETER_INT);
  const CCTK_INT max_num_levels =
      *static_cast<const CCTK_INT *>(max_num_levels_p);

  if (CarpetX::ghext->do_reflux && !rk.reflux_ok && max_num_levels > 1) {
    for (int gi = 0; gi < CCTK_NumGroups(); ++gi) {
      if (!CarpetX::group_has_flux_register(gi))
        continue;
      CCTK_VERROR("CarpetX::do_reflux with fluxes= tags and "
                  "CarpetX::max_num_levels > 1 requires an explicit "
                  "ODESolvers::method, but \"%s\" is implicit and group "
                  "\"%s\" (integrated by ODESolvers) carries a fluxes= tag. "
                  "Choose an explicit method or set CarpetX::do_reflux = no.",
                  method, CCTK_FullGroupName(gi));
    }
  }
}

} // namespace ODESolvers
