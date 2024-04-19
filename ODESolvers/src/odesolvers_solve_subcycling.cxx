#include "solve.hxx"

namespace ODESolvers {
using namespace std;

constexpr int rkstages = 4;

extern "C" void ODESolvers_Solve_Subcycling(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTS_ODESolvers_Solve_Subcycling;
  DECLARE_CCTK_PARAMETERS;

  static bool did_output = false;
  if (verbose || !did_output)
    CCTK_VINFO("Integrator is %s", method);
  did_output = true;

  const CCTK_REAL dt = cctk_delta_time;
  const int tl = 0;

  statecomp_t var, rhs;
  array<statecomp_t, rkstages> ks;
  std::vector<int> var_groups, rhs_groups, dep_groups;
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
      const auto &ks_gi = get_group_ks<int, rkstages>(groupdata.groupindex);
      if (rhs_gi >= 0) {
        assert(rhs_gi != groupdata.groupindex);
        auto &rhs_groupdata = *leveldata.groupdata.at(rhs_gi);
        assert(rhs_groupdata.numvars == groupdata.numvars);

        var.groupdatas.push_back(&groupdata);
        var.mfabs.push_back(groupdata.mfab.at(tl).get());
        rhs.groupdatas.push_back(&rhs_groupdata);
        rhs.mfabs.push_back(rhs_groupdata.mfab.at(tl).get());
        for (int i = 0; i < rkstages; i++) {
          auto &ki_groupdata = *leveldata.groupdata.at(ks_gi[i]);
          ks[i].groupdatas.push_back(&ki_groupdata);
          ks[i].mfabs.push_back(ki_groupdata.mfab.at(tl).get());
        }

        if (do_accumulate_nvars) {
          nvars += groupdata.numvars;
          var_groups.push_back(groupdata.groupindex);
          rhs_groups.push_back(rhs_gi);
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

  statecomp_t::init_tmp_mfabs();

  const CCTK_REAL saved_time = cctkGH->cctk_time;
  const CCTK_REAL old_time = cctkGH->cctk_time - dt;

  if (CCTK_EQUALS(method, "constant")) {

    // y1 = y0

    // do nothing

  } else if (CCTK_EQUALS(method, "RK4")) {

    // k1 = f(y0)
    // k2 = f(y0 + h/2 k1)
    // k3 = f(y0 + h/2 k2)
    // k4 = f(y0 + h k3)
    // y1 = y0 + h/6 k1 + h/3 k2 + h/3 k3 + h/6 k4

    const auto old = var.copy(make_valid_int /*all*/ ());
    *const_cast<CCTK_REAL *>(&cctkGH->cctk_time) = old_time;

    // Step 1
    if (verbose)
      CCTK_VINFO("Calculating RHS #1 at t=%g", double(cctkGH->cctk_time));
    CallScheduleGroup(cctkGH, "ODESolvers_RHS");
    rhs.check_valid(make_valid_int(),
                    "ODESolvers after calling ODESolvers_RHS");
    // const auto k1 = rhs.copy(make_valid_int());
    const auto kaccum = rhs.copy(make_valid_int());

    // Step 2

    // Add scaled RHS to state vector
    statecomp_t::lincomb(var, 1, make_array(dt / 2), make_array(&rhs),
                         make_valid_int());
    var.check_valid(make_valid_int(),
                    "ODESolvers after defining new state vector");
    mark_invalid(dep_groups);
    *const_cast<CCTK_REAL *>(&cctkGH->cctk_time) = old_time + dt / 2;
    CallScheduleGroup(cctkGH, "ODESolvers_PostStep");

    if (verbose)
      CCTK_VINFO("Calculating RHS #2 at t=%g", double(cctkGH->cctk_time));
    CallScheduleGroup(cctkGH, "ODESolvers_RHS");
    rhs.check_valid(make_valid_int(),
                    "ODESolvers after calling ODESolvers_RHS");
    // const auto k2 = rhs.copy(make_valid_int());
    statecomp_t::lincomb(kaccum, 1, make_array(CCTK_REAL(2)), make_array(&rhs),
                         make_valid_int());

    // Step 3

    // Add scaled RHS to state vector
    statecomp_t::lincomb(var, 0, make_array(CCTK_REAL(1), dt / 2),
                         make_array(&old, &rhs), make_valid_int());
    var.check_valid(make_valid_int(),
                    "ODESolvers after defining new state vector");
    mark_invalid(dep_groups);
    *const_cast<CCTK_REAL *>(&cctkGH->cctk_time) = old_time + dt / 2;
    CallScheduleGroup(cctkGH, "ODESolvers_PostStep");

    if (verbose)
      CCTK_VINFO("Calculating RHS #3 at t=%g", double(cctkGH->cctk_time));
    CallScheduleGroup(cctkGH, "ODESolvers_RHS");
    rhs.check_valid(make_valid_int(),
                    "ODESolvers after calling ODESolvers_RHS");
    // const auto k3 = rhs.copy(make_valid_int());
    statecomp_t::lincomb(kaccum, 1, make_array(CCTK_REAL(2)), make_array(&rhs),
                         make_valid_int());

    // Step 4

    // Add scaled RHS to state vector
    statecomp_t::lincomb(var, 0, make_array(CCTK_REAL(1), dt),
                         make_array(&old, &rhs), make_valid_int());
    var.check_valid(make_valid_int(),
                    "ODESolvers after defining new state vector");
    mark_invalid(dep_groups);
    *const_cast<CCTK_REAL *>(&cctkGH->cctk_time) = old_time + dt;
    CallScheduleGroup(cctkGH, "ODESolvers_PostStep");

    if (verbose)
      CCTK_VINFO("Calculating RHS #4 at t=%g", double(cctkGH->cctk_time));
    CallScheduleGroup(cctkGH, "ODESolvers_RHS");
    rhs.check_valid(make_valid_int(),
                    "ODESolvers after calling ODESolvers_RHS");
    const auto &k4 = rhs;

    // Calculate new state vector
    statecomp_t::lincomb(var, 0, make_array(CCTK_REAL(1), dt / 6, dt / 6),
                         make_array(&old, &kaccum, &k4), make_valid_int());
    var.check_valid(make_valid_int(),
                    "ODESolvers after defining new state vector");
    mark_invalid(dep_groups);

  } else {
    assert(0);
  }

  statecomp_t::free_tmp_mfabs();

  // Reset current time
  *const_cast<CCTK_REAL *>(&cctkGH->cctk_time) = saved_time;
  // Apply last boundary conditions
  CallScheduleGroup(cctkGH, "ODESolvers_PostStep");
  if (verbose)
    CCTK_VINFO("Calculated new state at t=%g", double(cctkGH->cctk_time));

  // TODO: Update time here, and not during time level cycling in the driver
}

} // namespace ODESolvers
