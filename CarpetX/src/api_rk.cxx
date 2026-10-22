#include <carpetx_rk.hxx>

#include "api.hxx"
#include "driver.hxx"
#include "schedule.hxx"
#include "subcycling.hxx"
#include "timer.hxx"

#include <cctk.h>
#include <cctk_Parameters.h>

#include <algorithm>
#include <cstddef>
#include <optional>

namespace CarpetX {

namespace {

const rk_config_t &require_integrator(const char *const what) {
  const std::optional<rk_config_t> &config = driver().rk_config;
  if (!config)
    CCTK_VERROR("%s: no time integrator is registered; call "
                "CarpetX::register_integrator AT WRAGH",
                what);
  return *config;
}

void require_grid_function_group(const char *const role, const int gi) {
  if (gi < 0 || gi >= CCTK_NumGroups())
    CCTK_VERROR("CarpetX::register_integrator: %s group index %d does not "
                "exist: there are %d groups",
                role, gi, CCTK_NumGroups());
  if (CCTK_GroupTypeI(gi) != CCTK_GF)
    CCTK_VERROR("CarpetX::register_integrator: %s group %s is not a grid "
                "function group",
                role, CCTK_FullGroupName(gi));
}

void check_pair(const rk_config_t &config, const std::size_t index) {
  const rk_pair_t &pair = config.integrated.at(index);
  require_grid_function_group("integrated", pair.state_gi);
  require_grid_function_group("right-hand side", pair.rhs_gi);
  const char *const state = CCTK_FullGroupName(pair.state_gi);
  if (pair.rhs_gi == pair.state_gi)
    CCTK_VERROR("CarpetX::register_integrator: group %s is registered as its "
                "own right-hand side",
                state);
  const int state_nvars = CCTK_NumVarsInGroupI(pair.state_gi);
  const int rhs_nvars = CCTK_NumVarsInGroupI(pair.rhs_gi);
  if (state_nvars != rhs_nvars)
    CCTK_VERROR("CarpetX::register_integrator: group %s has %d variables, "
                "but its right-hand side %s has %d",
                state, state_nvars, CCTK_FullGroupName(pair.rhs_gi), rhs_nvars);
  if (!get_group_checkpoint_flag(pair.state_gi))
    CCTK_VERROR("CarpetX::register_integrator: integrated group %s is tagged "
                "checkpoint=\"no\"; a checkpoint would omit its state (and, "
                "under subcycling, its RK source bands), so recovery could "
                "not restore it; remove the tag",
                state);
  if (!get_group_evolve_flag(pair.state_gi))
    CCTK_VERROR("CarpetX::register_integrator: integrated group %s is tagged "
                "evolve=\"no\"; a regrid would not reliably refill its "
                "state; remove the tag",
                state);
  for (std::size_t other = 0; other < index; ++other) {
    const rk_pair_t &earlier = config.integrated.at(other);
    if (earlier.state_gi == pair.state_gi)
      CCTK_VERROR("CarpetX::register_integrator: group %s is registered as "
                  "integrated state twice",
                  state);
    if (earlier.rhs_gi == pair.rhs_gi)
      CCTK_VERROR("CarpetX::register_integrator: group %s is registered as "
                  "the right-hand side of both %s and %s",
                  CCTK_FullGroupName(pair.rhs_gi),
                  CCTK_FullGroupName(earlier.state_gi), state);
  }
  for (const rk_pair_t &other : config.integrated)
    if (other.rhs_gi == pair.state_gi)
      CCTK_VERROR("CarpetX::register_integrator: group %s is registered both "
                  "as integrated state and as the right-hand side of %s",
                  state, CCTK_FullGroupName(other.state_gi));
}

} // namespace

void register_integrator(const rk_config_t &config) {
  DECLARE_CCTK_PARAMETERS;
  GHExt &ext = driver();
  if (ext.rk_config)
    CCTK_VERROR("CarpetX::register_integrator: a time integrator is already "
                "registered (%d RK stages, %d integrated groups); an "
                "integrator registers exactly once, AT WRAGH",
                ext.rk_config->nstages, int(ext.rk_config->integrated.size()));
  if (ext.rk_config_read)
    CCTK_ERROR("CarpetX::register_integrator: the driver has already read its "
               "integrator record; schedule the registration AT WRAGH");
  if (config.nstages < 1)
    CCTK_VERROR("CarpetX::register_integrator: an integrator has at least one "
                "RK stage, but %d were registered",
                config.nstages);
  if (use_subcycling && config.nstages != 3 && config.nstages != 4)
    CCTK_VERROR("CarpetX::register_integrator: CarpetX::use_subcycling = yes "
                "supports 3 or 4 RK stages (SSPRK3, RK4), the stage counts "
                "the dense output of the refinement-boundary fill implements, "
                "but the integrator registered %d",
                config.nstages);
  if (use_subcycling && !config.subcycling_ok)
    CCTK_ERROR("CarpetX::register_integrator: CarpetX::use_subcycling = yes, "
               "but the integrator registered subcycling_ok = false; choose a "
               "method the subcycling solver implements");
  for (std::size_t index = 0; index < config.integrated.size(); ++index)
    check_pair(config, index);
  if (do_reflux && max_num_levels > 1 && !config.reflux_ok)
    for (const rk_pair_t &pair : config.integrated)
      if (get_group_fluxes(pair.state_gi)[0] >= 0)
        CCTK_VERROR("CarpetX::register_integrator: CarpetX::do_reflux = yes "
                    "with CarpetX::max_num_levels = %d needs an integrator "
                    "that feeds the flux registers at every stage "
                    "(reflux_ok), but it registered reflux_ok = false and "
                    "integrates group %s, which carries a fluxes= tag; "
                    "register an explicit method or set CarpetX::do_reflux = "
                    "no",
                    int(max_num_levels), CCTK_FullGroupName(pair.state_gi));
  ext.rk_config = config;
  CCTK_VINFO("Registered the time integrator: %d RK stages, %d integrated "
             "group%s, subcycling_ok = %s, reflux_ok = %s",
             config.nstages, int(config.integrated.size()),
             config.integrated.size() == 1 ? "" : "s",
             config.subcycling_ok ? "yes" : "no",
             config.reflux_ok ? "yes" : "no");
}

void run_schedule_group_at(const cGH *const cctkGH_, const char *const group,
                           const CCTK_REAL time, const CCTK_REAL delta_time) {
  if (!cctkGH_)
    CCTK_VERROR("CarpetX::run_schedule_group_at(\"%s\"): the cGH is null",
                group);
  if (!group)
    CCTK_ERROR("CarpetX::run_schedule_group_at: the schedule group name is "
               "null");
  static Timer timer("CarpetX::run_schedule_group_at");
  Interval interval(timer);
  cGH *const cctkGH = const_cast<cGH *>(cctkGH_);
  const CCTK_REAL saved_time = cctkGH->cctk_time;
  const CCTK_REAL saved_delta_time = cctkGH->cctk_delta_time;
  cctkGH->cctk_time = time;
  cctkGH->cctk_delta_time = delta_time;
  update_cctkGHs(cctkGH);
  const int ierr = CCTK_ScheduleTraverse(group, cctkGH, CallFunction);
  cctkGH->cctk_time = saved_time;
  cctkGH->cctk_delta_time = saved_delta_time;
  update_cctkGHs(cctkGH);
  if (ierr)
    CCTK_VERROR("CarpetX::run_schedule_group_at: traversing schedule group "
                "\"%s\" at time %g failed with error %d",
                group, double(time), ierr);
}

void accumulate_fluxes(const level_t level, const int stage,
                       const CCTK_REAL weight) {
  const rk_config_t &config = require_integrator("CarpetX::accumulate_fluxes");
  resolve(level);
  if (stage < 1 || stage > config.nstages)
    CCTK_VERROR("CarpetX::accumulate_fluxes: RK stage %d is not one of the "
                "registered integrator's stages 1..%d",
                stage, config.nstages);
  AccumulateFluxes(level.patch, level.level, stage, weight);
}

void reflux(const window_t window) {
  const rk_config_t &config = require_integrator("CarpetX::reflux");
  std::vector<int> states;
  states.reserve(config.integrated.size());
  for (const rk_pair_t &pair : config.integrated)
    states.push_back(pair.state_gi);
  require_operable("CarpetX::reflux", window, states);
  const window_scope scope(window);
  Reflux(driver().get_global_cctkGH(), window.min_level, window.max_level);
}

bool window_is_restricted(const window_t window) {
  to_active_levels(window);
  if (window.min_level >= window.max_level)
    return false;
  return window_is_restricted(window.min_level, window.max_level);
}

} // namespace CarpetX
