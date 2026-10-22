#include <carpetx_rk.hxx>

#include "api.hxx"
#include "driver.hxx"
#include "schedule.hxx"
#include "subcycling.hxx"
#include <carpetx.hxx>

#include <cctk.h>
#include <cctk_Parameters.h>

#if defined _OPENMP
#include <omp.h>
#else
static inline int omp_in_parallel() { return 0; }
#endif

#include <algorithm>
#include <cstddef>
#include <functional>
#include <optional>
#include <string>
#include <vector>

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

namespace {

using GroupData = GHExt::PatchData::LevelData::GroupData;

constexpr int rk_tl = 0;

void require_serial_context(const char *const what) {
  if (omp_in_parallel())
    CCTK_VERROR("%s must run outside an OpenMP parallel region: it opens its "
                "own MFIter and OpenMP regions and warms AMReX's FPinfo cache",
                what);
}

void require_subcycling(const char *const what) {
  if (!driver().use_subcycling)
    CCTK_VERROR("%s fills the refinement boundary from the RK source bands, "
                "which exist only with CarpetX::use_subcycling = yes",
                what);
}

void require_refined_level(const char *const what, const level_t level) {
  if (level.level == 0)
    CCTK_VERROR("%s: level 0 of patch %d has no refinement boundary to fill",
                what, level.patch);
}

void require_evaluation_point(const char *const what, const CCTK_REAL xsi,
                              const CCTK_REAL dtc) {
  if (!(xsi >= 0 && xsi + rk_substep_ratio <= 1))
    CCTK_VERROR("%s: the substep starts at xsi = %g of the parent's step, "
                "but a fine substep of length %g must start within [0, %g]",
                what, double(xsi), double(rk_substep_ratio),
                double(1 - rk_substep_ratio));
  if (!(dtc > 0))
    CCTK_VERROR("%s: the parent's time step dtc = %g is not positive", what,
                double(dtc));
}

struct integrated_groups_t {
  std::vector<int> states, rhss;
};

integrated_groups_t integrated_groups_with_storage(const char *const what,
                                                   const level_t level,
                                                   const rk_config_t &config) {
  const GHExt::PatchData::LevelData &leveldata = resolve(level);
  integrated_groups_t groups;
  for (const rk_pair_t &pair : config.integrated) {
    const GroupData *const state = leveldata.groupdata.at(pair.state_gi).get();
    if (!state || state->mfab.empty())
      continue;
    const GroupData *const rhs = leveldata.groupdata.at(pair.rhs_gi).get();
    if (!rhs || rhs->mfab.empty())
      CCTK_VERROR("%s: integrated group %s has storage on patch %d, refinement "
                  "level %d, but its right-hand side %s has none",
                  what, CCTK_FullGroupName(pair.state_gi), level.patch,
                  level.level, CCTK_FullGroupName(pair.rhs_gi));
    groups.states.push_back(pair.state_gi);
    groups.rhss.push_back(pair.rhs_gi);
  }
  return groups;
}

void mark_refinement_boundary_filled(const level_t level,
                                     const std::vector<int> &states,
                                     const std::function<std::string()> &why) {
  GHExt::PatchData::LevelData &leveldata = resolve(level);
  for (const int gi : states) {
    GroupData &groupdata = *leveldata.groupdata.at(gi);
    for (why_valid_t &why_valid : groupdata.valid.at(rk_tl)) {
      why_valid.set_ghosts(true, why);
      if (groupdata.all_faces_have_symmetries_or_boundaries())
        why_valid.set_outer(true, why);
    }
  }
}

} // namespace

struct rk_step_t::impl {
  level_t level;
  int nstages;
  integrated_groups_t groups;
  int next = 1;
  bool stage_open = false;
};

struct rk_stage_t::impl {
  rk_step_t::impl *step;
  int n;
  bool rhs_done = false;
  bool fill_done = false;
};

rk_step_t open_rk_step(const level_t level) {
  const rk_config_t &config = require_integrator("CarpetX::open_rk_step");
  require_serial_context("CarpetX::open_rk_step");
  integrated_groups_t groups =
      integrated_groups_with_storage("CarpetX::open_rk_step", level, config);
  const GHExt::PatchData::LevelData &leveldata = resolve(level);
  for (const int gi : groups.states) {
    const GroupData &groupdata = *leveldata.groupdata.at(gi);
    for (int vi = 0; vi < groupdata.numvars; ++vi)
      error_if_invalid(groupdata, vi, rk_tl, make_valid_int(), []() {
        return "CarpetX::open_rk_step: the RK step starts from the state's "
               "current time level";
      });
  }
  StoreRKOldState(level.patch, level.level, groups.states, rk_tl);
  return rk_step_t(
      new rk_step_t::impl{level, config.nstages, std::move(groups)});
}

rk_step_t::rk_step_t(impl *const p) noexcept : p(p) {}

rk_step_t::rk_step_t(rk_step_t &&other) noexcept : p(other.p) {
  other.p = nullptr;
}

rk_stage_t rk_step_t::stage(const int n) {
  if (!p)
    CCTK_ERROR("CarpetX::rk_step_t::stage called on a moved-from RK step");
  if (p->stage_open)
    CCTK_VERROR("CarpetX::rk_step_t::stage(%d) on patch %d, refinement level "
                "%d: stage %d is still open; destroy its rk_stage_t first",
                n, p->level.patch, p->level.level, p->next - 1);
  if (p->next > p->nstages)
    CCTK_VERROR("CarpetX::rk_step_t::stage(%d) on patch %d, refinement level "
                "%d: all %d stages of this step have run; destroy the "
                "rk_step_t and open the next step",
                n, p->level.patch, p->level.level, p->nstages);
  if (n != p->next)
    CCTK_VERROR("CarpetX::rk_step_t::stage(%d) on patch %d, refinement level "
                "%d: expected stage %d of %d; the stages of a step run in "
                "order, each exactly once",
                n, p->level.patch, p->level.level, p->next, p->nstages);
  p->stage_open = true;
  ++p->next;
  return rk_stage_t(new rk_stage_t::impl{p, n});
}

rk_step_t::~rk_step_t() {
  if (!p)
    return;
  if (p->stage_open)
    CCTK_VERROR("CarpetX::rk_step_t on patch %d, refinement level %d destroyed "
                "while stage %d of %d is still open",
                p->level.patch, p->level.level, p->next - 1, p->nstages);
  if (p->next != p->nstages + 1)
    CCTK_VERROR("CarpetX::rk_step_t on patch %d, refinement level %d destroyed "
                "after %d of its %d stages; every stage of a step runs before "
                "the step closes",
                p->level.patch, p->level.level, p->next - 1, p->nstages);
  delete p;
}

rk_stage_t::rk_stage_t(impl *const p) noexcept : p(p) {}

rk_stage_t::rk_stage_t(rk_stage_t &&other) noexcept : p(other.p) {
  other.p = nullptr;
}

void rk_stage_t::rhs_computed(const CCTK_REAL flux_weight) {
  if (!p)
    CCTK_ERROR("CarpetX::rk_stage_t::rhs_computed called on a moved-from RK "
               "stage");
  const rk_step_t::impl &step = *p->step;
  if (p->rhs_done)
    CCTK_VERROR("CarpetX::rk_stage_t::rhs_computed on patch %d, refinement "
                "level %d, RK stage %d of %d: called twice; expected "
                "fill_next_boundary",
                step.level.patch, step.level.level, p->n, step.nstages);
  AccumulateFluxes(step.level.patch, step.level.level, p->n, flux_weight);
  StoreRKStage(step.level.patch, step.level.level, step.groups.states,
               step.groups.rhss, p->n);
  p->rhs_done = true;
}

void rk_stage_t::fill_next_boundary(const CCTK_REAL xsi, const CCTK_REAL dtc) {
  if (!p)
    CCTK_ERROR("CarpetX::rk_stage_t::fill_next_boundary called on a "
               "moved-from RK stage");
  const rk_step_t::impl &step = *p->step;
  if (!p->rhs_done)
    CCTK_VERROR("CarpetX::rk_stage_t::fill_next_boundary on patch %d, "
                "refinement level %d, RK stage %d of %d: expected "
                "rhs_computed first",
                step.level.patch, step.level.level, p->n, step.nstages);
  if (p->fill_done)
    CCTK_VERROR("CarpetX::rk_stage_t::fill_next_boundary on patch %d, "
                "refinement level %d, RK stage %d of %d: called twice; the "
                "stage is complete, destroy it",
                step.level.patch, step.level.level, p->n, step.nstages);
  require_subcycling("CarpetX::rk_stage_t::fill_next_boundary");
  require_refined_level("CarpetX::rk_stage_t::fill_next_boundary", step.level);
  require_evaluation_point("CarpetX::rk_stage_t::fill_next_boundary", xsi, dtc);
  const bool virtual_end = p->n == step.nstages;
  FillRKBoundary(step.level.patch, step.level.level, step.groups.states, rk_tl,
                 virtual_end ? 1 : p->n + 1,
                 virtual_end ? xsi + rk_substep_ratio : xsi, dtc);
  const int iteration = driver().get_global_cctkGH()->cctk_iteration;
  mark_refinement_boundary_filled(
      step.level, step.groups.states,
      [n = p->n, nstages = step.nstages, iteration]() {
        return "CarpetX::rk_stage_t::fill_next_boundary after RK stage " +
               std::to_string(n) + " of " + std::to_string(nstages) +
               " (iteration " + std::to_string(iteration) + ")";
      });
  p->fill_done = true;
}

rk_stage_t::~rk_stage_t() {
  if (!p)
    return;
  rk_step_t::impl &step = *p->step;
  if (!p->rhs_done)
    CCTK_VERROR("CarpetX::rk_stage_t on patch %d, refinement level %d, RK "
                "stage %d of %d destroyed before rhs_computed",
                step.level.patch, step.level.level, p->n, step.nstages);
  if (!p->fill_done && driver().use_subcycling && step.level.level > 0)
    CCTK_VERROR("CarpetX::rk_stage_t on patch %d, refinement level %d, RK "
                "stage %d of %d destroyed before fill_next_boundary; with "
                "CarpetX::use_subcycling = yes every stage on a refined level "
                "fills its refinement boundary before it closes",
                step.level.patch, step.level.level, p->n, step.nstages);
  step.stage_open = false;
  delete p;
}

void rk_recover_boundary(const level_t level, const CCTK_REAL xsi,
                         const CCTK_REAL dtc) {
  const rk_config_t &config =
      require_integrator("CarpetX::rk_recover_boundary");
  require_serial_context("CarpetX::rk_recover_boundary");
  require_subcycling("CarpetX::rk_recover_boundary");
  require_refined_level("CarpetX::rk_recover_boundary", level);
  require_evaluation_point("CarpetX::rk_recover_boundary", xsi, dtc);
  const integrated_groups_t groups = integrated_groups_with_storage(
      "CarpetX::rk_recover_boundary", level, config);
  FillRKBoundary(level.patch, level.level, groups.states, rk_tl, 1,
                 xsi + rk_substep_ratio, dtc);
  const int iteration = driver().get_global_cctkGH()->cctk_iteration;
  mark_refinement_boundary_filled(level, groups.states, [iteration]() {
    return "CarpetX::rk_recover_boundary (iteration " +
           std::to_string(iteration) + ")";
  });
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
