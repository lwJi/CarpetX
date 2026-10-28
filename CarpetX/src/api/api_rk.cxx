#include <carpetx_rk.hxx>

#include "api/api.hxx"
#include "hierarchy/driver.hxx"
#include "schedule/schedule.hxx"
#include "comm/subcycling.hxx"
#include "hierarchy/valid.hxx"
#include <carpetx.hxx>

#include <cctk.h>
#include <cctk_Parameters.h>

#if defined _OPENMP
#include <omp.h>
#else
static inline int omp_in_parallel() { return 0; }
#endif

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <source_location>
#include <span>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace CarpetX {

namespace {

const rk_config_t &require_integrator() {
  const std::optional<rk_config_t> &config = driver().rk_config;
  if (!config)
    refuse(std::source_location::current(),
           "no time integrator is registered; call "
           "CarpetX::register_integrator AT WRAGH");
  return *config;
}

const char *dense_output_name(const rk_dense_output_t dense_output) {
  switch (dense_output) {
  case rk_dense_output_t::none:
    return "none";
  case rk_dense_output_t::ssprk3:
    return "SSPRK3";
  case rk_dense_output_t::rk4:
    return "RK4";
  }
  CCTK_VERROR("CarpetX: unknown rk_dense_output_t value %d", int(dense_output));
}

void require_grid_function_group(const char *const role, const gi_t gi) {
  if (gi.index() < 0 || gi.index() >= CCTK_NumGroups())
    CCTK_VERROR("CarpetX::register_integrator: %s group index %d does not "
                "exist: there are %d groups",
                role, gi.index(), CCTK_NumGroups());
  if (CCTK_GroupTypeI(gi.index()) != CCTK_GF)
    CCTK_VERROR("CarpetX::register_integrator: %s group %s is not a grid "
                "function group",
                role, CCTK_FullGroupName(gi.index()));
}

void check_weights(const std::vector<CCTK_REAL> &b) {
  using std::fabs, std::isfinite;
  CCTK_REAL sum = 0;
  bool all_zero = true;
  for (std::size_t s = 0; s < b.size(); ++s) {
    if (!isfinite(b[s]))
      CCTK_VERROR("CarpetX::register_integrator: the weight b of RK stage %d "
                  "is %g; every registered weight is finite",
                  int(s) + 1, double(b[s]));
    sum += b[s];
    all_zero = all_zero && b[s] == 0;
  }
  const CCTK_REAL tolerance =
      4 * b.size() * std::numeric_limits<CCTK_REAL>::epsilon();
  if (!all_zero && !(fabs(sum - 1) <= tolerance))
    CCTK_VERROR("CarpetX::register_integrator: the %d RK stage weights b sum "
                "to %.17g, not 1; the final update is y1 = y0 + dt * sum_s "
                "b_s f(Y_s), so an explicit method's weights sum to 1, and a "
                "method that never updates the state registers all-zero "
                "weights",
                int(b.size()), double(sum));
}

std::string describe_weights(const std::vector<CCTK_REAL> &b) {
  if (b.empty())
    return "no weights (nothing feeds the flux registers)";
  std::ostringstream buf;
  buf << "weights b = [";
  for (std::size_t s = 0; s < b.size(); ++s)
    buf << (s == 0 ? "" : ", ") << b[s];
  buf << "]";
  return buf.str();
}

void check_pair(const rk_config_t &config, const std::size_t index) {
  const rk_pair_t &pair = config.integrated.at(index);
  require_grid_function_group("integrated", pair.state);
  require_grid_function_group("right-hand side", pair.rhs);
  const char *const state = CCTK_FullGroupName(pair.state.index());
  if (pair.rhs == pair.state)
    CCTK_VERROR("CarpetX::register_integrator: group %s is registered as its "
                "own right-hand side",
                state);
  const int state_nvars = CCTK_NumVarsInGroupI(pair.state.index());
  const int rhs_nvars = CCTK_NumVarsInGroupI(pair.rhs.index());
  if (state_nvars != rhs_nvars)
    CCTK_VERROR("CarpetX::register_integrator: group %s has %d variables, "
                "but its right-hand side %s has %d",
                state, state_nvars, CCTK_FullGroupName(pair.rhs.index()),
                rhs_nvars);
  if (!get_group_checkpoint_flag(pair.state))
    CCTK_VERROR("CarpetX::register_integrator: integrated group %s is tagged "
                "checkpoint=\"no\"; a checkpoint would omit its state (and, "
                "under subcycling, its RK source bands), so recovery could "
                "not restore it; remove the tag",
                state);
  if (!get_group_evolve_flag(pair.state))
    CCTK_VERROR("CarpetX::register_integrator: integrated group %s is tagged "
                "evolve=\"no\"; a regrid would not reliably refill its "
                "state; remove the tag",
                state);
  for (std::size_t other = 0; other < index; ++other) {
    const rk_pair_t &earlier = config.integrated.at(other);
    if (earlier.state == pair.state)
      CCTK_VERROR("CarpetX::register_integrator: group %s is registered as "
                  "integrated state twice",
                  state);
    if (earlier.rhs == pair.rhs)
      CCTK_VERROR("CarpetX::register_integrator: group %s is registered as "
                  "the right-hand side of both %s and %s",
                  CCTK_FullGroupName(pair.rhs.index()),
                  CCTK_FullGroupName(earlier.state.index()), state);
  }
  for (const rk_pair_t &other : config.integrated)
    if (other.rhs == pair.state)
      CCTK_VERROR("CarpetX::register_integrator: group %s is registered both "
                  "as integrated state and as the right-hand side of %s",
                  state, CCTK_FullGroupName(other.state.index()));
}

} // namespace

void register_integrator(const rk_config_t &config) {
  DECLARE_CCTK_PARAMETERS;
  GHExt &ext = driver();
  if (ext.rk_config)
    CCTK_VERROR("CarpetX::register_integrator: a time integrator is already "
                "registered (%d RK stages, %d integrated groups); an "
                "integrator registers exactly once, AT WRAGH",
                ext.rk_config->nstages(),
                int(ext.rk_config->integrated.size()));
  if (ext.rk_config_read)
    CCTK_ERROR("CarpetX::register_integrator: the driver has already read its "
               "integrator record; schedule the registration AT WRAGH");
  if (config.dense_output != rk_dense_output_t::none &&
      config.nstages() != rk_dense_output_stages(config.dense_output))
    CCTK_VERROR("CarpetX::register_integrator: the %s dense output belongs to "
                "a method with %d RK stages, but the integrator registered %d",
                dense_output_name(config.dense_output),
                rk_dense_output_stages(config.dense_output), config.nstages());
  if (use_subcycling && config.dense_output == rk_dense_output_t::none)
    CCTK_ERROR("CarpetX::register_integrator: CarpetX::use_subcycling = yes "
               "fills the refinement boundary by dense output, but the "
               "integrator registered dense_output = none; choose a method "
               "whose dense output CarpetX implements (SSPRK3, RK4)");
  check_weights(config.b);
  for (std::size_t index = 0; index < config.integrated.size(); ++index)
    check_pair(config, index);
  if (do_reflux && max_num_levels > 1 && !config.feeds_flux_registers())
    for (const rk_pair_t &pair : config.integrated)
      if (get_group_fluxes(pair.state))
        CCTK_VERROR("CarpetX::register_integrator: CarpetX::do_reflux = yes "
                    "with CarpetX::max_num_levels = %d needs an integrator "
                    "that feeds the flux registers at every stage (registered "
                    "weights b), but it registered no weights and integrates "
                    "group %s, which carries a fluxes= tag; register an "
                    "explicit method or set CarpetX::do_reflux = no",
                    int(max_num_levels), CCTK_FullGroupName(pair.state.index()));
  ext.rk_config = config;
  CCTK_VINFO("Registered the time integrator: %d RK stages, dense output %s, "
             "%d integrated group%s, %s",
             config.nstages(), dense_output_name(config.dense_output),
             int(config.integrated.size()),
             config.integrated.size() == 1 ? "" : "s",
             describe_weights(config.b).c_str());
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

struct integrated_groups_t {
  std::vector<gi_t> states, rhss;
};

struct rk_level_t {
  level_t level;
  integrated_groups_t groups;
};

namespace {

using GroupData = GHExt::PatchData::LevelData::GroupData;

constexpr tl_t rk_tl{0};

void require_serial_context() {
  if (omp_in_parallel())
    CCTK_VERROR("%s must run outside an OpenMP parallel region: it opens its "
                "own MFIter and OpenMP regions and warms AMReX's FPinfo cache",
                current_operation());
}

std::string describe(const window_t &window) {
  const auto levels = window.level_bounds();
  const auto patches = window.patch_bounds();
  std::ostringstream buf;
  buf << "patches [" << patches.min << ", " << patches.max
      << "), refinement levels [" << levels.min << ", " << levels.max << ")";
  return buf.str();
}

integrated_groups_t integrated_groups_with_storage(const level_t level,
                                                   const rk_config_t &config) {
  const GHExt &ext = driver();
  ext.level(level);
  integrated_groups_t groups;
  for (const rk_pair_t &pair : config.integrated) {
    if (!ext.record(pair.state).has_storage())
      continue;
    if (!ext.record(pair.rhs).has_storage())
      refuse(std::source_location::current(),
             "integrated group %s has storage on patch %d, refinement level "
             "%d, but its right-hand side %s has none",
             CCTK_FullGroupName(pair.state.index()), level.patch().index(),
             level.index(), CCTK_FullGroupName(pair.rhs.index()));
    groups.states.push_back(pair.state);
    groups.rhss.push_back(pair.rhs);
  }
  return groups;
}

std::vector<rk_level_t> integrated_levels(const window_t window,
                                          const rk_config_t &config) {
  std::vector<rk_level_t> levels;
  loop_coarse_to_fine(
      window, [&](const GHExt::PatchData::LevelData &leveldata) {
        levels.push_back({leveldata.level, integrated_groups_with_storage(
                                               leveldata.level, config)});
      });
  return levels;
}

bool aligned_with_parent(const level_t level) {
  return driver().level(level).iteration ==
         driver().level(level.parent()).iteration;
}

void fill_refinement_boundary(const cGH *const cctkGH, const level_t level,
                              const std::span<const gi_t> states,
                              const int stage, const int nstages,
                              const std::string &event) {
  assert(driver().use_subcycling);
  assert(level.index() > 0);
  assert(stage >= 1 && stage <= nstages);
  const GHExt::PatchData::LevelData &leveldata = driver().level(level);
  const GHExt::PatchData::LevelData &parentdata =
      driver().level(level.parent());
  const CCTK_REAL substep_start =
      aligned_with_parent(level) ? rk_substep_ratio : 0;
  const CCTK_REAL dtc =
      cctkGH->cctk_delta_time * CCTK_REAL(parentdata.delta_iteration);
  assert(std::isfinite(dtc) && dtc > 0);
  const bool end_of_substep = stage == nstages;
  const window_t window = window_t::one_level(level);
  std::vector<write_scope_t> scopes;
  for (const gi_t gi : states) {
    const GroupData &groupdata = leveldata.group(gi);
    const valid_t outer = groupdata.all_faces_have_symmetries_or_boundaries()
                              ? make_valid_outer()
                              : valid_t();
    scopes.push_back(open_write(window, {.gi = gi,
                                         .tl = rk_tl,
                                         .fills = make_valid_ghosts() | outer,
                                         .keeps = make_valid_all(),
                                         .event = event}));
  }
  FillRKBoundary(
      level, states, rk_tl, end_of_substep ? 1 : stage + 1,
      end_of_substep ? substep_start + rk_substep_ratio : substep_start, dtc);
}

} // namespace

struct rk_step_t::impl {
  window_t window;
  int nstages;
  std::vector<CCTK_REAL> b;
  std::vector<gi_t> states;
  std::vector<rk_level_t> levels;
  int rhs_count = 0;
  int update_count = 0;
};

rk_step_t open_rk_step(const window_t window) {
  const operation_scope_t op("CarpetX::open_rk_step");
  const rk_config_t &config = require_integrator();
  require_serial_context();
  auto step =
      std::unique_ptr<rk_step_t::impl>(new rk_step_t::impl{.window = window});
  step->nstages = config.nstages();
  step->b = config.b;
  for (const rk_pair_t &pair : config.integrated)
    step->states.push_back(pair.state);
  step->levels = integrated_levels(window, config);
  for (const rk_level_t &entry : step->levels) {
    for (const gi_t gi : entry.groups.states)
      require_valid(group_t{entry.level, gi}, rk_tl, make_valid_int(),
                    "CarpetX::open_rk_step: the RK step starts from the "
                    "state's current time level");
    StoreRKOldState(entry.level, entry.groups.states, rk_tl);
  }
  synchronize();
  return rk_step_t(std::move(step));
}

rk_step_t::rk_step_t(std::unique_ptr<impl> p) noexcept : p(std::move(p)) {}

rk_step_t::rk_step_t(rk_step_t &&) noexcept = default;

void rk_step_t::rhs_computed() {
  const operation_scope_t op("CarpetX::rk_step_t::rhs_computed");
  if (!p)
    CCTK_ERROR("CarpetX::rk_step_t::rhs_computed called on a moved-from RK "
               "step");
  if (p->rhs_count > p->update_count)
    CCTK_VERROR("CarpetX::rk_step_t::rhs_computed on %s, RK stage %d of %d: "
                "the stage already has its right-hand side; expected "
                "state_updated",
                describe(p->window).c_str(), p->rhs_count, p->nstages);
  if (p->rhs_count == p->nstages)
    CCTK_VERROR("CarpetX::rk_step_t::rhs_computed on %s: all %d stages of "
                "this step have run; destroy the rk_step_t and open the next "
                "step",
                describe(p->window).c_str(), p->nstages);
  const int stage = ++p->rhs_count;
  const CCTK_REAL weight = p->b.at(stage - 1);
  const CCTK_REAL delta_time = driver().get_global_cctkGH()->cctk_delta_time;
  using std::isfinite;
  if (!(isfinite(delta_time) && delta_time != 0))
    CCTK_VERROR("CarpetX::rk_step_t::rhs_computed on %s, RK stage %d of %d: "
                "cctk_delta_time is %g",
                describe(p->window).c_str(), stage, p->nstages,
                double(delta_time));
  const std::string why = "CarpetX::rk_step_t::rhs_computed on " +
                          describe(p->window) + ": RK stage " +
                          std::to_string(stage) + " of " +
                          std::to_string(p->nstages) +
                          " reads the interior of the right-hand side";
  for (const rk_level_t &entry : p->levels)
    for (const gi_t gi : entry.groups.rhss)
      require_valid(group_t{entry.level, gi}, rk_tl, make_valid_int(),
                    why.c_str());
  for (const rk_level_t &entry : p->levels) {
    const CCTK_REAL dt =
        delta_time * CCTK_REAL(driver().level(entry.level).delta_iteration);
    AccumulateFluxes(entry.level, stage, weight * dt);
    StoreRKStage(entry.level, entry.groups.states, entry.groups.rhss, stage);
  }
  synchronize();
}

void rk_step_t::state_updated() {
  const operation_scope_t op("CarpetX::rk_step_t::state_updated");
  if (!p)
    CCTK_ERROR("CarpetX::rk_step_t::state_updated called on a moved-from RK "
               "step");
  if (p->update_count == p->rhs_count)
    CCTK_VERROR("CarpetX::rk_step_t::state_updated on %s, RK stage %d of %d: "
                "expected rhs_computed first",
                describe(p->window).c_str(), p->update_count + 1, p->nstages);
  const int stage = ++p->update_count;
  require_operable(p->window, p->states);
  const cGH *const cctkGH = driver().get_global_cctkGH();
  if (driver().use_subcycling) {
    require_serial_context();
    const int nstages = p->nstages;
    const std::string event =
        "CarpetX::rk_step_t::state_updated after RK stage " +
        std::to_string(stage) + " of " + std::to_string(nstages);
    for (const rk_level_t &entry : p->levels) {
      if (entry.level.index() == 0)
        continue;
      const std::vector<gi_t> &states = entry.groups.states;
      SyncGroupsByDirIGhostOnly(window_t::one_level(entry.level), cctkGH,
                                states, timelevels_t::one(rk_tl));
      fill_refinement_boundary(cctkGH, entry.level, states, stage, nstages,
                               event);
    }
  } else if (stage == p->nstages) {
    Reflux(cctkGH, p->window);
  }
  synchronize();
}

rk_step_t::~rk_step_t() {
  if (!p)
    return;
  if (p->update_count != p->nstages)
    CCTK_VERROR("CarpetX::rk_step_t on %s destroyed after %d of its %d "
                "stages; every stage calls rhs_computed and then "
                "state_updated before the step closes",
                describe(p->window).c_str(), p->update_count, p->nstages);
}

void RecoverRKBoundary(const window_t window, const cGH *const cctkGH) {
  DECLARE_CCTK_PARAMETERS;
  const GHExt &ext = driver();
  if (!ext.use_subcycling)
    return;
  const std::optional<rk_config_t> &integrator = ext.rk();
  if (!integrator || integrator->integrated.empty())
    return;
  const operation_scope_t op("CarpetX::RecoverRKBoundary");

  if (verbose)
    CCTK_VINFO("RecoverRKBoundary on %s: refilling the refinement-boundary "
               "ghosts of the integrated groups (spatial prolongation on "
               "time-aligned levels of a restricted window, dense output from "
               "the level's restored source bands otherwise)",
               describe(window).c_str());

  static Timer timer("RecoverRKBoundary");
  Interval interval(timer);

  const std::vector<rk_level_t> levels = integrated_levels(window, *integrator);
  for (const rk_level_t &entry : levels)
    for (const gi_t gi : entry.groups.states)
      require_valid(group_t{entry.level, gi}, rk_tl, make_valid_int(),
                    "CarpetX::RecoverRKBoundary requires the tl=0 interior of "
                    "the integrated groups to be populated by checkpoint "
                    "recovery");

  std::vector<gi_t> states;
  for (const rk_pair_t &pair : integrator->integrated)
    states.push_back(pair.state);
  std::sort(states.begin(), states.end());
  SyncGroupsByDirIGhostOnly(window, ext.get_global_cctkGH(), states,
                            timelevels_t::one(rk_tl));
  const bool restricted = window_is_restricted(window);
  if (restricted)
    SyncGroupsByDirIProlongateOnlyAligned(window, ext.get_global_cctkGH(),
                                          states, timelevels_t::one(rk_tl));

  for (const rk_level_t &entry : levels) {
    if (entry.level.index() == 0)
      continue;
    const bool aligned = aligned_with_parent(entry.level);
    if (aligned != (entry.level.index() > window.level_bounds().min))
      CCTK_VERROR("CarpetX::RecoverRKBoundary on %s: refinement level %d is "
                  "%s its parent, but recovery traverses each clock group "
                  "separately, so only the group's coarsest level lags its "
                  "parent",
                  describe(window).c_str(), entry.level.index(),
                  aligned ? "aligned with" : "lagging");
    if (restricted && aligned)
      continue;
    fill_refinement_boundary(cctkGH, entry.level, entry.groups.states,
                             integrator->nstages(), integrator->nstages(),
                             "CarpetX::RecoverRKBoundary after recovery");
  }
  synchronize();
}

} // namespace CarpetX
