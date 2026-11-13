#include <carpetx_rk.hxx>

#include "api/api.hxx"
#include "hierarchy/driver.hxx"
#include "hierarchy/group_tags.hxx"
#include "hierarchy/integrator.hxx"
#include "writers/writers.hxx"
#include "hierarchy/rk_levels.hxx"
#include "schedule/schedule.hxx"
#include "hierarchy/loops.hxx"
#include "comm/comm.hxx"
#include "comm/reflux.hxx"
#include "comm/subcycling.hxx"
#include "validity/valid.hxx"
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
#include <map>
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

const GHExt::integrator_t &require_integrator() {
  const std::optional<GHExt::integrator_t> &integrator = driver().integrator();
  if (!integrator)
    refuse(std::source_location::current(),
           "no time integrator is registered; call "
           "CarpetX::register_integrator AT WRAGH");
  return *integrator;
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
  const operation_scope_t op("CarpetX::register_integrator");
  const GHExt &ext = driver();
  if (config.dense_output != rk_dense_output_t::none &&
      config.nstages() != rk_dense_output_stages(config.dense_output))
    CCTK_VERROR("CarpetX::register_integrator: the %s dense output belongs to "
                "a method with %d RK stages, but the integrator registered %d",
                dense_output_name(config.dense_output),
                rk_dense_output_stages(config.dense_output), config.nstages());
  if (ext.use_subcycling && config.dense_output == rk_dense_output_t::none)
    CCTK_ERROR("CarpetX::register_integrator: CarpetX::use_subcycling = yes "
               "fills the refinement boundary by dense output, but the "
               "integrator registered dense_output = none; choose a method "
               "whose dense output CarpetX implements (SSPRK3, RK4)");
  check_weights(config.b);
  for (std::size_t index = 0; index < config.integrated.size(); ++index)
    check_pair(config, index);
  if (ext.do_reflux && max_num_levels > 1 && !config.feeds_flux_registers())
    for (const rk_pair_t &pair : config.integrated)
      if (ext.record(pair.state).grid_function().fluxes)
        CCTK_VERROR("CarpetX::register_integrator: CarpetX::do_reflux = yes "
                    "with CarpetX::max_num_levels = %d needs an integrator "
                    "that feeds the flux registers at every stage (registered "
                    "weights b), but it registered no weights and integrates "
                    "group %s, which carries a fluxes= tag; register an "
                    "explicit method or set CarpetX::do_reflux = no",
                    int(max_num_levels),
                    CCTK_FullGroupName(pair.state.index()));
  std::map<gi_t, gi_t> rhs;
  for (const rk_pair_t &pair : config.integrated)
    rhs.emplace(pair.state, pair.rhs);
  registry_writer().record_integrator(
      {config.dense_output, config.b, std::move(rhs)});
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

namespace {

using GroupData = GHExt::PatchData::LevelData::GroupData;

void require_serial_context() {
  if (omp_in_parallel())
    CCTK_VERROR("%s must run outside an OpenMP parallel region: it opens its "
                "own MFIter and OpenMP regions and warms AMReX's FPinfo cache",
                current_operation());
}

} // namespace

struct rk_step_t::impl {
  window_t window;
  stage_range stages;
  stage_range::iterator stage;
  bool have_rhs = false;
  std::vector<CCTK_REAL> b;
  std::vector<gi_t> states;
  std::vector<rk_level_t> levels;
};

rk_step_t open_rk_step(const window_t window) {
  const operation_scope_t op("CarpetX::open_rk_step");
  const GHExt::integrator_t &integrator = require_integrator();
  require_serial_context();
  const stage_range stages = integrator.stages();
  auto step = std::unique_ptr<rk_step_t::impl>(new rk_step_t::impl{
      .window = window, .stages = stages, .stage = stages.begin()});
  step->b = integrator.weights;
  step->states = integrated_states(integrator);
  step->levels = integrated_levels(window, integrator);
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
  if (p->stage == p->stages.end())
    CCTK_VERROR("CarpetX::rk_step_t::rhs_computed on %s: all %d stages of "
                "this step have run; destroy the rk_step_t and open the next "
                "step",
                describe(p->window).c_str(), p->stages.size());
  if (p->have_rhs)
    CCTK_VERROR("CarpetX::rk_step_t::rhs_computed on %s, RK stage %d of %d: "
                "the stage already has its right-hand side; expected "
                "state_updated",
                describe(p->window).c_str(), (*p->stage).index(),
                p->stages.size());
  p->have_rhs = true;
  const stage_t stage = *p->stage;
  const CCTK_REAL weight = p->b.at(p->stages.offset(stage));
  const CCTK_REAL delta_time = driver().get_global_cctkGH()->cctk_delta_time;
  using std::isfinite;
  if (!(isfinite(delta_time) && delta_time != 0))
    CCTK_VERROR("CarpetX::rk_step_t::rhs_computed on %s, RK stage %d of %d: "
                "cctk_delta_time is %g",
                describe(p->window).c_str(), stage.index(), p->stages.size(),
                double(delta_time));
  const std::string why = "CarpetX::rk_step_t::rhs_computed on " +
                          describe(p->window) + ": RK stage " +
                          std::to_string(stage.index()) + " of " +
                          std::to_string(p->stages.size()) +
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
  if (!p->have_rhs)
    CCTK_VERROR("CarpetX::rk_step_t::state_updated on %s, RK stage %d of %d: "
                "expected rhs_computed first",
                describe(p->window).c_str(), (*p->stage).index(),
                p->stages.size());
  const stage_t stage = *p->stage;
  ++p->stage;
  p->have_rhs = false;
  const bool end_of_step = p->stage == p->stages.end();
  require_operable(p->window, p->states);
  if (driver().use_subcycling) {
    require_serial_context();
    const std::string event =
        "CarpetX::rk_step_t::state_updated after RK stage " +
        std::to_string(stage.index()) + " of " +
        std::to_string(p->stages.size());
    for (const rk_level_t &entry : p->levels) {
      if (entry.level.index() == 0)
        continue;
      FillRKBoundary(entry.level, entry.groups.states, stage, event);
    }
  } else if (end_of_step) {
    Reflux(p->window);
  }
  synchronize();
}

rk_step_t::~rk_step_t() {
  if (!p)
    return;
  assert(p->stage != p->stages.end() || !p->have_rhs);
  if (p->stage != p->stages.end())
    CCTK_VERROR("CarpetX::rk_step_t on %s destroyed before RK stage %d of %d "
                "was updated; every stage calls rhs_computed and then "
                "state_updated before the step closes",
                describe(p->window).c_str(), (*p->stage).index(),
                p->stages.size());
}

} // namespace CarpetX
