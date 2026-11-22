#include "rk_step.hxx"

#include "rk_run.hxx"

#include <carpetx_rk.hxx>

#include <cctk_Parameters.h>

#include <cassert>
#include <optional>
#include <variant>
#include <vector>

namespace ODESolvers {

namespace {

class grid_machine_t {
public:
  grid_machine_t(const cGH *const cctkGH, const CarpetX::window_t window,
                 statecomp_t &var, const statecomp_t &rhs,
                 const std::span<const CarpetX::gi_t> dep_groups)
      : cctkGH(cctkGH), window(window), var(var), rhs(rhs),
        dep_groups(dep_groups),
        step_size(cctkGH->cctk_delta_time / cctkGH->cctk_timefac),
        old_time(cctkGH->cctk_time - step_size), time(old_time) {}

  CCTK_REAL dt() const { return step_size; }

  void compute_rhs(const int s) {
    DECLARE_CCTK_PARAMETERS;
    static CarpetX::Timer timer_rhs("ODESolvers::Solve::rhs");
    if (!rk_step)
      rk_step.emplace(CarpetX::open_rk_step(window));
    CarpetX::Interval interval_rhs(timer_rhs);
    if (verbose)
      CCTK_VINFO("Calculating RHS #%d at t=%g", s + 1, double(time));
    CarpetX::run_schedule_group_at(cctkGH, "ODESolvers_RHS", time,
                                   cctkGH->cctk_delta_time);
    rk_step->rhs_computed();
  }

  void assign(const assign_t &assign) {
    CarpetX::Interval interval_lincomb(timer_lincomb());
    statecomp_t &dst = written(assign);
    std::vector<CCTK_REAL> factors;
    std::vector<const statecomp_t *> srcs;
    for (int r = 0; r < num_regs; ++r) {
      if (r == assign.dst || assign.coeff[r] == 0)
        continue;
      factors.push_back(assign.coeff[r]);
      srcs.push_back(&read(r));
    }
    statecomp_t::lincomb(dst, assign.coeff[assign.dst], factors, srcs,
                         CarpetX::make_valid_int());
  }

  void land(const int s, const CCTK_REAL fraction) {
    DECLARE_CCTK_PARAMETERS;
    static CarpetX::Timer timer_poststep("ODESolvers::Solve::poststep");
    {
      CarpetX::Interval interval_lincomb(timer_lincomb());
      CarpetX::invalidate(window, dep_groups, CarpetX::tl_t{0},
                          "ODESolvers updated the state vector");
    }
    assert(rk_step);
    rk_step->state_updated();
    CarpetX::Interval interval_poststep(timer_poststep);
    time = old_time + fraction * step_size;
    CarpetX::run_schedule_group_at(cctkGH, "ODESolvers_PostStep", time,
                                   cctkGH->cctk_delta_time);
    if (verbose)
      CCTK_VINFO("Calculated new state #%d at t=%g", s + 1, double(time));
  }

  void release(const int r) {
    assert(is_scratch(r));
    assert(scratch[r]);
    scratch[r].reset();
  }

private:
  static CarpetX::Timer &timer_lincomb() {
    static CarpetX::Timer timer("ODESolvers::Solve::lincomb");
    return timer;
  }

  statecomp_t &written(const assign_t &assign) {
    assert(assign.dst != reg_rhs);
    if (assign.dst == reg_var)
      return var;
    std::optional<statecomp_t> &dst = scratch[assign.dst];
    if (!dst) {
      assert(assign.coeff[assign.dst] == 0);
      dst.emplace((is_state(assign.dst) ? var : rhs).scratch_like());
    }
    return *dst;
  }

  const statecomp_t &read(const int r) const {
    if (r == reg_var)
      return var;
    if (r == reg_rhs)
      return rhs;
    assert(scratch[r]);
    return *scratch[r];
  }

  const cGH *const cctkGH;
  const CarpetX::window_t window;
  statecomp_t &var;
  const statecomp_t &rhs;
  const std::span<const CarpetX::gi_t> dep_groups;
  const CCTK_REAL step_size, old_time;
  CCTK_REAL time;
  std::optional<CarpetX::rk_step_t> rk_step;
  per_reg<std::optional<statecomp_t> > scratch;
};

void step_explicit(const cGH *const cctkGH, const tableau_t &tableau,
                   const CarpetX::window_t window, statecomp_t &var,
                   const statecomp_t &rhs,
                   const std::span<const CarpetX::gi_t> dep_groups) {
  grid_machine_t machine(cctkGH, window, var, rhs, dep_groups);
  run_explicit(tableau, machine);
}

} // namespace

void step(const cGH *const cctkGH, const rk_method_t &method,
          const CarpetX::window_t window, statecomp_t &var,
          const statecomp_t &rhs,
          const std::span<const CarpetX::gi_t> dep_groups) {
  std::visit(overloaded{[](const frozen_t &) {},
                        [&](const explicit_rk_t &scheme) {
                          step_explicit(cctkGH, scheme.tableau, window, var,
                                        rhs, dep_groups);
                        }},
             method.scheme);
}

} // namespace ODESolvers
