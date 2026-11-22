#include "rk_step.hxx"

#include "rk_program.hxx"

#include <carpetx_rk.hxx>

#include <cctk_Parameters.h>

#include <cassert>
#include <optional>
#include <variant>
#include <vector>

namespace ODESolvers {

namespace {

class registers_t {
public:
  registers_t(statecomp_t &var, const statecomp_t &rhs) : var(var), rhs(rhs) {}

  void assign(const assign_t &a, const CCTK_REAL dt) {
    statecomp_t &dst = written(a);
    std::vector<CCTK_REAL> factors;
    std::vector<const statecomp_t *> srcs;
    for (int r = 0; r < num_regs; ++r) {
      if (r == a.dst || a.coeff[r] == 0)
        continue;
      factors.push_back(a.coeff[r] *
                        (is_state(a.dst) && !is_state(r) ? dt : 1));
      srcs.push_back(&read(r));
    }
    statecomp_t::lincomb(dst, a.coeff[a.dst], factors, srcs,
                         CarpetX::make_valid_int());
  }

  void release(const per_reg<bool> &frees) {
    for (int r = 0; r < num_regs; ++r) {
      if (!frees[r])
        continue;
      assert(is_scratch(r));
      assert(scratch[r]);
      scratch[r].reset();
    }
  }

private:
  statecomp_t &written(const assign_t &a) {
    assert(a.dst != reg_rhs);
    if (a.dst == reg_var)
      return var;
    std::optional<statecomp_t> &dst = scratch[a.dst];
    if (!dst) {
      assert(a.coeff[a.dst] == 0);
      dst.emplace((is_state(a.dst) ? var : rhs).scratch_like());
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

  statecomp_t &var;
  const statecomp_t &rhs;
  per_reg<std::optional<statecomp_t> > scratch;
};

void step_explicit(const cGH *const cctkGH, const program_t &program,
                   const CarpetX::window_t window, statecomp_t &var,
                   const statecomp_t &rhs,
                   const std::span<const CarpetX::gi_t> dep_groups) {
  DECLARE_CCTK_PARAMETERS;

  static CarpetX::Timer timer_lincomb("ODESolvers::Solve::lincomb");
  static CarpetX::Timer timer_rhs("ODESolvers::Solve::rhs");
  static CarpetX::Timer timer_poststep("ODESolvers::Solve::poststep");

  const CCTK_REAL dt = cctkGH->cctk_delta_time / cctkGH->cctk_timefac;
  const CCTK_REAL old_time = cctkGH->cctk_time - dt;
  CCTK_REAL t = old_time;

  CarpetX::rk_step_t rk_step = CarpetX::open_rk_step(window);

  const auto calcrhs = [&](const int n) {
    CarpetX::Interval interval_rhs(timer_rhs);
    if (verbose)
      CCTK_VINFO("Calculating RHS #%d at t=%g", n, double(t));
    CarpetX::run_schedule_group_at(cctkGH, "ODESolvers_RHS", t,
                                   cctkGH->cctk_delta_time);
    rk_step.rhs_computed();
  };
  const auto calcpoststep = [&](const int n, const CCTK_REAL c) {
    rk_step.state_updated();
    CarpetX::Interval interval_poststep(timer_poststep);
    t = old_time + c;
    CarpetX::run_schedule_group_at(cctkGH, "ODESolvers_PostStep", t,
                                   cctkGH->cctk_delta_time);
    if (verbose)
      CCTK_VINFO("Calculated new state #%d at t=%g", n, double(t));
  };

  registers_t regs(var, rhs);

  for (int s = 0; s < program.nstages; ++s) {
    const stage_t &stage = program.stage[s];
    calcrhs(s + 1);
    {
      CarpetX::Interval interval_lincomb(timer_lincomb);
      for (int n = 0; n < stage.nassign; ++n)
        regs.assign(stage.assign[n], dt);
      CarpetX::invalidate(window, dep_groups, CarpetX::tl_t{0},
                          "ODESolvers updated the state vector");
    }
    calcpoststep(s + 1, stage.time * dt);
    regs.release(stage.frees);
  }
}

} // namespace

void step(const cGH *const cctkGH, const rk_method_t &method,
          const CarpetX::window_t window, statecomp_t &var,
          const statecomp_t &rhs,
          const std::span<const CarpetX::gi_t> dep_groups) {
  std::visit(overloaded{[](const frozen_t &) {},
                        [&](const explicit_rk_t &scheme) {
                          step_explicit(cctkGH, scheme.program, window, var,
                                        rhs, dep_groups);
                        }},
             method.scheme);
}

} // namespace ODESolvers
