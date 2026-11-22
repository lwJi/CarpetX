#include "rk_step.hxx"

#include <carpetx_rk.hxx>

#include <cctk_Parameters.h>

#include <cassert>
#include <optional>
#include <variant>
#include <vector>

namespace ODESolvers {

namespace {

void step_explicit(const cGH *const cctkGH, const tableau_t &tableau,
                   const CarpetX::window_t window, statecomp_t &var,
                   const statecomp_t &rhs,
                   const std::span<const CarpetX::gi_t> dep_groups) {
  DECLARE_CCTK_PARAMETERS;

  static CarpetX::Timer timer_lincomb("ODESolvers::Solve::lincomb");
  static CarpetX::Timer timer_rhs("ODESolvers::Solve::rhs");
  static CarpetX::Timer timer_poststep("ODESolvers::Solve::poststep");

  const int nstages = tableau.nstages;
  const storage_plan_t plan = storage_plan(tableau);
  assert(plan.needs_old || nstages == 1);

  const CCTK_REAL dt = cctkGH->cctk_delta_time / cctkGH->cctk_timefac;
  const CCTK_REAL old_time = cctkGH->cctk_time - dt;
  CCTK_REAL t = old_time;

  std::optional<statecomp_t> old;
  if (plan.needs_old)
    old.emplace(var.copy());

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

  std::optional<statecomp_t> acc;
  std::vector<std::optional<statecomp_t> > ks(nstages);

  for (int s = 0; s < nstages; ++s) {
    calcrhs(s + 1);

    const int row = s + 1;
    const bool last = row == nstages;
    const row_reads_t &reads = plan.reads[s];

    if (plan.keep_until[s] >= 0)
      ks[s].emplace(rhs.copy());

    std::vector<CCTK_REAL> factors;
    std::vector<const statecomp_t *> srcs;
    const auto read = [&](const CCTK_REAL factor, const statecomp_t &src) {
      factors.push_back(factor);
      srcs.push_back(&src);
    };
    if (old)
      read(1, *old);
    if (last)
      assert(plan.needs_acc == bool(acc));
    if (reads.acc != 0) {
      assert(acc);
      read(reads.acc * dt, *acc);
    }
    for (int i = 0; i < s; ++i) {
      if (reads.stage[i] == 0)
        continue;
      assert(ks[i]);
      read(reads.stage[i] * dt, *ks[i]);
    }
    if (reads.stage[s] != 0)
      read(reads.stage[s] * dt, rhs);
    {
      CarpetX::Interval interval_lincomb(timer_lincomb);
      statecomp_t::lincomb(var, old ? 0 : 1, factors, srcs,
                           CarpetX::make_valid_int());
      if (!last && tableau.b[s] != 0) {
        if (acc)
          statecomp_t::lincomb(*acc, 1, reals<1>{tableau.b[s]}, states<1>{&rhs},
                               CarpetX::make_valid_int());
        else
          acc.emplace(rhs.scaled(tableau.b[s]));
      }
      CarpetX::invalidate(window, dep_groups, CarpetX::tl_t{0},
                          "ODESolvers updated the state vector");
    }
    calcpoststep(row, last ? dt : tableau.c[row] * dt);

    for (int i = 0; i < row; ++i)
      if (plan.keep_until[i] == row)
        ks[i].reset();
  }
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
