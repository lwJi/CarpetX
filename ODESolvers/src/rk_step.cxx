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
  const auto calcupdate = [&](const int n, const CCTK_REAL c,
                              const CCTK_REAL scale,
                              const std::vector<CCTK_REAL> &factors,
                              const std::vector<const statecomp_t *> &srcs) {
    {
      CarpetX::Interval interval_lincomb(timer_lincomb);
      statecomp_t::lincomb(var, scale, factors, srcs,
                           CarpetX::make_valid_int());
      CarpetX::invalidate(window, dep_groups, CarpetX::tl_t{0},
                          "ODESolvers updated the state vector");
    }
    rk_step.state_updated();
    {
      CarpetX::Interval interval_poststep(timer_poststep);
      t = old_time + c;
      CarpetX::run_schedule_group_at(cctkGH, "ODESolvers_PostStep", t,
                                     cctkGH->cctk_delta_time);
      if (verbose)
        CCTK_VINFO("Calculated new state #%d at t=%g", n, double(t));
    }
  };

  std::optional<statecomp_t> acc;
  std::vector<std::optional<statecomp_t> > ks(nstages);

  for (int s = 0; s < nstages; ++s) {
    calcrhs(s + 1);

    if (s + 1 < nstages && tableau.b[s] != 0) {
      CarpetX::Interval interval_lincomb(timer_lincomb);
      if (acc) {
        statecomp_t::lincomb(*acc, 1, reals<1>{tableau.b[s]}, states<1>{&rhs},
                             CarpetX::make_valid_int());
      } else {
        acc.emplace(rhs.copy());
        statecomp_t::lincomb(*acc, 0, reals<1>{tableau.b[s]}, states<1>{&rhs},
                             CarpetX::make_valid_int());
      }
    }

    if (plan.keep_until[s] >= 0)
      ks[s].emplace(rhs.copy());

    if (s + 1 < nstages) {
      const int row = s + 1;
      std::vector<CCTK_REAL> factors{1};
      std::vector<const statecomp_t *> srcs{&old.value()};
      for (int i = 0; i < row; ++i) {
        if (tableau.a[row][i] == 0)
          continue;
        assert(i == s || ks[i]);
        factors.push_back(tableau.a[row][i] * dt);
        srcs.push_back(i == s ? &rhs : &*ks[i]);
      }
      calcupdate(row, tableau.c[row] * dt, 0, factors, srcs);
      for (int i = 0; i < row; ++i)
        if (plan.keep_until[i] == row)
          ks[i].reset();
    }
  }

  assert(plan.needs_acc == bool(acc));
  std::vector<CCTK_REAL> factors;
  std::vector<const statecomp_t *> srcs;
  if (old) {
    factors.push_back(1);
    srcs.push_back(&*old);
  }
  if (acc) {
    factors.push_back(dt);
    srcs.push_back(&*acc);
  }
  if (tableau.b[nstages - 1] != 0) {
    factors.push_back(tableau.b[nstages - 1] * dt);
    srcs.push_back(&rhs);
  }
  calcupdate(nstages, dt, old ? 0 : 1, factors, srcs);
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
