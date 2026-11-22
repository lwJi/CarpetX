#include "rk_methods.hxx"
#include "rk_run.hxx"
#include "rk_tableau.hxx"

#include <carpetx_rk.hxx>

#include <array>
#include <type_traits>
#include <utility>
#include <variant>

namespace ODESolvers {
namespace {

constexpr tableau_t tableau_of(const char *const name) {
  for (const rk_method_t &rk : rk_methods)
    if (rk_methods_detail::names_equal(rk.name, name))
      return std::get<explicit_rk_t>(rk.scheme).tableau;
  return tableau_t{};
}

constexpr tableau_t rk3_stages_ssprk3_weights = [] {
  tableau_t tableau = tableau_of("RK3");
  tableau.b = tableau_of("SSPRK3").b;
  return tableau;
}();

static_assert(is_explicit(rk3_stages_ssprk3_weights) &&
                  rows_sum_to_c(rk3_stages_ssprk3_weights) &&
                  weights_sum_to_one(rk3_stages_ssprk3_weights) &&
                  !has_declared_order(rk3_stages_ssprk3_weights),
              "rk_rejected: RK3's stages with SSPRK3's weights pass the "
              "row-sum and weight-sum rules and must be refused by the "
              "order rule");

constexpr tableau_t rk4_one_coefficient_off = [] {
  tableau_t tableau = tableau_of("RK4");
  tableau.a[2][1] = 1.0 / 3;
  return tableau;
}();

static_assert(is_explicit(rk4_one_coefficient_off) &&
                  !rows_sum_to_c(rk4_one_coefficient_off),
              "rk_rejected: RK4 with one mistyped a must be refused by the "
              "row-sum rule");

static_assert(
    !std::is_constructible_v<explicit_rk_t, CarpetX::rk_dense_output_t,
                             tableau_t>,
    "rk_rejected: a row is a tableau or a CarpetX dense-output method; a "
    "dense output written beside a tableau of the row's own must not compile");

static_assert(std::is_const_v<decltype(explicit_rk_t::dense_output)> &&
                  std::is_const_v<decltype(explicit_rk_t::tableau)>,
              "rk_rejected: a finished row's dense output and tableau are "
              "fixed by its constructor and must not be writable afterwards");

enum class event_t { compute_rhs, assign, land, release };

struct call_t {
  event_t event;
  int n;
  CCTK_REAL time;
  assign_t assign;
};

struct trace_t {
  int ncalls;
  std::array<call_t, 6 * max_stages + 4> call;
};

class recorder_t {
public:
  constexpr CCTK_REAL dt() const { return symbolic_dt; }
  constexpr void compute_rhs(const int s) {
    record({event_t::compute_rhs, s, 0, {}});
  }
  constexpr void assign(const assign_t &assign) {
    record({event_t::assign, assign.dst, 0, assign});
  }
  constexpr void land(const int s, const CCTK_REAL time) {
    record({event_t::land, s, time, {}});
  }
  constexpr void release(const int r) { record({event_t::release, r, 0, {}}); }

  trace_t trace{};

private:
  constexpr void record(const call_t &call) {
    trace.call[trace.ncalls++] = call;
  }
};

constexpr trace_t trace_of(const char *const name) {
  recorder_t recorder;
  run_explicit(tableau_of(name), recorder);
  return recorder.trace;
}

constexpr run_check_t replayed(const char *const name, const trace_t &trace) {
  const tableau_t tableau = tableau_of(name);
  symbolic_machine_t machine(tableau);
  for (int n = 0; n < trace.ncalls; ++n) {
    const call_t &call = trace.call[n];
    switch (call.event) {
    case event_t::compute_rhs:
      machine.compute_rhs(call.n);
      break;
    case event_t::assign:
      machine.assign(call.assign);
      break;
    case event_t::land:
      machine.land(call.n, call.time);
      break;
    case event_t::release:
      machine.release(call.n);
      break;
    }
  }
  return machine.report().check;
}

constexpr int call_index(const trace_t &trace, const int stage,
                         const event_t event, const int n) {
  int current_stage = -1;
  for (int index = 0; index < trace.ncalls; ++index) {
    const call_t &call = trace.call[index];
    if (call.event == event_t::compute_rhs)
      current_stage = call.n;
    if (current_stage == stage && call.event == event && call.n == n)
      return index;
  }
  return -1;
}

constexpr void insert_call(trace_t &trace, const int index,
                           const call_t &call) {
  for (int n = trace.ncalls; n > index; --n)
    trace.call[n] = trace.call[n - 1];
  trace.call[index] = call;
  ++trace.ncalls;
}

constexpr call_t remove_call(trace_t &trace, const int index) {
  const call_t call = trace.call[index];
  for (int n = index; n + 1 < trace.ncalls; ++n)
    trace.call[n] = trace.call[n + 1];
  --trace.ncalls;
  return call;
}

constexpr run_check_t refused_only_by(bool run_check_t::*const rule) {
  run_check_t check = every_rule_holds;
  check.*rule = false;
  return check;
}

static_assert(replayed("SSPRK3", trace_of("SSPRK3")) == every_rule_holds &&
                  replayed("RK2", trace_of("RK2")) == every_rule_holds &&
                  replayed("RK4", trace_of("RK4")) == every_rule_holds &&
                  replayed("DP87", trace_of("DP87")) == every_rule_holds,
              "rk_rejected: the calls run_explicit makes for a shipped "
              "method, recorded and replayed unchanged, must pass every rule");

constexpr trace_t ssprk3_feeds_before_state = [] {
  trace_t trace = trace_of("SSPRK3");
  const int update = call_index(trace, 1, event_t::assign, reg_var);
  const int feed = call_index(trace, 1, event_t::assign, reg_acc);
  std::swap(trace.call[update], trace.call[feed]);
  return trace;
}();

static_assert(replayed("SSPRK3", ssprk3_feeds_before_state) ==
                  refused_only_by(&run_check_t::reaches_rows),
              "rk_rejected: SSPRK3 with stage 1 feeding the weighted sum "
              "before the state reads it must be refused by the rule that "
              "each stage reaches its row, and by no other");

constexpr trace_t rk2_rhs_without_dt = [] {
  trace_t trace = trace_of("RK2");
  assign_t &update =
      trace.call[call_index(trace, 1, event_t::assign, reg_var)].assign;
  update.coeff[reg_rhs] = tableau_of("RK2").b[1];
  return trace;
}();

static_assert(replayed("RK2", rk2_rhs_without_dt) ==
                  refused_only_by(&run_check_t::reaches_rows),
              "rk_rejected: RK2 with the last stage's right-hand side added "
              "to the state without the factor dt must be refused by the "
              "rule that each stage reaches its row, and by no other");

constexpr trace_t ssprk3_shu_osher_stage = [] {
  trace_t trace = trace_of("SSPRK3");
  assign_t &update =
      trace.call[call_index(trace, 1, event_t::assign, reg_var)].assign;
  update.coeff = {};
  update.coeff[reg_var] = 1.0 / 4;
  update.coeff[reg_old] = 3.0 / 4;
  update.coeff[reg_rhs] = symbolic_dt / 4;
  return trace;
}();

static_assert(replayed("SSPRK3", ssprk3_shu_osher_stage) ==
                  refused_only_by(&run_check_t::reads_live),
              "rk_rejected: SSPRK3 with stage 1 in Shu-Osher form, a quarter "
              "of the state plus three quarters of the old state, reaches "
              "row 2 on paper and must be refused by the rule that a "
              "register is read only while it holds what was written, and "
              "by no other: the state has been through ODESolvers_PostStep");

constexpr trace_t dp87_reads_released_copy = [] {
  trace_t trace = trace_of("DP87");
  const call_t release =
      remove_call(trace, call_index(trace, 3, event_t::release, reg_k0 + 2));
  insert_call(trace, call_index(trace, 2, event_t::land, 2) + 1, release);
  return trace;
}();

static_assert(replayed("DP87", dp87_reads_released_copy) ==
                  refused_only_by(&run_check_t::reads_live),
              "rk_rejected: DP87 with the copy of stage 2 released one stage "
              "early, before stage 3 reads it, must be refused by the rule "
              "that a register is read only while it holds what was "
              "written, and by no other");

constexpr trace_t rk4_lands_at_own_c = [] {
  trace_t trace = trace_of("RK4");
  trace.call[call_index(trace, 0, event_t::land, 0)].time =
      tableau_of("RK4").c[0];
  return trace;
}();

static_assert(replayed("RK4", rk4_lands_at_own_c) ==
                  refused_only_by(&run_check_t::lands_at_c),
              "rk_rejected: RK4 with stage 0 landing at c[0], the time its "
              "right-hand side was computed at, must be refused by the rule "
              "that stage s lands at c[s + 1], and by no other");

constexpr trace_t rk4_releases_unheld_copy = [] {
  trace_t trace = trace_of("RK4");
  insert_call(trace, trace.ncalls, {event_t::release, reg_k0, 0, {}});
  return trace;
}();

static_assert(replayed("RK4", rk4_releases_unheld_copy) ==
                  refused_only_by(&run_check_t::releases_held),
              "rk_rejected: RK4 releasing a copy of stage 0, which it never "
              "takes, must be refused by the rule that only a held register "
              "is released, and by no other");

constexpr trace_t rk2_computes_rhs_twice = [] {
  trace_t trace = trace_of("RK2");
  const int rhs = call_index(trace, 1, event_t::compute_rhs, 1);
  insert_call(trace, rhs, trace.call[rhs]);
  return trace;
}();

static_assert(replayed("RK2", rk2_computes_rhs_twice) ==
                  refused_only_by(&run_check_t::stages_in_order),
              "rk_rejected: RK2 computing the right-hand side of stage 1 "
              "twice, which would feed the flux registers twice, reaches "
              "its rows and must be refused by the rule that each stage "
              "computes one right-hand side and lands once, and by no other");

static_assert(rk_machine<symbolic_machine_t> && rk_machine<recorder_t>,
              "rk_rejected: run_explicit takes a machine whose five "
              "operations return nothing but dt, so the calls it makes are "
              "fixed by the tableau alone");

} // namespace
} // namespace ODESolvers
