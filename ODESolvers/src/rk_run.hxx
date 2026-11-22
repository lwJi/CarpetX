#ifndef CARPETX_ODESOLVERS_RK_RUN_HXX
#define CARPETX_ODESOLVERS_RK_RUN_HXX

#include "rk_tableau.hxx"

#include <cctk.h>

#include <array>
#include <concepts>

namespace ODESolvers {

inline constexpr int reg_var = 0, reg_old = 1, reg_acc = 2, reg_k0 = 3,
                     reg_rhs = reg_k0 + max_stages, num_regs = reg_rhs + 1;
template <typename T> using per_reg = std::array<T, num_regs>;

constexpr bool is_state(const int r) { return r == reg_var || r == reg_old; }

constexpr bool is_scratch(const int r) { return r != reg_var && r != reg_rhs; }

struct assign_t {
  int dst;
  per_reg<CCTK_REAL> coeff;
};

constexpr assign_t assign_to(const int dst) {
  assign_t assign{};
  assign.dst = dst;
  return assign;
}

template <typename Machine>
concept rk_machine = requires(Machine &machine, const assign_t &assign,
                              const int n, const CCTK_REAL time) {
  { machine.dt() } -> std::same_as<CCTK_REAL>;
  { machine.compute_rhs(n) } -> std::same_as<void>;
  { machine.assign(assign) } -> std::same_as<void>;
  { machine.land(n, time) } -> std::same_as<void>;
  { machine.release(n) } -> std::same_as<void>;
};

template <rk_machine Machine>
constexpr void run_explicit(const tableau_t &tableau, Machine &machine) {
  const int nstages = tableau.nstages;
  const storage_plan_t plan = storage_plan(tableau);
  const CCTK_REAL dt = machine.dt();

  if (plan.needs_old) {
    assign_t save = assign_to(reg_old);
    save.coeff[reg_var] = 1;
    machine.assign(save);
  }

  bool acc_holds_a_sum = false;
  for (int s = 0; s < nstages; ++s) {
    machine.compute_rhs(s);

    const int row = s + 1;
    const bool last = row == nstages;
    const row_reads_t &reads = plan.reads[s];

    if (plan.keep_until[s] >= 0) {
      assign_t keep = assign_to(reg_k0 + s);
      keep.coeff[reg_rhs] = 1;
      machine.assign(keep);
    }

    assign_t update = assign_to(reg_var);
    update.coeff[plan.needs_old ? reg_old : reg_var] = 1;
    update.coeff[reg_acc] = reads.acc * dt;
    for (int i = 0; i < s; ++i)
      update.coeff[reg_k0 + i] = reads.stage[i] * dt;
    update.coeff[reg_rhs] = reads.stage[s] * dt;
    machine.assign(update);

    if (!last && tableau.b[s] != 0) {
      assign_t feed = assign_to(reg_acc);
      feed.coeff[reg_acc] = acc_holds_a_sum ? 1 : 0;
      feed.coeff[reg_rhs] = tableau.b[s];
      machine.assign(feed);
      acc_holds_a_sum = true;
    }

    machine.land(s, last ? 1 : tableau.c[row]);

    for (int i = 0; i < row; ++i)
      if (plan.keep_until[i] == row)
        machine.release(reg_k0 + i);
  }
}

struct run_check_t {
  bool reaches_rows;
  bool lands_at_c;
  bool reads_live;
  bool releases_held;
  bool stages_in_order;
  constexpr bool operator==(const run_check_t &) const = default;
};

inline constexpr run_check_t every_rule_holds{true, true, true, true, true};

struct run_report_t {
  run_check_t check;
  int peak_copies;
};

inline constexpr CCTK_REAL symbolic_dt = CCTK_REAL(1) / 2;

class symbolic_machine_t {
public:
  constexpr explicit symbolic_machine_t(const tableau_t &tableau)
      : tableau(tableau) {
    reg[reg_var].live = true;
    reg[reg_var].of_step_start_state = 1;
  }

  constexpr CCTK_REAL dt() const { return symbolic_dt; }

  constexpr void compute_rhs(const int s) {
    if (s != nlanded || ncomputed != nlanded || state_written_since_landing)
      check.stages_in_order = false;
    ++ncomputed;
    reg[reg_rhs] = value_t{};
    reg[reg_rhs].live = true;
    if (s >= 0 && s < max_stages)
      reg[reg_rhs].of_stage_rhs[s] = 1;
  }

  constexpr void assign(const assign_t &assign) {
    value_t written{};
    written.live = true;
    for (int r = 0; r < num_regs; ++r) {
      if (assign.coeff[r] == 0)
        continue;
      if (!reg[r].live)
        check.reads_live = false;
      written.of_step_start_state +=
          assign.coeff[r] * reg[r].of_step_start_state;
      for (int i = 0; i < max_stages; ++i)
        written.of_stage_rhs[i] += assign.coeff[r] * reg[r].of_stage_rhs[i];
    }
    if (is_scratch(assign.dst) && !reg[assign.dst].live) {
      ++nheld;
      peak = nheld > peak ? nheld : peak;
    }
    if (assign.dst == reg_var)
      state_written_since_landing = true;
    reg[assign.dst] = written;
  }

  constexpr void land(const int s, const CCTK_REAL time) {
    using rk_tableau_detail::magnitude;
    using rk_tableau_detail::row_weight;
    using rk_tableau_detail::sum_tolerance;
    if (s != nlanded || ncomputed != nlanded + 1)
      check.stages_in_order = false;
    ++nlanded;
    state_written_since_landing = false;

    const int row = nlanded;
    if (row > tableau.nstages)
      return;
    const value_t &state = reg[reg_var];
    if (!state.live ||
        magnitude(state.of_step_start_state - 1) > sum_tolerance(tableau))
      check.reaches_rows = false;
    for (int i = 0; i < max_stages; ++i) {
      const CCTK_REAL weight = i < row ? row_weight(tableau, row, i) : 0;
      if (magnitude(state.of_stage_rhs[i] - symbolic_dt * weight) >
          sum_tolerance(tableau))
        check.reaches_rows = false;
    }
    if (time != (row == tableau.nstages ? 1 : tableau.c[row]))
      check.lands_at_c = false;
    reg[reg_var].live = false;
  }

  constexpr void release(const int r) {
    if (!is_scratch(r) || !reg[r].live) {
      check.releases_held = false;
      return;
    }
    reg[r].live = false;
    --nheld;
  }

  constexpr run_report_t report() const {
    run_report_t result{check, peak};
    if (nlanded != tableau.nstages)
      result.check.stages_in_order = false;
    return result;
  }

private:
  struct value_t {
    bool live;
    CCTK_REAL of_step_start_state;
    per_stage<CCTK_REAL> of_stage_rhs;
  };

  const tableau_t &tableau;
  per_reg<value_t> reg{};
  run_check_t check = every_rule_holds;
  bool state_written_since_landing = false;
  int ncomputed = 0, nlanded = 0, nheld = 0, peak = 0;
};

constexpr run_report_t symbolic_run(const tableau_t &tableau) {
  if (!is_explicit(tableau))
    return run_report_t{};
  symbolic_machine_t machine(tableau);
  run_explicit(tableau, machine);
  return machine.report();
}

} // namespace ODESolvers

#endif
