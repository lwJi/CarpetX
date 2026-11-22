#ifndef CARPETX_ODESOLVERS_RK_PROGRAM_HXX
#define CARPETX_ODESOLVERS_RK_PROGRAM_HXX

#include "rk_tableau.hxx"

#include <cctk.h>

#include <array>

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

struct stage_t {
  int nassign;
  std::array<assign_t, 4> assign;
  CCTK_REAL time;
  per_reg<bool> frees;
};

struct program_t {
  int nstages;
  per_stage<stage_t> stage;
};

namespace rk_program_detail {

constexpr CCTK_REAL row_weight(const tableau_t &tableau, const int row,
                               const int i) {
  return row == tableau.nstages ? tableau.b[i] : tableau.a[row][i];
}

struct ratio_to_weights_t {
  bool exists;
  CCTK_REAL ratio;
};

constexpr ratio_to_weights_t
earlier_stages_ratio_to_weights(const tableau_t &tableau, const int row) {
  const int nearlier = row - 1;
  int pivot = -1;
  for (int i = nearlier - 1; i >= 0; --i)
    if (tableau.b[i] != 0)
      pivot = i;
  if (pivot < 0) {
    for (int i = 0; i < nearlier; ++i)
      if (row_weight(tableau, row, i) != 0)
        return {false, 0};
    return {true, 0};
  }
  for (int i = 0; i < nearlier; ++i)
    if (row_weight(tableau, row, i) * tableau.b[pivot] !=
        row_weight(tableau, row, pivot) * tableau.b[i])
      return {false, 0};
  return {true, row_weight(tableau, row, pivot) / tableau.b[pivot]};
}

constexpr assign_t &next_assign(stage_t &stage, const int dst) {
  assign_t &assign = stage.assign[stage.nassign++];
  assign.dst = dst;
  return assign;
}

struct symbolic_value_t {
  bool live;
  CCTK_REAL of_step_start_state;
  per_stage<CCTK_REAL> of_stage_rhs;
};

} // namespace rk_program_detail

constexpr program_t plan(const tableau_t &tableau) {
  using namespace rk_program_detail;
  program_t program{};
  if (!is_explicit(tableau))
    return program;
  const int nstages = tableau.nstages;
  program.nstages = nstages;

  per_stage<int> copy_last_read_by_row{};
  for (int i = 0; i < max_stages; ++i)
    copy_last_read_by_row[i] = -1;
  for (int row = 2; row < nstages; ++row) {
    if (earlier_stages_ratio_to_weights(tableau, row).exists)
      continue;
    for (int i = 0; i + 1 < row; ++i)
      if (tableau.a[row][i] != 0)
        copy_last_read_by_row[i] = row;
  }

  const bool saves_old = nstages >= 2;
  bool acc_holds_a_sum = false;
  for (int s = 0; s < nstages; ++s) {
    const int row = s + 1;
    const bool last = row == nstages;
    stage_t &stage = program.stage[s];

    if (s == 0 && saves_old)
      next_assign(stage, reg_old).coeff[reg_var] = 1;

    if (copy_last_read_by_row[s] >= 0)
      next_assign(stage, reg_k0 + s).coeff[reg_rhs] = 1;

    assign_t &state = next_assign(stage, reg_var);
    state.coeff[s == 0 ? reg_var : reg_old] = 1;
    state.coeff[reg_rhs] = row_weight(tableau, row, s);
    bool has_every_copy = true;
    for (int i = 0; i < s; ++i)
      if (row_weight(tableau, row, i) != 0 && copy_last_read_by_row[i] < row)
        has_every_copy = false;
    if (has_every_copy)
      for (int i = 0; i < s; ++i)
        state.coeff[reg_k0 + i] = row_weight(tableau, row, i);
    else
      state.coeff[reg_acc] =
          earlier_stages_ratio_to_weights(tableau, row).ratio;

    if (!last && tableau.b[s] != 0) {
      assign_t &feed = next_assign(stage, reg_acc);
      feed.coeff[reg_acc] = acc_holds_a_sum ? 1 : 0;
      feed.coeff[reg_rhs] = tableau.b[s];
      acc_holds_a_sum = true;
    }

    stage.time = last ? 1 : tableau.c[row];

    for (int i = 0; i < row; ++i)
      stage.frees[reg_k0 + i] = copy_last_read_by_row[i] == row;
    if (last) {
      stage.frees[reg_old] = saves_old;
      stage.frees[reg_acc] = acc_holds_a_sum;
    }
  }
  return program;
}

struct program_check_t {
  bool reaches_rows;
  bool reads_live;
  bool rates_read_rates;
  bool reads_var_at_stage_0_only;
  bool order_free;
  constexpr bool operator==(const program_check_t &) const = default;
};

inline constexpr program_check_t every_rule_holds{true, true, true, true, true};

constexpr program_check_t check_program(const tableau_t &tableau,
                                        const program_t &program) {
  using namespace rk_program_detail;
  using rk_tableau_detail::magnitude;
  using rk_tableau_detail::sum_tolerance;
  if (!is_explicit(tableau))
    return program_check_t{};
  program_check_t check = every_rule_holds;
  per_reg<symbolic_value_t> reg{};
  reg[reg_var].live = true;
  reg[reg_var].of_step_start_state = 1;
  for (int s = 0; s < tableau.nstages; ++s) {
    const stage_t &stage = program.stage[s];
    reg[reg_rhs] = symbolic_value_t{};
    reg[reg_rhs].live = true;
    reg[reg_rhs].of_stage_rhs[s] = 1;

    const per_reg<symbolic_value_t> before_stage = reg;
    per_reg<bool> written_earlier_in_stage{};
    for (int n = 0; n < stage.nassign; ++n) {
      const assign_t &assign = stage.assign[n];
      if (s > 0 && assign.coeff[reg_var] != 0)
        check.reads_var_at_stage_0_only = false;
      symbolic_value_t written{};
      written.live = true;
      for (int r = 0; r < num_regs; ++r) {
        if (assign.coeff[r] == 0)
          continue;
        const symbolic_value_t &read = before_stage[r];
        if (!read.live)
          check.reads_live = false;
        if (!is_state(assign.dst) && is_state(r))
          check.rates_read_rates = false;
        if (written_earlier_in_stage[r])
          check.order_free = false;
        written.of_step_start_state +=
            assign.coeff[r] * read.of_step_start_state;
        for (int i = 0; i < max_stages; ++i)
          written.of_stage_rhs[i] += assign.coeff[r] * read.of_stage_rhs[i];
      }
      reg[assign.dst] = written;
      written_earlier_in_stage[assign.dst] = true;
    }

    const int row = s + 1;
    const symbolic_value_t &state = reg[reg_var];
    if (magnitude(state.of_step_start_state - 1) > sum_tolerance(tableau))
      check.reaches_rows = false;
    for (int i = 0; i < max_stages; ++i) {
      const CCTK_REAL weight = i < row ? row_weight(tableau, row, i) : 0;
      if (magnitude(state.of_stage_rhs[i] - weight) > sum_tolerance(tableau))
        check.reaches_rows = false;
    }

    for (int r = 0; r < num_regs; ++r)
      if (stage.frees[r])
        reg[r].live = false;
  }
  return check;
}

constexpr int peak_copies(const program_t &program) {
  per_reg<bool> live{};
  int peak = 0;
  for (int s = 0; s < program.nstages; ++s) {
    const stage_t &stage = program.stage[s];
    for (int n = 0; n < stage.nassign; ++n)
      live[stage.assign[n].dst] = true;
    int nlive_scratch = 0;
    for (int r = 0; r < num_regs; ++r)
      nlive_scratch += live[r] && is_scratch(r);
    peak = nlive_scratch > peak ? nlive_scratch : peak;
    for (int r = 0; r < num_regs; ++r)
      if (stage.frees[r])
        live[r] = false;
  }
  return peak;
}

} // namespace ODESolvers

#endif
