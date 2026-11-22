#ifndef CARPETX_ODESOLVERS_RK_TABLEAU_HXX
#define CARPETX_ODESOLVERS_RK_TABLEAU_HXX

#include <cctk.h>

#include <array>
#include <limits>

namespace ODESolvers {

inline constexpr int max_stages = 13;
template <typename T> using per_stage = std::array<T, max_stages>;

struct tableau_t {
  int nstages, order;
  per_stage<per_stage<CCTK_REAL>> a;
  per_stage<CCTK_REAL> b, c;
};

namespace rk_tableau_detail {

constexpr CCTK_REAL magnitude(const CCTK_REAL x) { return x < 0 ? -x : x; }

constexpr CCTK_REAL sum_tolerance(const tableau_t &tableau) {
  return 4 * tableau.nstages * std::numeric_limits<CCTK_REAL>::epsilon();
}

constexpr CCTK_REAL order_tolerance(const tableau_t &tableau) {
  return 16 * tableau.nstages * std::numeric_limits<CCTK_REAL>::epsilon();
}

inline constexpr int max_tree_order = 8;
inline constexpr int num_rooted_trees = 200;

struct rooted_tree_t {
  int order;
  int gamma;
  int nchildren;
  std::array<int, max_tree_order - 1> children;
};

using rooted_tree_table_t = std::array<rooted_tree_t, num_rooted_trees>;

struct rooted_tree_generation_t {
  rooted_tree_table_t trees;
  int count;
};

constexpr void add_rooted_trees(rooted_tree_generation_t &generation,
                                rooted_tree_t tree, const int remaining,
                                const int max_child) {
  if (remaining == 0) {
    tree.gamma = tree.order;
    for (int n = 0; n < tree.nchildren; ++n)
      tree.gamma *= generation.trees[tree.children[n]].gamma;
    generation.trees[generation.count++] = tree;
    return;
  }
  for (int child = max_child; child >= 0; --child) {
    if (generation.trees[child].order > remaining)
      continue;
    rooted_tree_t grown = tree;
    grown.children[grown.nchildren++] = child;
    add_rooted_trees(generation, grown,
                     remaining - generation.trees[child].order, child);
  }
}

constexpr rooted_tree_generation_t generate_rooted_trees() {
  rooted_tree_generation_t generation{};
  for (int order = 1; order <= max_tree_order; ++order)
    add_rooted_trees(generation, rooted_tree_t{order, 0, 0, {}}, order - 1,
                     generation.count - 1);
  return generation;
}

static_assert(generate_rooted_trees().count == num_rooted_trees,
              "rk_tableau: there are 1, 1, 2, 4, 9, 20, 48 and 115 rooted "
              "trees with 1 to 8 vertices, 200 in all");

inline constexpr rooted_tree_table_t rooted_trees =
    generate_rooted_trees().trees;

} // namespace rk_tableau_detail

constexpr bool is_explicit(const tableau_t &tableau) {
  if (tableau.nstages < 1 || tableau.nstages > max_stages)
    return false;
  for (int s = 0; s < max_stages; ++s) {
    for (int i = 0; i < max_stages; ++i)
      if ((s >= tableau.nstages || i >= s) && tableau.a[s][i] != 0)
        return false;
    if (s >= tableau.nstages && (tableau.b[s] != 0 || tableau.c[s] != 0))
      return false;
  }
  return true;
}

constexpr bool rows_sum_to_c(const tableau_t &tableau) {
  for (int s = 0; s < tableau.nstages; ++s) {
    CCTK_REAL sum = 0;
    for (int i = 0; i < s; ++i)
      sum += tableau.a[s][i];
    if (rk_tableau_detail::magnitude(sum - tableau.c[s]) >
        rk_tableau_detail::sum_tolerance(tableau))
      return false;
  }
  return true;
}

constexpr bool weights_sum_to_one(const tableau_t &tableau) {
  CCTK_REAL sum = 0;
  for (int s = 0; s < tableau.nstages; ++s)
    sum += tableau.b[s];
  return rk_tableau_detail::magnitude(sum - 1) <=
         rk_tableau_detail::sum_tolerance(tableau);
}

constexpr bool has_declared_order(const tableau_t &tableau) {
  using namespace rk_tableau_detail;
  if (tableau.order < 1 || tableau.order > max_tree_order)
    return false;
  std::array<per_stage<CCTK_REAL>, num_rooted_trees> a_phi{};
  for (int k = 0;
       k < num_rooted_trees && rooted_trees[k].order <= tableau.order; ++k) {
    const rooted_tree_t &tree = rooted_trees[k];
    per_stage<CCTK_REAL> phi{};
    CCTK_REAL weighted = 0;
    for (int s = 0; s < tableau.nstages; ++s) {
      phi[s] = 1;
      for (int n = 0; n < tree.nchildren; ++n)
        phi[s] *= a_phi[tree.children[n]][s];
      weighted += tableau.b[s] * phi[s];
    }
    if (magnitude(tree.gamma * weighted - 1) > order_tolerance(tableau))
      return false;
    if (tree.order == tableau.order)
      continue;
    for (int s = 0; s < tableau.nstages; ++s) {
      CCTK_REAL sum = 0;
      for (int i = 0; i < s; ++i)
        sum += tableau.a[s][i] * phi[i];
      a_phi[k][s] = sum;
    }
  }
  return true;
}

namespace rk_tableau_detail {

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

} // namespace rk_tableau_detail

struct row_reads_t {
  CCTK_REAL acc;
  per_stage<CCTK_REAL> stage;
};

struct storage_plan_t {
  bool needs_old, needs_acc;
  per_stage<int> keep_until;
  per_stage<row_reads_t> reads;
};

constexpr storage_plan_t storage_plan(const tableau_t &tableau) {
  using namespace rk_tableau_detail;
  storage_plan_t plan{};
  plan.needs_old = tableau.nstages >= 2;
  for (int i = 0; i < max_stages; ++i)
    plan.keep_until[i] = -1;
  for (int row = 2; row < tableau.nstages; ++row) {
    if (earlier_stages_ratio_to_weights(tableau, row).exists)
      continue;
    for (int i = 0; i + 1 < row; ++i)
      if (tableau.a[row][i] != 0)
        plan.keep_until[i] = row;
  }
  for (int row = 1; row <= tableau.nstages; ++row) {
    const int s = row - 1;
    row_reads_t &reads = plan.reads[s];
    reads.stage[s] = row_weight(tableau, row, s);
    bool has_every_copy = true;
    for (int i = 0; i < s; ++i)
      if (row_weight(tableau, row, i) != 0 && plan.keep_until[i] < row)
        has_every_copy = false;
    if (has_every_copy)
      for (int i = 0; i < s; ++i)
        reads.stage[i] = row_weight(tableau, row, i);
    else
      reads.acc = earlier_stages_ratio_to_weights(tableau, row).ratio;
    plan.needs_acc = plan.needs_acc || reads.acc != 0;
  }
  return plan;
}

constexpr bool plan_reproduces_rows(const tableau_t &tableau) {
  using namespace rk_tableau_detail;
  const storage_plan_t plan = storage_plan(tableau);
  for (int row = 1; row <= tableau.nstages; ++row) {
    const row_reads_t &reads = plan.reads[row - 1];
    for (int i = 0; i < row; ++i) {
      const CCTK_REAL read =
          reads.stage[i] + (i + 1 < row ? reads.acc * tableau.b[i] : 0);
      if (magnitude(read - row_weight(tableau, row, i)) >
          sum_tolerance(tableau))
        return false;
      if (i + 1 < row && reads.stage[i] != 0 && plan.keep_until[i] < row)
        return false;
    }
  }
  return true;
}

constexpr int peak_copies(const storage_plan_t &plan) {
  int peak_stage_copies = 0;
  for (int s = 0; s < max_stages; ++s) {
    int live = 0;
    for (int i = 0; i <= s; ++i)
      live += plan.keep_until[i] > s;
    peak_stage_copies = live > peak_stage_copies ? live : peak_stage_copies;
  }
  return int(plan.needs_old) + int(plan.needs_acc) + peak_stage_copies;
}

} // namespace ODESolvers

#endif
