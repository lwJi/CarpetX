#ifndef CARPETX_ODESOLVERS_RK_METHODS_HXX
#define CARPETX_ODESOLVERS_RK_METHODS_HXX

#include <carpetx_rk.hxx>

#include <cctk.h>

#include <array>
#include <cstddef>
#include <initializer_list>
#include <limits>

namespace ODESolvers {

inline constexpr int max_table_stages = 13;

struct rk_method_t {
  const char *name;
  int nstages;
  CarpetX::rk_dense_output_t dense_output;
  bool implicit;
  std::array<CCTK_REAL, max_table_stages> b;
};

inline constexpr std::array<rk_method_t, 10> rk_methods{{
    {"constant", 4, CarpetX::rk_dense_output_t::rk4, false, {}},
    {"Euler", 1, CarpetX::rk_dense_output_t::none, false, {1}},
    {"RK2", 2, CarpetX::rk_dense_output_t::none, false, {0, 1}},
    {"RK3",
     3,
     CarpetX::rk_dense_output_t::none,
     false,
     {1.0 / 6, 2.0 / 3, 1.0 / 6}},
    {"SSPRK3",
     3,
     CarpetX::rk_dense_output_t::ssprk3,
     false,
     {1.0 / 6, 1.0 / 6, 2.0 / 3}},
    {"RK4",
     4,
     CarpetX::rk_dense_output_t::rk4,
     false,
     {1.0 / 6, 1.0 / 3, 1.0 / 3, 1.0 / 6}},
    {"RKF78",
     11,
     CarpetX::rk_dense_output_t::none,
     false,
     {41.0 / 840, 0, 0, 0, 0, 34.0 / 105, 9.0 / 35, 9.0 / 35, 9.0 / 280,
      9.0 / 280, 41.0 / 840}},
    {"DP87",
     13,
     CarpetX::rk_dense_output_t::none,
     false,
     {14005451.0 / 335480064, 0, 0, 0, 0, -59238493.0 / 1068277825,
      181606767.0 / 758867731, 561292985.0 / 797845732,
      -1041891430.0 / 1371343529, 760417239.0 / 1151165299,
      118820643.0 / 751138087, -528747749.0 / 2220607170, 1.0 / 4}},
    {"IMEX122", 2, CarpetX::rk_dense_output_t::none, true, {}},
    {"Implicit Euler", 2, CarpetX::rk_dense_output_t::none, true, {}},
}};

namespace rk_methods_detail {

constexpr bool row_consistent(const rk_method_t &rk) {
  if (rk.nstages < 1 || rk.nstages > max_table_stages)
    return false;
  const bool has_dense_output =
      rk.dense_output != CarpetX::rk_dense_output_t::none;
  if (has_dense_output &&
      rk.nstages != CarpetX::rk_dense_output_stages(rk.dense_output))
    return false;
  if (has_dense_output && rk.implicit)
    return false;
  CCTK_REAL sum = 0;
  bool all_zero = true;
  for (std::size_t s = 0; s < rk.b.size(); ++s) {
    sum += rk.b[s];
    all_zero = all_zero && rk.b[s] == 0;
    if (int(s) >= rk.nstages && rk.b[s] != 0)
      return false;
  }
  if (rk.implicit)
    return all_zero;
  const CCTK_REAL err = sum - 1 < 0 ? 1 - sum : sum - 1;
  return all_zero ||
         err <= 4 * rk.nstages * std::numeric_limits<CCTK_REAL>::epsilon();
}

constexpr bool table_consistent() {
  for (const auto &rk : rk_methods)
    if (!row_consistent(rk))
      return false;
  return true;
}

constexpr char lower(const char c) {
  return c >= 'A' && c <= 'Z' ? char(c - 'A' + 'a') : c;
}

constexpr bool names_equal(const char *a, const char *b) {
  while (*a && lower(*a) == lower(*b)) {
    ++a;
    ++b;
  }
  return lower(*a) == lower(*b);
}

constexpr bool names_distinct() {
  for (std::size_t i = 0; i < rk_methods.size(); ++i)
    for (std::size_t j = 0; j < i; ++j)
      if (names_equal(rk_methods[i].name, rk_methods[j].name))
        return false;
  return true;
}

constexpr bool
dense_output_rows_are(const std::initializer_list<const char *> methods) {
  std::size_t listed_rows = 0;
  for (const auto &rk : rk_methods) {
    bool listed = false;
    for (const char *const method : methods)
      listed = listed || names_equal(rk.name, method);
    if (listed != (rk.dense_output != CarpetX::rk_dense_output_t::none))
      return false;
    listed_rows += listed;
  }
  return listed_rows == methods.size();
}

} // namespace rk_methods_detail

static_assert(rk_methods_detail::table_consistent(),
              "rk_methods: a row's b must sum to 1 (or be all zero), vanish "
              "beyond nstages, and be all zero when the row is implicit; a "
              "row with a dense output must have that method's stage count "
              "(CarpetX::rk_dense_output_stages) and be explicit");
static_assert(rk_methods_detail::names_distinct(),
              "rk_methods: rk_method finds a row by its name, compared "
              "case-insensitively, so no two rows may share a name");

// Look up the row for the given ODESolvers::method (case-insensitive, like
// CCTK_EQUALS). Every keyword value has a row; a missing one is a hard error.
inline const rk_method_t &rk_method(const char *const method) {
  for (const auto &rk : rk_methods)
    if (CCTK_Equals(method, rk.name))
      return rk;
  CCTK_VERROR("ODESolvers::method = \"%s\" has no row in the RK method table "
              "(ODESolvers/src/rk_methods.hxx)",
              method);
}

} // namespace ODESolvers

#endif // #ifndef CARPETX_ODESOLVERS_RK_METHODS_HXX
