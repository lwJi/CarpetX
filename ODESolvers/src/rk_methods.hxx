#ifndef CARPETX_ODESOLVERS_RK_METHODS_HXX
#define CARPETX_ODESOLVERS_RK_METHODS_HXX

#include "rk_tableau.hxx"

#include <carpetx_rk.hxx>

#include <cctk.h>

#include <array>
#include <cstddef>
#include <variant>

namespace ODESolvers {

struct frozen_t {};

constexpr tableau_t
dense_output_tableau(const CarpetX::rk_dense_output_t dense_output) {
  const CarpetX::rk_dense_output_method_t method =
      CarpetX::rk_dense_output_method(dense_output);
  tableau_t tableau{};
  tableau.nstages = method.nstages;
  tableau.order = method.order;
  for (int s = 0; s < method.nstages; ++s) {
    for (int i = 0; i < s; ++i)
      tableau.a[s][i] = method.a[s][i];
    tableau.b[s] = method.b[s];
    tableau.c[s] = method.c[s];
  }
  return tableau;
}

struct explicit_rk_t {
  constexpr explicit_rk_t(const tableau_t &tableau)
      : dense_output(CarpetX::rk_dense_output_t::none), tableau(tableau) {}
  constexpr explicit_rk_t(const CarpetX::rk_dense_output_t dense_output)
      : dense_output(dense_output),
        tableau(dense_output_tableau(dense_output)) {}

  const CarpetX::rk_dense_output_t dense_output;
  const tableau_t tableau;
};

struct rk_method_t {
  const char *name;
  std::variant<frozen_t, explicit_rk_t> scheme;
};

template <typename... Arms> struct overloaded : Arms... {
  using Arms::operator()...;
};
template <typename... Arms> overloaded(Arms...) -> overloaded<Arms...>;

inline constexpr std::array<rk_method_t, 8> rk_methods{{
    {"constant", frozen_t{}},
    {"Euler", explicit_rk_t{tableau_t{1, 1, {}, {1}, {0}}}},
    {"RK2",
     explicit_rk_t{tableau_t{2, 2, {{{}, {1.0 / 2}}}, {0, 1}, {0, 1.0 / 2}}}},
    {"RK3", explicit_rk_t{tableau_t{3,
                                    3,
                                    {{{}, {1.0 / 2}, {-1, 2}}},
                                    {1.0 / 6, 2.0 / 3, 1.0 / 6},
                                    {0, 1.0 / 2, 1}}}},
    {"SSPRK3", explicit_rk_t{CarpetX::rk_dense_output_t::ssprk3}},
    {"RK4", explicit_rk_t{CarpetX::rk_dense_output_t::rk4}},
    {"RKF78",
     explicit_rk_t{
         tableau_t{11,
          7,
          {{{},
            {2.0 / 27},
            {1.0 / 36, 3.0 / 36},
            {1.0 / 24, 0, 3.0 / 24},
            {20.0 / 48, 0, -75.0 / 48, 75.0 / 48},
            {1.0 / 20, 0, 0, 5.0 / 20, 4.0 / 20},
            {-25.0 / 108, 0, 0, 125.0 / 108, -260.0 / 108, 250.0 / 108},
            {31.0 / 300, 0, 0, 0, 61.0 / 225, -2.0 / 9, 13.0 / 900},
            {2, 0, 0, -53.0 / 6, 704.0 / 45, -107.0 / 9, 67.0 / 90, 3},
            {-91.0 / 108, 0, 0, 23.0 / 108, -976.0 / 135, 311.0 / 54,
             -19.0 / 60, 17.0 / 6, -1.0 / 12},
            {2383.0 / 4100, 0, 0, -341.0 / 164, 4496.0 / 1025, -301.0 / 82,
             2133.0 / 4100, 45.0 / 82, 45.0 / 164, 18.0 / 41}}},
          {41.0 / 840, 0, 0, 0, 0, 34.0 / 105, 9.0 / 35, 9.0 / 35, 9.0 / 280,
           9.0 / 280, 41.0 / 840},
          {0, 2.0 / 27, 1.0 / 9, 1.0 / 6, 5.0 / 12, 1.0 / 2, 5.0 / 6, 1.0 / 6,
           2.0 / 3, 1.0 / 3, 1}}}},
    // doi:10.1016/0771-050X(81)90010-3
    {"DP87",
     explicit_rk_t{
         tableau_t{13,
          8,
          {{{},
            {1.0 / 18},
            {1.0 / 48, 1.0 / 16},
            {1.0 / 32, 0, 3.0 / 32},
            {5.0 / 16, 0, -75.0 / 64, 75.0 / 64},
            {3.0 / 80, 0, 0, 3.0 / 16, 3.0 / 20},
            {29443841.0 / 614563906, 0, 0, 77736538.0 / 692538347,
             -28693883.0 / 1125000000, 23124283.0 / 1800000000},
            {16016141.0 / 946692911, 0, 0, 61564180.0 / 158732637,
             22789713.0 / 633445777, 545815736.0 / 2771057229,
             -180193667.0 / 1043307555},
            {39632708.0 / 573591083, 0, 0, -433636366.0 / 683701615,
             -421739975.0 / 2616292301, 100302831.0 / 723423059,
             790204164.0 / 839813087, 800635310.0 / 3783071287},
            {246121993.0 / 1340847787, 0, 0, -37695042795.0 / 15268766246,
             -309121744.0 / 1061227803, -12992083.0 / 490766935,
             6005943493.0 / 2108947869, 393006217.0 / 1396673457,
             123872331.0 / 1001029789},
            {-1028468189.0 / 846180014, 0, 0, 8478235783.0 / 508512852,
             1311729495.0 / 1432422823, -10304129995.0 / 1701304382,
             -48777925059.0 / 3047939560, 15336726248.0 / 1032824649,
             -45442868181.0 / 3398467696, 3065993473.0 / 597172653},
            {185892177.0 / 718116043, 0, 0, -3185094517.0 / 667107341,
             -477755414.0 / 1098053517, -703635378.0 / 230739211,
             5731566787.0 / 1027545527, 5232866602.0 / 850066563,
             -4093664535.0 / 808688257, 3962137247.0 / 1805957418,
             65686358.0 / 487910083},
            {403863854.0 / 491063109, 0, 0, -5068492393.0 / 434740067,
             -411421997.0 / 543043805, 652783627.0 / 914296604,
             11173962825.0 / 925320556, -13158990841.0 / 6184727034,
             3936647629.0 / 1978049680, -160528059.0 / 685178525,
             248638103.0 / 1413531060, 0}}},
          {14005451.0 / 335480064, 0, 0, 0, 0, -59238493.0 / 1068277825,
           181606767.0 / 758867731, 561292985.0 / 797845732,
           -1041891430.0 / 1371343529, 760417239.0 / 1151165299,
           118820643.0 / 751138087, -528747749.0 / 2220607170, 1.0 / 4},
          {0, 1.0 / 18, 1.0 / 12, 1.0 / 8, 5.0 / 16, 3.0 / 8, 59.0 / 400,
           93.0 / 200, 5490023248.0 / 9719169821, 13.0 / 20,
           1201146811.0 / 1299019798, 1, 1}}}},
}};

namespace rk_methods_detail {

template <typename Rule> constexpr bool every_explicit_row(const Rule rule) {
  for (const rk_method_t &rk : rk_methods)
    if (std::holds_alternative<explicit_rk_t>(rk.scheme))
      if (!rule(std::get<explicit_rk_t>(rk.scheme)))
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

constexpr int peak_copies_of(const char *const method) {
  for (const rk_method_t &rk : rk_methods)
    if (names_equal(rk.name, method))
      if (std::holds_alternative<explicit_rk_t>(rk.scheme))
        return peak_copies(
            storage_plan(std::get<explicit_rk_t>(rk.scheme).tableau));
  return -1;
}

} // namespace rk_methods_detail

static_assert(
    rk_methods_detail::every_explicit_row([](const explicit_rk_t &scheme) {
      return is_explicit(scheme.tableau);
    }),
    "rk_methods: a tableau has 1 to max_stages stages, a[s][i] is "
    "zero for i >= s, and a, b and c are zero from nstages on");
static_assert(
    rk_methods_detail::every_explicit_row([](const explicit_rk_t &scheme) {
      return rows_sum_to_c(scheme.tableau);
    }),
    "rk_methods: each row s of a tableau's a must sum to c[s]");
static_assert(
    rk_methods_detail::every_explicit_row([](const explicit_rk_t &scheme) {
      return weights_sum_to_one(scheme.tableau);
    }),
    "rk_methods: a tableau's b must sum to 1");
static_assert(
    rk_methods_detail::every_explicit_row([](const explicit_rk_t &scheme) {
      return has_declared_order(scheme.tableau);
    }),
    "rk_methods: a tableau must satisfy the order condition of "
    "every rooted tree with at most its declared order of vertices");
static_assert(rk_methods_detail::names_distinct(),
              "rk_methods: rk_method finds a row by its name, compared "
              "case-insensitively, so no two rows may share a name");
static_assert(rk_methods_detail::peak_copies_of("Euler") == 0,
              "rk_methods: Euler must step with no copy of the state");
static_assert(rk_methods_detail::peak_copies_of("RK2") == 1,
              "rk_methods: RK2 must step holding only the old state");
static_assert(rk_methods_detail::peak_copies_of("RK3") == 3,
              "rk_methods: RK3 must step holding at most the old state, the "
              "weighted sum and k1");
static_assert(rk_methods_detail::peak_copies_of("SSPRK3") == 3,
              "rk_methods: SSPRK3 must step holding at most the old state, "
              "the weighted sum and k1");
static_assert(rk_methods_detail::peak_copies_of("RK4") == 2,
              "rk_methods: RK4 must step holding at most the old state and "
              "the weighted sum");
static_assert(rk_methods_detail::peak_copies_of("RKF78") == 9,
              "rk_methods: RKF78 must step holding at most the old state, "
              "the weighted sum and seven stage right-hand sides");
static_assert(rk_methods_detail::peak_copies_of("DP87") == 11,
              "rk_methods: DP87 must step holding at most the old state, the "
              "weighted sum and nine stage right-hand sides");

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
