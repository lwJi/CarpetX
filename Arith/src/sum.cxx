#include "sum.hxx"

#include <cctk.h>

#include <algorithm>
#include <cstdlib>
#include <functional>

namespace Arith {

namespace {
template <typename T> constexpr T mid(T i, T j, T k) {
  if (i <= j && j <= k)
    return j;
  if (j <= i && i <= k)
    return i;
  if (i <= k && k <= j)
    return k;
  if (k <= i && i <= j)
    return i;
  if (j <= k && k <= i)
    return k;
  if (k <= j && j <= i)
    return j;
  std::abort();
}
} // namespace

namespace {
constexpr CCTK_REAL f0(int i) { return 0; }
constexpr CCTK_REAL f1(int i) { return 1; }
constexpr CCTK_REAL fi(int i) { return i; }

constexpr CCTK_REAL g0(int i, int j) { return 0; }
constexpr CCTK_REAL g1(int i, int j) { return 1; }
constexpr CCTK_REAL gi(int i, int j) { return i; }
constexpr CCTK_REAL gj(int i, int j) { return j; }
constexpr CCTK_REAL gij(int i, int j) { return i + 10 * j; }
constexpr CCTK_REAL gi_symm(int i, int j) {
  using std::min;
  return min(i, j);
}
constexpr CCTK_REAL gj_symm(int i, int j) {
  using std::max;
  return max(i, j);
}
constexpr CCTK_REAL gij_symm(int i, int j) {
  using std::max, std::min;
  return min(i, j) + 10 * max(i, j);
}

constexpr CCTK_REAL h0(int i, int j, int k) { return 0; }
constexpr CCTK_REAL h1(int i, int j, int k) { return 1; }
constexpr CCTK_REAL hi(int i, int j, int k) { return i; }
constexpr CCTK_REAL hj(int i, int j, int k) { return j; }
constexpr CCTK_REAL hk(int i, int j, int k) { return k; }
constexpr CCTK_REAL hijk(int i, int j, int k) { return i + 10 * j + 100 * k; }
constexpr CCTK_REAL hi_symm(int i, int j, int k) {
  using std::min;
  return min(i, min(j, k));
}
constexpr CCTK_REAL hj_symm(int i, int j, int k) { return mid(i, j, k); }
constexpr CCTK_REAL hk_symm(int i, int j, int k) {
  using std::max;
  return max(i, max(j, k));
}
constexpr CCTK_REAL hijk_symm(int i, int j, int k) {
  using std::max, std::min;
  return min(i, min(j, k)) + 10 * mid(i, j, k) + 100 * max(i, max(j, k));
}
} // namespace

void TestSum() {
  constexpr std::equal_to<CCTK_REAL> eq;

  static_assert(eq(sum<3>(f0), 0.0));
  static_assert(eq(sum<3>(f1), 3.0));
  static_assert(eq(sum<3>(fi), 3.0));

  static_assert(eq(sum<4>(f0), 0.0));
  static_assert(eq(sum<4>(f1), 4.0));
  static_assert(eq(sum<4>(fi), 6.0));

  static_assert(eq(sum<3>(g0), 0.0));
  static_assert(eq(sum<3>(g1), 9.0));
  static_assert(eq(sum<3>(gi), 9.0));
  static_assert(eq(sum<3>(gj), 9.0));
  static_assert(eq(sum<3>(gij), 99.0));
  static_assert(eq(sum<3>(gi_symm), 5.0));
  static_assert(eq(sum<3>(gj_symm), 13.0));
  static_assert(eq(sum<3>(gij_symm), 135.0));

  static_assert(eq(sum<3>(h0), 0.0));
  static_assert(eq(sum<3>(h1), 27.0));
  static_assert(eq(sum<3>(hi), 27.0));
  static_assert(eq(sum<3>(hj), 27.0));
  static_assert(eq(sum<3>(hk), 27.0));
  static_assert(eq(sum<3>(hijk), 2997.0));
  static_assert(eq(sum<3>(hi_symm), 9.0));
  static_assert(eq(sum<3>(hj_symm), 27.0));
  static_assert(eq(sum<3>(hk_symm), 45.0));
  static_assert(eq(sum<3>(hijk_symm), 4779.0));

  static_assert(eq(sum_symm<3>(g0), 0.0));
  static_assert(eq(sum_symm<3>(g1), 9.0));
  static_assert(eq(sum_symm<3>(gi_symm), 5.0));
  static_assert(eq(sum_symm<3>(gj_symm), 13.0));
  static_assert(eq(sum_symm<3>(gij_symm), 135.0));

  static_assert(eq(sum_symm<3>(h0), 0.0));
  static_assert(eq(sum_symm<3>(h1), 27.0));
  static_assert(eq(sum_symm<3>(hi_symm), 9.0));
  static_assert(eq(sum_symm<3>(hj_symm), 27.0));
  static_assert(eq(sum_symm<3>(hk_symm), 45.0));
  static_assert(eq(sum_symm<3>(hijk_symm), 4779.0));
}

} // namespace Arith
