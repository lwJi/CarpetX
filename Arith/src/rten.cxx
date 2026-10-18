#include "rten.hxx"

#include <cctk.h>

#include <functional>

using namespace std;

namespace Arith {

template <typename T, typename U> constexpr bool eq(const T &x, const U &y) {
  using std::isnan;
  return equal_to<CCTK_REAL>()(x, y) || (isnan(x) && isnan(y));
}

template <typename T, int D>
constexpr bool eqm(const rten<T, D> &x, const rten<T, D> &y) {
  using std::isnan;
  return equal_to<rten<T, D> >()(x, y) || (isnan(x) && isnan(y));
}

void TestRten() {
  using R4 = rten<CCTK_REAL, 4>;

  constexpr R4 r4 = {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                     12, 13, 14, 15, 16, 17, 18, 19, 20, 21};
  static_assert(eq(r4(0, 1, 0, 1), 1));
  static_assert(eq(r4(0, 1, 0, 2), 2));
  static_assert(eq(r4(0, 1, 0, 3), 3));
  static_assert(eq(r4(0, 1, 1, 2), 4));
  static_assert(eq(r4(0, 1, 1, 3), 5));
  static_assert(eq(r4(0, 1, 2, 3), 6));
  static_assert(eq(r4(0, 2, 0, 2), 7));
  static_assert(eq(r4(0, 2, 0, 3), 8));
  static_assert(eq(r4(0, 2, 1, 2), 9));
  static_assert(eq(r4(0, 2, 1, 3), 10));
  static_assert(eq(r4(0, 2, 2, 3), 11));
  static_assert(eq(r4(0, 3, 0, 3), 12));
  static_assert(eq(r4(0, 3, 1, 2), 13));
  static_assert(eq(r4(0, 3, 1, 3), 14));
  static_assert(eq(r4(0, 3, 2, 3), 15));
  static_assert(eq(r4(1, 2, 1, 2), 16));
  static_assert(eq(r4(1, 2, 1, 3), 17));
  static_assert(eq(r4(1, 2, 2, 3), 18));
  static_assert(eq(r4(1, 3, 1, 3), 19));
  static_assert(eq(r4(1, 3, 2, 3), 20));
  static_assert(eq(r4(2, 3, 2, 3), 21));
}

} // namespace Arith
