#include "dual.hxx"
#include "smallvector.hxx"
#include "spvect.hxx"

#include <cctk.h>
#include <cctk_Arguments.h>

#include <cassert>
#include <limits>

namespace Arith {

// #ifndef __CUDACC__

namespace {
void test_smallvector() {
  using R = CCTK_REAL;
  constexpr std::size_t N = 6;
  using VR = smallvector<R, N>;

  {
    VR x, y, z;
    x.emplace_back(1);
    x.emplace_back(2);
    x.emplace_back(3);
    y.emplace_back(12);
    y.emplace_back(14);
    z.emplace_back(20);
    z.emplace_back(24);
  }

  const VR n{0, 0, 0, 0, 0};
  const VR x{1, 2, 3, 0, 0};
  const VR y{0, 12, 0, 14, 0};
  const VR z{0, 20, 0, 0, 24};

  const R a = -2;
  const R b = -3;

  assert(n == n);

  assert(x == x);
  assert(x + n == x);
  assert(n + x == x);
  assert(+x == n + x);

  assert(x + (-x) == n);
  assert(-(-x) == x);
  assert(x - y == x + (-y));

  assert(y == y);
  assert(x + y == y + x);

  assert(z == z);
  assert((x + y) + z == x + (y + z));

  assert(0 * x == n);
  assert(1 * x == x);
  assert((-1) * x == -x);

  assert(a * x == x * a);
  assert((a + b) * x == a * x + b * x);
  assert((a * b) * x == a * (b * x));
  assert(a * (x + y) == a * x + a * y);
}

template <typename T> using smallvector1 = smallvector<T, 6>;

void test_spvect() {
  using R = CCTK_REAL;
  using VR = spvect<int, R, smallvector1>;

  const VR n;
  VR x, y, z;
  x.emplace_back(1, 1);
  x.emplace_back(2, 2);
  x.emplace_back(3, 3);
  y.emplace_back(2, 12);
  y.emplace_back(4, 14);
  z.emplace_back(1, 20);
  z.emplace_back(4, 24);

  const R a = -2;
  const R b = -3;

  assert(n == n);

  assert(x == x);
  assert(x + n == x);
  assert(n + x == x);
  assert(+x == n + x);

  assert(x + (-x) == n);
  assert(-(-x) == x);
  assert(x - y == x + (-y));

  assert(y == y);
  assert(x + y == y + x);

  assert(z == z);
  assert((x + y) + z == x + (y + z));

  assert(0 * x == n);
  assert(1 * x == x);
  assert((-1) * x == -x);

  assert(a * x == x * a);
  assert((a + b) * x == a * x + b * x);
  assert((a * b) * x == a * (b * x));
  assert(a * (x + y) == a * x + a * y);
}

void test_dual_spvect() {
  using R = CCTK_REAL;
  using VR = spvect<int, R, smallvector1>;
  using DR = dual<R, VR>;

  VR xv, yv;
  xv.emplace_back(0, 1);
  yv.emplace_back(1, 1);

  const DR x(3, xv); // x=3
  const DR y(4, yv); // y=4

  assert((x + y).val == 7);
  assert((x + y).eps == xv + yv);

  assert(pow2(x).val == 9);
  assert(pow2(x).eps == 2 * 3 * xv);

  assert(sqrt(pow2(x) + pow2(y)).val == 5);
  assert(maximum(abs(sqrt(pow2(x) + pow2(y)).eps - (3 * xv + 4 * yv) / 5)) <=
         10 * std::numeric_limits<R>::epsilon());
}

} // namespace

// #endif

extern "C" void Test_smallvector(CCTK_ARGUMENTS) {
  // #ifndef __CUDACC__
  CCTK_INFO("Test_smallvector");

  test_smallvector();
  test_spvect();
  test_dual_spvect();
  // #endif
}

} // namespace Arith
