#ifndef CARPETX_ALGO_ROOTS_HXX
#define CARPETX_ALGO_ROOTS_HXX

#include <mat.hxx>
#include <sum.hxx>
#include <vec.hxx>

#include <cctk.h>

#include <boost/math/policies/error_handling.hpp>
#include <boost/math/tools/roots.hpp>

#ifdef __HIPCC__
#include <hip/hip_runtime.h>
#endif

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cmath>
#include <limits>
#include <utility>

#if defined __CUDACC__
#define ALGO_DEVICE __device__
#define ALGO_HOST __host__
#elif defined __HIPCC__
#define ALGO_DEVICE __device__
#define ALGO_HOST __host__
#else
#define ALGO_DEVICE
#define ALGO_HOST
#endif

namespace Algo {

namespace {
template <typename T> constexpr ALGO_HOST ALGO_DEVICE void swap1(T &x, T &y) {
  T z = std::move(x);
  x = std::move(y);
  y = std::move(z);
}
} // namespace

namespace {
template <typename T>
constexpr ALGO_HOST ALGO_DEVICE T ldexp1(const T &x, const int n) {
  using std::ldexp;
  return ldexp(x, n);
}
} // namespace

namespace {
template <typename T>
constexpr ALGO_HOST ALGO_DEVICE T max1(const T &x, const T &y) {
  using std::max;
  return max(x, y);
}
} // namespace

namespace {
template <typename T>
constexpr ALGO_HOST ALGO_DEVICE T clamp1(const T &x, const T &lo, const T &hi) {
  using std::max, std::min;
  return min(max(x, lo), hi);
}
} // namespace

namespace {
template <typename T>
constexpr ALGO_HOST ALGO_DEVICE bool opposite_signs1(const T &x, const T &y) {
  return (x < 0 && y > 0) || (x > 0 && y < 0);
}
} // namespace

template <typename T> class eps_tolerance {
  T eps;

public:
  constexpr ALGO_HOST ALGO_DEVICE eps_tolerance()
      : eps(4 * std::numeric_limits<T>::epsilon()) {}
  explicit constexpr ALGO_HOST ALGO_DEVICE eps_tolerance(const int min_bits)
      : eps(max1(ldexp1(T(1), 1 - min_bits),
                 T(4 * std::numeric_limits<T>::epsilon()))) {}
  constexpr ALGO_HOST ALGO_DEVICE bool operator()(const T &x,
                                                  const T &y) const {
    using std::abs, std::min;
    return abs(x - y) <= eps * min(abs(x), abs(y));
  }
};

template <typename F, typename T>
std::pair<T, T> bisect(F &&f, T min, T max, int min_bits, int max_iters,
                       int &iters, bool &failed) {
  iters = 0;
  failed = false;
  assert(max_iters >= 0);
  if (max_iters <= 0) {
    failed = true;
    return {min, max};
  }
  std::uintmax_t max_iter = max_iters;
  try {
    auto res = boost::math::tools::bisect(
        std::forward<F>(f), min, max,
        boost::math::tools::eps_tolerance<T>(min_bits), max_iter);
    iters = max_iter;
    return res;
  } catch (const boost::math::evaluation_error &) {
    iters = max_iter;
    failed = true;
    return {min, max};
  }
}

template <typename F, typename T>
std::pair<T, T> bracket_and_solve_root(F &&f, T guess, T factor, bool rising,
                                       int min_bits, int max_iters, int &iters,
                                       bool &failed) {
  iters = 0;
  failed = false;
  assert(max_iters >= 0);
  if (max_iters <= 0) {
    failed = true;
    return {guess, guess};
  }
  std::uintmax_t max_iter = max_iters;
  try {
    auto res = boost::math::tools::bracket_and_solve_root(
        std::forward<F>(f), guess, factor, rising, eps_tolerance<T>(min_bits),
        max_iter);
    iters = max_iter;
    return res;
  } catch (const boost::math::evaluation_error &) {
    iters = max_iter;
    failed = true;
    return {guess, guess};
  }
}

// See <https://en.wikipedia.org/wiki/Brent%27s_method>
template <typename F, typename T>
inline CCTK_ATTRIBUTE_ALWAYS_INLINE ALGO_HOST ALGO_DEVICE std::pair<T, T>
brent(F f, T a, T b, int min_bits, int max_iters, int &iters, bool &failed) {
  using std::abs, std::min, std::max;

  const auto tol = eps_tolerance<T>(min_bits);

  iters = 0;
  failed = false;
  auto fa = f(a);
  auto fb = f(b);
  if (abs(fa) < abs(fb)) {
    swap1(a, b);
    swap1(fa, fb);
  }
  if (fb == 0)
    return {b, b};
  if (!opposite_signs1(fa, fb)) {
    failed = true;
    return {min(a, b), max(a, b)};
  }
  T c = a;
  auto fc = fa;
  bool mflag = true;
  T d{};

  while (fb != 0 && !tol(a, b) && iters < max_iters) {
    T s;
    if (fa != fc && fb != fc)
      s = (a * fb * fc) / ((fa - fb) * (fa - fc)) +
          (b * fa * fc) / ((fb - fa) * (fb - fc)) +
          (c * fa * fb) / ((fc - fa) * (fc - fb));
    else
      s = (a + b) / 2 - (fa + fb) / 2 * (b - a) / (fb - fa);
    T u = (3 * a + b) / 4;
    T v = b;
    if (u > v)
      swap1(u, v);
    bool cond1 = !(u <= s && s <= v);
    bool cond2 = mflag && abs(s - b) >= abs(b - c) / 2;
    bool cond3 = !mflag && abs(s - b) >= abs(c - d) / 2;
    bool cond4 = mflag && tol(c, b);
    bool cond5 = !mflag && tol(c, d);
    if (cond1 || cond2 || cond3 || cond4 || cond5) {
      s = (a + b) / 2;
      mflag = true;
    } else {
      mflag = false;
    }
    auto fs = f(s);
    d = c;
    c = b;
    fc = fb;
    if (opposite_signs1(fa, fs)) {
      b = s;
      fb = fs;
    } else {
      a = s;
      fa = fs;
    }
    assert(fa == 0 || fb == 0 || opposite_signs1(fa, fb));
    if (abs(fa) < abs(fb)) {
      swap1(a, b);
      swap1(fa, fb);
    }
    ++iters;
  }

  if (fb == 0)
    return {b, b};
  failed = !tol(a, b);
  return {min(a, b), max(a, b)};
}

template <typename F, typename T>
T newton_raphson(F &&f, T guess, T min, T max, int min_bits, int max_iters,
                 int &iters, bool &failed) {
  iters = 0;
  failed = false;
  assert(max_iters >= 0);
  if (max_iters <= 0) {
    failed = true;
    return guess;
  }
  std::uintmax_t max_iter = max_iters;
  try {
    auto res = boost::math::tools::newton_raphson_iterate(
        std::forward<F>(f), guess, min, max, min_bits, max_iter);
    iters = max_iter;
    return res;
  } catch (const boost::math::evaluation_error &) {
    iters = max_iter;
    failed = true;
    return guess;
  }
}

template <typename F, typename T>
T halley(F &&f, T guess, T min, T max, int min_bits, int max_iters, int &iters,
         bool &failed) {
  iters = 0;
  failed = false;
  assert(max_iters >= 0);
  if (max_iters <= 0) {
    failed = true;
    return guess;
  }
  std::uintmax_t max_iter = max_iters;
  try {
    auto res = boost::math::tools::halley_iterate(std::forward<F>(f), guess,
                                                  min, max, min_bits, max_iter);
    iters = max_iter;
    return res;
  } catch (const boost::math::evaluation_error &) {
    iters = max_iter;
    failed = true;
    return guess;
  }
}

template <typename F, typename T>
T schroder(F &&f, T guess, T min, T max, int min_bits, int max_iters,
           int &iters, bool &failed) {
  iters = 0;
  failed = false;
  assert(max_iters >= 0);
  if (max_iters <= 0) {
    failed = true;
    return guess;
  }
  std::uintmax_t max_iter = max_iters;
  try {
    auto res = boost::math::tools::schroder_iterate(
        std::forward<F>(f), guess, min, max, min_bits, max_iter);
    iters = max_iter;
    return res;
  } catch (const boost::math::evaluation_error &) {
    iters = max_iter;
    failed = true;
    return guess;
  }
}

template <typename F, typename T, int N>
inline CCTK_ATTRIBUTE_ALWAYS_INLINE ALGO_HOST ALGO_DEVICE Arith::vec<T, N>
newton_raphson_nd(F f, const Arith::vec<T, N> &guess,
                  const Arith::vec<T, N> &min, const Arith::vec<T, N> &max,
                  int min_bits, int max_iters, int &iters, bool &failed) {
  using vec = Arith::vec<T, N>;
  using mat = Arith::mat<T, N>;
  using std::isfinite;

  const auto tolfx = eps_tolerance<T>(min_bits);

  failed = true;
  vec x([&](int i) { return clamp1(guess(i), min(i), max(i)); });

  for (iters = 0; iters <= max_iters; ++iters) {
    const auto [fx0, jac0] = f(x);
    const vec fx = fx0;
    const mat jac = jac0;

    const T errfx = sumabs(fx);
    if (tolfx(1 + errfx, 1)) {
      failed = false;
      return x;
    }
    if (iters == max_iters)
      break;

    const T det_jac = calc_det(jac);
    if (!isfinite(det_jac) || det_jac == 0)
      break;
    const mat inv_jac = calc_inv(jac, det_jac);
    const vec dx([&](int i) {
      return -Arith::sum<N>([&](int j) { return inv_jac(i, j) * fx(j); });
    });
    const vec xnew([&](int i) { return clamp1(x(i) + dx(i), min(i), max(i)); });

    bool moved = false;
    for (int i = 0; i < N; ++i)
      moved |= xnew(i) != x(i);
    if (!moved)
      break;
    x = xnew;
  }

  return x;
}

} // namespace Algo

#endif
