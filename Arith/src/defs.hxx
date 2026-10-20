#ifndef CARPETX_ARITH_DEFS_HXX
#define CARPETX_ARITH_DEFS_HXX

#include <cctk.h>

#ifdef __HIPCC__
#include <hip/hip_runtime.h>
#endif

#include <cmath>
#include <limits>
#include <type_traits>

#ifdef CCTK_DEBUG
#define ARITH_INLINE
#else
#define ARITH_INLINE CCTK_ATTRIBUTE_ALWAYS_INLINE
#endif

#if defined __CUDACC__
#define ARITH_DEVICE __device__
#define ARITH_HOST __host__
#elif defined __HIPCC__
#define ARITH_DEVICE __device__
#define ARITH_HOST __host__
#else
#define ARITH_DEVICE
#define ARITH_HOST
#endif

namespace Arith {

template <typename T> struct zero;

template <> struct zero<bool> : std::integral_constant<bool, 0> {};
template <> struct zero<short> : std::integral_constant<short, 0> {};
template <> struct zero<int> : std::integral_constant<int, 0> {};
template <> struct zero<long> : std::integral_constant<long, 0> {};
template <> struct zero<long long> : std::integral_constant<long long, 0> {};

template <> struct zero<float> {
  typedef float value_type;
  static constexpr value_type value = 0;
  constexpr ARITH_INLINE ARITH_DEVICE ARITH_HOST operator value_type() const {
    return value;
  }
  constexpr ARITH_INLINE ARITH_DEVICE ARITH_HOST value_type operator()() const {
    return value;
  }
};

template <> struct zero<double> {
  typedef double value_type;
  static constexpr value_type value = 0;
  constexpr ARITH_INLINE ARITH_DEVICE ARITH_HOST operator value_type() const {
    return value;
  }
  constexpr ARITH_INLINE ARITH_DEVICE ARITH_HOST value_type operator()() const {
    return value;
  }
};

template <typename T> struct one;

template <> struct one<bool> : std::integral_constant<bool, 1> {};
template <> struct one<short> : std::integral_constant<short, 1> {};
template <> struct one<int> : std::integral_constant<int, 1> {};
template <> struct one<long> : std::integral_constant<long, 1> {};

template <> struct one<float> {
  typedef float value_type;
  static constexpr value_type value = 1;
  constexpr ARITH_INLINE ARITH_DEVICE ARITH_HOST operator value_type() const {
    return value;
  }
  constexpr ARITH_INLINE ARITH_DEVICE ARITH_HOST value_type operator()() const {
    return value;
  }
};

template <> struct one<double> {
  typedef double value_type;
  static constexpr value_type value = 1;
  constexpr ARITH_INLINE ARITH_DEVICE ARITH_HOST operator value_type() const {
    return value;
  }
  constexpr ARITH_INLINE ARITH_DEVICE ARITH_HOST value_type operator()() const {
    return value;
  }
};

template <typename T> struct nan;

template <>
struct nan<bool>
    : std::integral_constant<bool, std::numeric_limits<bool>::max()> {};
template <>
struct nan<short>
    : std::integral_constant<short, std::numeric_limits<short>::min()> {};
template <>
struct nan<int> : std::integral_constant<int, std::numeric_limits<int>::min()> {
};
template <>
struct nan<long>
    : std::integral_constant<long, std::numeric_limits<long>::min()> {};

template <> struct nan<float> {
  typedef float value_type;
  static constexpr value_type value =
      std::numeric_limits<value_type>::quiet_NaN();
  constexpr ARITH_INLINE ARITH_DEVICE ARITH_HOST operator value_type() const {
    return value;
  }
  constexpr ARITH_INLINE ARITH_DEVICE ARITH_HOST value_type operator()() const {
    return value;
  }
};

template <> struct nan<double> {
  typedef double value_type;
  static constexpr value_type value =
      std::numeric_limits<value_type>::quiet_NaN();
  constexpr ARITH_INLINE ARITH_DEVICE ARITH_HOST operator value_type() const {
    return value;
  }
  constexpr ARITH_INLINE ARITH_DEVICE ARITH_HOST value_type operator()() const {
    return value;
  }
};

template <
    int imin, int imax, int istep = 1, typename F,
    std::enable_if_t<(istep > 0 ? imin >= imax : imin <= imax)> * = nullptr>
constexpr ARITH_INLINE ARITH_DEVICE ARITH_HOST void unroll_for(const F &f) {
}

template <
    int imin, int imax, int istep = 1, typename F,
    std::enable_if_t<!(istep > 0 ? imin >= imax : imin <= imax)> * = nullptr>
constexpr ARITH_INLINE ARITH_DEVICE ARITH_HOST void unroll_for(const F &f) {
  f(imin);
  unroll_for<imin + istep, imax, istep>(f);
}

constexpr ARITH_INLINE ARITH_DEVICE ARITH_HOST bool all(bool x) { return x; }

constexpr ARITH_INLINE ARITH_DEVICE ARITH_HOST bool any(bool x) { return x; }

constexpr ARITH_INLINE ARITH_DEVICE ARITH_HOST bool
allisfinite(const float &x) {
  using std::isfinite;
  return isfinite(x);
}
constexpr ARITH_INLINE ARITH_DEVICE ARITH_HOST bool
allisfinite(const double &x) {
  using std::isfinite;
  return isfinite(x);
}
constexpr ARITH_INLINE ARITH_DEVICE ARITH_HOST bool anyisnan(const float &x) {
  using std::isnan;
  return isnan(x);
}
constexpr ARITH_INLINE ARITH_DEVICE ARITH_HOST bool anyisnan(const double &x) {
  using std::isnan;
  return isnan(x);
}

template <typename T, typename U>
constexpr ARITH_INLINE ARITH_DEVICE ARITH_HOST auto if_else(bool c, const T &x,
                                                            const U &y) {
  return c ? x : y;
}

constexpr ARITH_INLINE ARITH_DEVICE ARITH_HOST int bitsign(bool c) {
  return if_else(c, -1, 1);
}
constexpr ARITH_INLINE ARITH_DEVICE ARITH_HOST int bitsign(int i) {
  return bitsign(i % 2 != 0);
}

template <typename T>
constexpr ARITH_INLINE ARITH_DEVICE ARITH_HOST T flipsign(const T &x,
                                                          const T &y) {
  using std::copysign;
  return copysign(T(1), y) * x;
}

template <typename T>
constexpr ARITH_INLINE ARITH_DEVICE ARITH_HOST T inv(const T &x) {
  return T(1) / x;
}

template <typename T>
constexpr ARITH_INLINE ARITH_DEVICE ARITH_HOST T max1(const T &x, const T &y) {
  using std::max;
  return if_else(x != x, x, if_else(y != y, y, max(x, y)));
}

template <typename T>
constexpr ARITH_INLINE ARITH_DEVICE ARITH_HOST T maxabs(const T &x) {
  using std::abs;
  return abs(x);
}

template <typename T>
constexpr ARITH_INLINE ARITH_DEVICE ARITH_HOST T sumabs(const T &x) {
  using std::abs;
  return abs(x);
}

template <typename T>
constexpr ARITH_INLINE ARITH_DEVICE ARITH_HOST T min1(const T &x, const T &y) {
  using std::min;
  return if_else(x != x, x, if_else(y != y, y, min(x, y)));
}

template <typename T>
constexpr ARITH_INLINE ARITH_DEVICE ARITH_HOST T muladd(const T &x, const T &y,
                                                        const T &z) {
  return x * y + z;
}
template <typename T>
constexpr ARITH_INLINE ARITH_DEVICE ARITH_HOST T mulsub(const T &x, const T &y,
                                                        const T &z) {
  return x * y - z;
}
template <typename T>
constexpr ARITH_INLINE ARITH_DEVICE ARITH_HOST T negmuladd(const T &x,
                                                           const T &y,
                                                           const T &z) {
  return -(x * y) + z;
}
template <typename T>
constexpr ARITH_INLINE ARITH_DEVICE ARITH_HOST T negmulsub(const T &x,
                                                           const T &y,
                                                           const T &z) {
  return -(x * y) - z;
}

constexpr ARITH_INLINE ARITH_DEVICE ARITH_HOST int factorial(int n) {
  int r = 1;
  while (n > 0)
    r *= n--;
  return r;
}

namespace detail {
template <typename T>
constexpr ARITH_INLINE ARITH_DEVICE ARITH_HOST T pown(const T &x, int n) {
  T r{1};
  T y{x};
  while (n) {
    if (n & 1)
      r *= y;
    y *= y;
    n >>= 1;
  }
  return r;
}
} // namespace detail

template <typename T>
constexpr ARITH_INLINE ARITH_DEVICE ARITH_HOST T pown(const T &x, const int n) {
  return n >= 0 ? detail::pown(x, n) : 1 / detail::pown(x, -n);
}

template <typename T>
constexpr ARITH_INLINE ARITH_DEVICE ARITH_HOST T pow2(const T &x) {
  return x * x;
}

template <typename T, typename I>
constexpr ARITH_INLINE ARITH_DEVICE ARITH_HOST T
lincom(const I &i0, const T &x0, const I &i1, const T &x1, const I &i) {
  return T(i - i1) / T(i0 - i1) * x0 + T(i - i0) / T(i1 - i0) * x1;
}

} // namespace Arith

#endif
