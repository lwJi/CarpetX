#include <limits>
#include <stdexcept>
#include <type_traits>

namespace Arith {

namespace {

[[noreturn]] void on_overflow() {
  throw std::overflow_error{"integer overflow"};
}

template <typename I> constexpr bool would_add_overflow(I x, I y) noexcept {
  if constexpr (std::is_signed_v<I>) {
    using L = std::numeric_limits<I>;
    return (y > 0 && x > L::max() - y) || (y < 0 && x < L::min() - y);
  } else {
    return x > std::numeric_limits<I>::max() - y;
  }
}

template <typename I> constexpr bool would_sub_overflow(I x, I y) noexcept {
  if constexpr (std::is_signed_v<I>) {
    using L = std::numeric_limits<I>;
    return (y < 0 && x > L::max() + y) || (y > 0 && x < L::min() + y);
  } else {
    return x < y;
  }
}

template <typename I> constexpr bool would_mul_overflow(I x, I y) noexcept {
  using L = std::numeric_limits<I>;

  if (x == 0 || y == 0)
    return false;

  if constexpr (std::is_signed_v<I>) {
    if (x > 0) {
      if (y > 0) {
        return x > L::max() / y;
      } else {
        return y < L::min() / x;
      }
    } else {
      if (y > 0) {
        return x < L::min() / y;
      } else {
        return x < L::max() / y;
      }
    }
  } else {
    return x > L::max() / y;
  }
}

} // namespace

template <typename I> constexpr I checked_pos(I x) noexcept { return +x; }

template <typename I> constexpr I checked_neg(I x) {
  if constexpr (std::is_signed_v<I>) {
    if (x == std::numeric_limits<I>::min())
      on_overflow();
  } else {
    if (x != I{0})
      on_overflow();
  }
  return -x;
}

template <typename I> constexpr I checked_add(I x, I y) {
  if (would_add_overflow(x, y))
    on_overflow();
  return x + y;
}

template <typename I> constexpr I checked_sub(I x, I y) {
  if (would_sub_overflow(x, y))
    on_overflow();
  return x - y;
}

template <typename I> constexpr I checked_mul(I x, I y) {
  if (would_mul_overflow(x, y))
    on_overflow();
  return x * y;
}

template <typename I> constexpr I checked_div(I x, I y) {
  if (y == I{0})
    on_overflow();
  if constexpr (std::is_signed_v<I>)
    if (x == std::numeric_limits<I>::min() && y == I{-1})
      on_overflow();
  return x / y;
}

template <typename I> constexpr I checked_mod(I x, I y) {
  if (y == I{0})
    on_overflow();
  if constexpr (std::is_signed_v<I>)
    if (x == std::numeric_limits<I>::min() && y == I{-1})
      on_overflow();
  return x % y;
}

} // namespace Arith
