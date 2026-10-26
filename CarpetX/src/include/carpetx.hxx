#ifndef CARPETX_CARPETX_INCLUDE_CARPETX_HXX
#define CARPETX_CARPETX_INCLUDE_CARPETX_HXX

#include <defs.hxx>
#include <loop.hxx>
#include <rational.hxx>
#include <vect.hxx>

#include <cctk.h>

#include <AMReX_Geometry.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Tuple.H>

#ifdef __CUDACC__
#include <nvtx3/nvToolsExt.h>
#endif

#include <array>
#include <cmath>
#include <compare>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <ostream>
#include <span>
#include <string>
#include <tuple>
#include <vector>

namespace CarpetX {

using rat64 = Arith::rational<std::int64_t>;

template <class Derived> struct index_t {
  constexpr explicit index_t(const int i) : i_(i) {}
  constexpr int index() const { return i_; }
  friend constexpr auto operator<=>(index_t, index_t) = default;

private:
  int i_;
};

template <class Derived>
std::ostream &operator<<(std::ostream &os, const index_t<Derived> i) {
  return os << i.index();
}

template <class Derived> struct index_hash {
  std::size_t operator()(const index_t<Derived> i) const noexcept {
    return std::hash<int>()(i.index());
  }
};

struct gi_t : index_t<gi_t> {
  using index_t::index_t;
};

enum class symmetry_t {
  none,
  interpatch,
  periodic,
  reflection,
};
std::ostream &operator<<(std::ostream &os, symmetry_t symmetry);

struct level_t {
  int patch, level;
};

struct group_t {
  level_t level;
  gi_t gi;
};

int num_patches();
int num_levels(int patch);

level_t parent(level_t level);
rat64 iteration(level_t level);
rat64 delta_iteration(level_t level);
const amrex::Geometry &geometry(level_t level);
std::array<std::array<symmetry_t, Loop::dim>, 2> symmetries(level_t level);

int numvars(group_t group);
std::array<int, Loop::dim> indextype(group_t group);
std::array<int, Loop::dim> nghostzones(group_t group);

struct window_t {
  int min_level, max_level;
  int min_patch, max_patch;

  void loop_serially(const std::function<void(level_t level)> &kernel) const;
  void
  loop_coarse_to_fine(const std::function<void(level_t level)> &kernel) const;
  void loop_parallel(
      const std::function<void(level_t level, int index, int component,
                               const cGH *cctkGH)> &kernel) const;
};

window_t active_window();
window_t all_levels();

struct valid_t {
  bool valid_int, valid_outer, valid_ghosts;
  constexpr valid_t() : valid_t(false) {}

  explicit constexpr valid_t(bool b)
      : valid_int(b), valid_outer(b), valid_ghosts(b) {}
  friend constexpr valid_t operator~(const valid_t &x) {
    valid_t r;
    r.valid_int = !x.valid_int;
    r.valid_outer = !x.valid_outer;
    r.valid_ghosts = !x.valid_ghosts;
    return r;
  }
  friend constexpr valid_t operator&(const valid_t &x, const valid_t &y) {
    valid_t r;
    r.valid_int = x.valid_int && y.valid_int;
    r.valid_outer = x.valid_outer && y.valid_outer;
    r.valid_ghosts = x.valid_ghosts && y.valid_ghosts;
    return r;
  }
  friend constexpr valid_t operator|(const valid_t &x, const valid_t &y) {
    valid_t r;
    r.valid_int = x.valid_int || y.valid_int;
    r.valid_outer = x.valid_outer || y.valid_outer;
    r.valid_ghosts = x.valid_ghosts || y.valid_ghosts;
    return r;
  }
  valid_t &operator&=(const valid_t &x) { return *this = *this & x; }
  valid_t &operator|=(const valid_t &x) { return *this = *this | x; }

  constexpr bool valid_all() const {
    return valid_int && valid_outer && valid_ghosts;
  }
  constexpr bool valid_any() const {
    return valid_int || valid_outer || valid_ghosts;
  }

  friend bool operator==(const valid_t &x, const valid_t &y) {
    return std::make_tuple(x.valid_int, x.valid_outer, x.valid_ghosts) ==
           std::make_tuple(y.valid_int, y.valid_outer, y.valid_ghosts);
  }
  friend bool operator!=(const valid_t &x, const valid_t &y) {
    return !(x == y);
  }
  friend bool operator<(const valid_t &x, const valid_t &y) {
    return std::make_tuple(x.valid_int, x.valid_outer, x.valid_ghosts) <
           std::make_tuple(y.valid_int, y.valid_outer, y.valid_ghosts);
  }

  std::string explanation() const;

  friend std::ostream &operator<<(std::ostream &os, const valid_t v);
  operator std::string() const;
};

inline constexpr valid_t make_valid_int() {
  valid_t valid;
  valid.valid_int = true;
  return valid;
}
inline constexpr valid_t make_valid_outer() {
  valid_t valid;
  valid.valid_outer = true;
  return valid;
}
inline constexpr valid_t make_valid_ghosts() {
  valid_t valid;
  valid.valid_ghosts = true;
  return valid;
}
inline constexpr valid_t make_valid_all() { return ~valid_t(); }

const amrex::MultiFab &data(group_t group, int tl);

class write_t {
public:
  write_t(write_t &&other) noexcept;
  write_t(const write_t &) = delete;
  write_t &operator=(const write_t &) = delete;
  write_t &operator=(write_t &&) = delete;
  ~write_t();

  amrex::MultiFab &mfab();

private:
  struct impl;
  std::unique_ptr<impl> p;
  explicit write_t(std::unique_ptr<impl> p) noexcept;
  friend write_t write(group_t group, int tl, valid_t leaves_valid,
                       const char *what);
};

write_t write(group_t group, int tl, valid_t leaves_valid, const char *what);

class scratch_t {
public:
  scratch_t(scratch_t &&other) noexcept;
  scratch_t(const scratch_t &) = delete;
  scratch_t &operator=(const scratch_t &) = delete;
  scratch_t &operator=(scratch_t &&) = delete;
  ~scratch_t();

  amrex::MultiFab &mfab();
  const amrex::MultiFab &mfab() const;

private:
  struct impl;
  std::unique_ptr<impl> p;
  explicit scratch_t(std::unique_ptr<impl> p) noexcept;
  friend scratch_t scratch(group_t group);
};

scratch_t scratch(group_t group);

valid_t valid(group_t group, int tl, int vi);
void require_valid(group_t group, int tl, valid_t where, const char *why);
void invalidate(window_t window, std::span<const gi_t> groups, int tl,
                const char *why);

void sync(window_t window, std::span<const gi_t> groups);
void sync_ghosts_only(window_t window, std::span<const gi_t> groups,
                      int tl = -1);
void prolongate_ghosts(window_t window, std::span<const gi_t> groups,
                       int tl = -1);
void prolongate_aligned(window_t window, std::span<const gi_t> groups,
                        int tl = -1);
void restrict_no_poison(window_t window, std::span<const gi_t> groups);
void synchronize();

template <typename T, int D> struct reduction {
  T min, max, sum, sum2;
  T vol, maxabs, sumabs, sum2abs;
  Arith::vect<T, D> minloc, maxloc, sumloc;

  using tuple_type = amrex::GpuTuple<T, T, T, T, T, T, T, T, T, T, T>;
  constexpr reduction(tuple_type);
  constexpr operator tuple_type() const;

  constexpr reduction();
  constexpr reduction(const Arith::vect<T, D> &p, const T &V, const T &x);
  constexpr reduction(const reduction &x, const reduction &y);
  constexpr reduction operator+(const reduction &y) const {
    return reduction(*this, y);
  }
  constexpr reduction &operator+=(const reduction &y) noexcept {
    return *this = *this + y;
  }

  constexpr T avg() const noexcept { return sum / vol; }
  constexpr T sdv() const noexcept {
    return sqrt(
        Arith::max1(T(0), sum2 / vol - Arith::pow2(sum) / Arith::pow2(vol)));
  }
  constexpr T norm0() const noexcept { return vol; }
  constexpr T norm1() const noexcept { return sumabs / vol; }
  constexpr T norm2() const noexcept { return sqrt(sum2abs / vol); }
  constexpr T norm_inf() const noexcept { return maxabs; }

  constexpr Arith::vect<T, D> com() const noexcept { return sumloc / sum; }

  template <typename T1, int D1>
  friend std::ostream &operator<<(std::ostream &os,
                                  const reduction<T1, D1> &red);
};

template <typename T, int D>
constexpr reduction<T, D>::reduction(tuple_type tuple)
    : min(amrex::get<0>(tuple)), max(amrex::get<1>(tuple)),
      sum(amrex::get<2>(tuple)), sum2(amrex::get<3>(tuple)),
      vol(amrex::get<4>(tuple)), maxabs(amrex::get<5>(tuple)),
      sumabs(amrex::get<6>(tuple)), sum2abs(amrex::get<7>(tuple)),
      minloc(Arith::vect<T, D>::pure(0.0 / 0.0)),
      maxloc(Arith::vect<T, D>::pure(0.0 / 0.0)),
      sumloc{amrex::get<8>(tuple), amrex::get<9>(tuple),
             amrex::get<10>(tuple)} {
  static_assert(D == 3, "`tuple_type` assumes that `sumloc` has 3 components");
}

template <typename T, int D>
constexpr reduction<T, D>::operator reduction<T, D>::tuple_type() const {
  static_assert(D == 3, "`tuple_type` assumes that `sumloc` has 3 components");
  return tuple_type{min,    max,     sum,       sum2,      vol,      maxabs,
                    sumabs, sum2abs, sumloc[0], sumloc[1], sumloc[2]};
}

template <typename T, int D>
constexpr reduction<T, D>::reduction()
    : min(1.0 / 0.0), max(-1.0 / 0.0), sum(0.0), sum2(0.0), vol(0.0),
      maxabs(0.0), sumabs(0.0), sum2abs(0.0),
      minloc(Arith::vect<T, D>::pure(0.0 / 0.0)),
      maxloc(Arith::vect<T, D>::pure(0.0 / 0.0)),
      sumloc(Arith::vect<T, D>::pure(0.0)) {}

template <typename T, int D>
constexpr reduction<T, D>::reduction(const Arith::vect<T, D> &p, const T &V,
                                     const T &x)
    : min(x), max(x), sum(V * x), sum2(V * Arith::pow2(x)), vol(V),
      maxabs(fabs(x)), sumabs(V * fabs(x)), sum2abs(V * Arith::pow2(fabs(x))),
      minloc(p), maxloc(p), sumloc(V * x * p) {}

template <typename T, int D>
constexpr reduction<T, D>::reduction(const reduction &x, const reduction &y)
    : min(Arith::min1(x.min, y.min)), max(Arith::max1(x.max, y.max)),
      sum(x.sum + y.sum), sum2(x.sum2 + y.sum2), vol(x.vol + y.vol),
      maxabs(Arith::max1(x.maxabs, y.maxabs)), sumabs(x.sumabs + y.sumabs),
      sum2abs(x.sum2abs + y.sum2abs),
      minloc(x.min <= y.min ? x.minloc : y.minloc),
      maxloc(x.max >= y.max ? x.maxloc : y.maxloc),
      sumloc(x.sumloc + y.sumloc) {}

template <typename T, int D>
std::ostream &operator<<(std::ostream &os, const reduction<T, D> &red) {
  return os << "reduction{\n"
            << "  min:      " << red.min << "\n"
            << "  max:      " << red.max << "\n"
            << "  sum:      " << red.sum << "\n"
            << "  sum2:     " << red.sum2 << "\n"
            << "  vol:      " << red.vol << "\n"
            << "  maxabs:   " << red.maxabs << "\n"
            << "  sumabs:   " << red.sumabs << "\n"
            << "  sum2abs:  " << red.sum2abs << "\n"
            << "  minloc:   " << red.minloc << "\n"
            << "  maxloc:   " << red.maxloc << "\n"
            << "  sumloc:   " << red.sumloc << "\n"
            << "  avg:      " << red.avg() << "\n"
            << "  sdv:      " << red.sdv() << "\n"
            << "  norm1:    " << red.norm1() << "\n"
            << "  norm2:    " << red.norm2() << "\n"
            << "  norm_inf: " << red.norm_inf() << "\n"
            << "  com:      " << red.com() << "\n"
            << "}\n";
}

reduction<CCTK_REAL, Loop::dim> reduce(gi_t gi, int vi, int tl);

class Interval;

class Timer {
public:
  std::string name;

private:
  int handle;
#ifdef __CUDACC__
  nvtxRangeId_t range;
#endif

public:
  Timer() = delete;
  Timer(const Timer &) = delete;
  Timer &operator=(const Timer &) = delete;
  Timer(Timer &&) = default;
  Timer &operator=(Timer &&) = default;

  Timer(const std::string &name);

  void start();
  void stop();

  void print() const;
};

class Interval {
  Timer &timer;

public:
  Interval() = delete;
  Interval(const Interval &) = delete;
  Interval(Interval &&) = delete;
  Interval &operator=(const Interval &) = delete;
  Interval &operator=(Interval &&) = delete;

  Interval(Timer &timer) : timer(timer) { timer.start(); }
  ~Interval() { timer.stop(); }
};

} // namespace CarpetX

template <>
struct std::hash<CarpetX::gi_t> : CarpetX::index_hash<CarpetX::gi_t> {};

#endif
