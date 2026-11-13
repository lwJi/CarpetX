#ifndef CARPETX_CARPETX_HIERARCHY_VALID_HXX
#define CARPETX_CARPETX_HIERARCHY_VALID_HXX

#include "internal.hxx"

#include "hierarchy/indices.hxx"
#include <carpetx.hxx>
#include <loop.hxx>
#include <tuple.hxx>

#include <AMReX_Box.H>
#include <AMReX_MultiFab.H>

#include <yaml-cpp/yaml.h>

#include <cstdint>
#include <deque>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <optional>
#include <source_location>
#include <sstream>
#include <string>
#include <vector>

namespace CarpetX {

YAML::Emitter &operator<<(YAML::Emitter &yaml, valid_t v);

class why_valid_t {
  valid_t valid;
  std::function<std::string()> why_int, why_outer, why_same_level,
      why_coarse_fine;

  void set(const valid_t &which, const valid_t &val,
           const std::function<std::string()> &why) {
    valid = (valid & ~which) | (val & which);
    if (which.valid_int)
      why_int = why;
    if (which.valid_outer)
      why_outer = why;
    if (which.valid_same_level)
      why_same_level = why;
    if (which.valid_coarse_fine)
      why_coarse_fine = why;
  }
  void set_invalid(const valid_t &which,
                   const std::function<std::string()> &why) {
    set(which, valid_t(false), why);
  }
  void set_valid(const valid_t &which,
                 const std::function<std::string()> &why) {
    set(which, valid_t(true), why);
  }
  friend class write_scope_t;
  friend class array_write_scope_t;

public:
  why_valid_t() = delete;
  why_valid_t(const std::function<std::string()> &why)
      : why_valid_t(false, why) {}
  why_valid_t(bool b, const std::function<std::string()> &why)
      : why_valid_t(valid_t(b), why) {}
  why_valid_t(const valid_t &val, const std::function<std::string()> &why)
      : valid(val), why_int(why), why_outer(why), why_same_level(why),
        why_coarse_fine(why) {}

  const valid_t &get() const { return valid; }

  std::string explanation() const;

  friend std::ostream &operator<<(std::ostream &os, const why_valid_t &why);
  operator std::string() const;
  friend YAML::Emitter &operator<<(YAML::Emitter &yaml, const why_valid_t &why);
};

////////////////////////////////////////////////////////////////////////////////

struct write_spec_t {
  gi_t gi;
  tl_t tl;
  std::optional<vi_t> vi;
  valid_t fills;
  valid_t keeps;
  std::string event;
};

struct require_spec_t {
  gi_t gi;
  tl_t tl;
  std::optional<vi_t> vi;
  valid_t where;
  std::string event;
  std::function<std::string()> hint;
};

struct invalidate_spec_t {
  gi_t gi;
  tl_t tl;
  std::optional<vi_t> vi;
  valid_t where;
  std::string event;
};

class write_scope_t;
[[nodiscard]] write_scope_t open_write(window_t window, write_spec_t spec);

class write_scope_t {
public:
  write_scope_t(write_scope_t &&other) noexcept;
  write_scope_t(const write_scope_t &) = delete;
  write_scope_t &operator=(const write_scope_t &) = delete;
  write_scope_t &operator=(write_scope_t &&) = delete;
  ~write_scope_t();

  amrex::MultiFab &
  mfab(level_t level,
       std::source_location site = std::source_location::current()) &;
  void
  mfab(level_t level,
       std::source_location site = std::source_location::current()) && = delete;

private:
  friend write_scope_t open_write(window_t window, write_spec_t spec);
  friend class write_scopes_t;
  write_scope_t(window_t window, write_spec_t spec, int iteration);
  window_t window;
  write_spec_t spec;
  int iteration;
  CCTK_INT epoch;
  bool live;
};

class write_scopes_t {
public:
  write_scopes_t() = default;
  write_scopes_t(const write_scopes_t &) = delete;
  write_scopes_t &operator=(const write_scopes_t &) = delete;
  ~write_scopes_t();

  void open(window_t window, write_spec_t spec);
  amrex::MultiFab &
  mfab(level_t level, gi_t gi, tl_t tl,
       std::source_location site = std::source_location::current()) &;
  void
  mfab(level_t level, gi_t gi, tl_t tl,
       std::source_location site = std::source_location::current()) && = delete;
  void close();

private:
  std::deque<write_scope_t> scopes;
};

void invalidate(window_t window, invalidate_spec_t spec);
void require(window_t window, require_spec_t spec);
void warn(window_t window, require_spec_t spec);

void loop_ghost_halves(
    const Loop::GridDescBase &grid, level_t level, gi_t gi,
    component_t component, valid_t where,
    const std::function<void(const Loop::PointDesc &)> &kernel);

int current_iteration();

class array_write_scope_t;
[[nodiscard]] array_write_scope_t open_array_write(write_spec_t spec);

class array_write_scope_t {
public:
  array_write_scope_t(array_write_scope_t &&other) noexcept;
  array_write_scope_t(const array_write_scope_t &) = delete;
  array_write_scope_t &operator=(const array_write_scope_t &) = delete;
  array_write_scope_t &operator=(array_write_scope_t &&) = delete;
  ~array_write_scope_t();

  void *data(std::source_location site = std::source_location::current()) &;
  void
  data(std::source_location site = std::source_location::current()) && = delete;

private:
  friend array_write_scope_t open_array_write(write_spec_t spec);
  array_write_scope_t(write_spec_t spec, int iteration);
  write_spec_t spec;
  int iteration;
  bool live;
};

void invalidate_array(invalidate_spec_t spec);
void require_array(require_spec_t spec);

////////////////////////////////////////////////////////////////////////////////

template <typename T> struct ipoison_t;

template <> struct ipoison_t<CCTK_COMPLEX> {
#if defined CCTK_REAL_PRECISION_4
  const std::uint32_t value[2] = {0xffc00000UL + 0xdead, 0xffc00000UL + 0xdead};
#elif defined CCTK_REAL_PRECISION_8
  const std::uint64_t value[2] = {0xfff8000000000000ULL + 0xdeadbeef,
                                  0xfff8000000000000ULL + 0xdeadbeef};
#endif
  static_assert(sizeof value == sizeof(CCTK_COMPLEX));
};

template <> struct ipoison_t<CCTK_REAL> {
#if defined CCTK_REAL_PRECISION_4
  const std::uint32_t value[1] = {0xffc00000UL + 0xdead};
#elif defined CCTK_REAL_PRECISION_8
  const std::uint64_t value[1] = {0xfff8000000000000ULL + 0xdeadbeef};
#endif
  static_assert(sizeof value == sizeof(CCTK_REAL));
};

template <> struct ipoison_t<CCTK_INT> {
  const std::uint32_t value[1] = {0xdeadbeef};
  static_assert(sizeof value == sizeof(CCTK_INT));
};

template <typename T> class poison_value_t {
public:
  poison_value_t() = default;

  CCTK_HOST CCTK_DEVICE bool is_poison(const T &val) const {
    typedef decltype(ipoison.value) array_type;
    typedef typename std::remove_extent<array_type>::type const_type;
    const size_t array_size = std::extent<array_type>::value;

    const_type *ival = reinterpret_cast<const_type *>(&val);
    bool retval = false;
    for (size_t i = 0; i < array_size; ++i)
      if (ival[i] == ipoison.value[i])
        retval = true;
    return retval;
  }

  void set_to_poison(T &val) const {
    std::memcpy(static_cast<void *>(&val), &ipoison.value, sizeof(val));
  }

  void set_to_poison(void *val, size_t count) const {
    set_to_poison(reinterpret_cast<T *>(val), count);
  }

  void set_to_poison(T *val, size_t count) const {
    for (size_t i = 0; i < count; ++i)
      set_to_poison(val[i]);
  }

private:
  const ipoison_t<T> ipoison;
};

} // namespace CarpetX

#endif
