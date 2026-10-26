#ifndef CARPETX_CARPETX_VALID_HXX
#define CARPETX_CARPETX_VALID_HXX

#include "internal.hxx"

#include <carpetx.hxx>
#include <tuple.hxx>

#include <AMReX_Box.H>

#include <yaml-cpp/yaml.h>
#include <zlib.h>

#include <cstdint>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

namespace CarpetX {

YAML::Emitter &operator<<(YAML::Emitter &yaml, valid_t v);

} // namespace CarpetX
namespace std {
using namespace CarpetX;
template <> struct equal_to<valid_t> {
  constexpr bool operator()(const valid_t &x, const valid_t &y) const {
    return std::make_tuple(x.valid_int, x.valid_outer, x.valid_ghosts) ==
           std::make_tuple(y.valid_int, y.valid_outer, y.valid_ghosts);
  }
};
template <> struct less<valid_t> {
  constexpr bool operator()(const valid_t &x, const valid_t &y) const {
    return std::make_tuple(x.valid_int, x.valid_outer, x.valid_ghosts) <
           std::make_tuple(y.valid_int, y.valid_outer, y.valid_ghosts);
  }
};
} // namespace std
namespace CarpetX {

class why_valid_t {
  valid_t valid;
  std::function<std::string()> why_int, why_outer, why_ghosts;

  void set(const valid_t &which, const valid_t &val,
           const std::function<std::string()> &why) {
    valid = (valid & ~which) | (val & which);
    if (which.valid_int)
      why_int = why;
    if (which.valid_outer)
      why_outer = why;
    if (which.valid_ghosts)
      why_ghosts = why;
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
  // The constructor that doesn't give a reason should never be called
  why_valid_t() = delete;
  why_valid_t(const std::function<std::string()> &why)
      : why_valid_t(false, why) {}
  why_valid_t(bool b, const std::function<std::string()> &why)
      : why_valid_t(valid_t(b), why) {}
  why_valid_t(const valid_t &val, const std::function<std::string()> &why)
      : valid(val), why_int(why), why_outer(why), why_ghosts(why) {}

  const valid_t &get() const { return valid; }

  std::string explanation() const;

  friend std::ostream &operator<<(std::ostream &os, const why_valid_t &why);
  operator std::string() const;
  friend YAML::Emitter &operator<<(YAML::Emitter &yaml, const why_valid_t &why);
};

////////////////////////////////////////////////////////////////////////////////

struct checksum_t {
  valid_t where;
  uLong crc;
  checksum_t() = default;
  inline checksum_t(const valid_t &where)
      : where(where), crc(crc32(0, nullptr, 0)) {}
  template <typename T> inline void add(const T &x) {
    crc = crc32(crc, static_cast<const Bytef *>(static_cast<const void *>(&x)),
                sizeof x);
  }

  friend bool operator==(const checksum_t &x, const checksum_t &y) {
    return x.where == y.where && x.crc == y.crc;
  }
  friend bool operator!=(const checksum_t &x, const checksum_t &y) {
    return !(x == y);
  }

  friend std::ostream &operator<<(std::ostream &os, const checksum_t &x) {
    return os << "checksum_t{where:" << x.where << ",crc:0x" << std::hex
              << std::setfill('0') << std::setw(8) << x.crc << "}";
  }
  operator std::string() const {
    std::ostringstream buf;
    buf << *this;
    return buf.str();
  }
};

struct tiletag_t {
  level_t level;
  int component;
  gi_t gi;
  vi_t vi;
  tl_t tl;

  friend auto operator<=>(const tiletag_t &, const tiletag_t &) = default;

  friend std::ostream &operator<<(std::ostream &os, const tiletag_t &x) {
    return os << "tiletag_t{"
              << "patch:" << x.level.patch() << ","
              << "level:" << x.level.index() << ","
              << "component:" << x.component << ","
              << "gi:" << x.gi << ","
              << "vi:" << x.vi << ","
              << "tl:" << x.tl << "}";
  }
  operator std::string() const {
    std::ostringstream buf;
    buf << *this;
    return buf.str();
  }
};

typedef std::map<tiletag_t, checksum_t> checksums_t;

struct active_levels_t;

checksums_t calculate_checksums(
    const active_levels_t &active_levels,
    const std::vector<std::vector<std::vector<valid_t> > > &will_write);
void check_checksums(const active_levels_t &active_levels,
                     const checksums_t &checksums,
                     const std::function<std::string()> &where);

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
[[nodiscard]] write_scope_t open_write(const active_levels_t &window,
                                       write_spec_t spec);

class write_scope_t {
public:
  write_scope_t(write_scope_t &&other) noexcept;
  write_scope_t(const write_scope_t &) = delete;
  write_scope_t &operator=(const write_scope_t &) = delete;
  write_scope_t &operator=(write_scope_t &&) = delete;
  ~write_scope_t();

private:
  friend write_scope_t open_write(const active_levels_t &window,
                                  write_spec_t spec);
  write_scope_t(const active_levels_t &window, write_spec_t spec,
                int iteration);
  window_t window;
  write_spec_t spec;
  int iteration;
  bool live;
};

void invalidate(const active_levels_t &window, invalidate_spec_t spec);
void require(const active_levels_t &window, require_spec_t spec);
void warn(const active_levels_t &window, require_spec_t spec);

class array_write_scope_t;
[[nodiscard]] array_write_scope_t open_array_write(write_spec_t spec);

class array_write_scope_t {
public:
  array_write_scope_t(array_write_scope_t &&other) noexcept;
  array_write_scope_t(const array_write_scope_t &) = delete;
  array_write_scope_t &operator=(const array_write_scope_t &) = delete;
  array_write_scope_t &operator=(array_write_scope_t &&) = delete;
  ~array_write_scope_t();

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

#endif // #ifndef CARPETX_CARPETX_VALID_HXX
