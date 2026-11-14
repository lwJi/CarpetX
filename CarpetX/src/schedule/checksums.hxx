#ifndef CARPETX_CARPETX_SCHEDULE_CHECKSUMS_HXX
#define CARPETX_CARPETX_SCHEDULE_CHECKSUMS_HXX

#include "internal.hxx"

#include <carpetx.hxx>

#include <zlib.h>

#include <compare>
#include <functional>
#include <iomanip>
#include <map>
#include <ostream>
#include <sstream>
#include <string>
#include <vector>

namespace CarpetX {

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
  tile_t tile;
  gi_t gi;
  vi_t vi;
  tl_t tl;

  friend auto operator<=>(const tiletag_t &, const tiletag_t &) = default;
  friend bool operator==(const tiletag_t &, const tiletag_t &) = default;

  friend std::ostream &operator<<(std::ostream &os, const tiletag_t &x) {
    return os << "tiletag_t{"
              << "patch:" << x.level.patch() << ","
              << "level:" << x.level.index() << ","
              << "component:" << x.tile.component() << ","
              << "tile:" << x.tile.index() << ","
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

checksums_t calculate_checksums(
    window_t window,
    const std::vector<std::vector<std::vector<valid_t> > > &will_write);
void check_checksums(window_t window, const checksums_t &checksums,
                     const std::function<std::string()> &where);

} // namespace CarpetX

#endif
