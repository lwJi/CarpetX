#ifndef CARPETX_CARPETX_SCHEDULE_CLAUSES_HXX
#define CARPETX_CARPETX_SCHEDULE_CLAUSES_HXX

#include "internal.hxx"

#include "hierarchy/loops.hxx"

#include <carpetx.hxx>

#include <cctk.h>
#include <cctk_Schedule.h>

#include <ostream>
#include <tuple>
#include <vector>

namespace CarpetX {

mode_t decode_mode(const cFunctionData *restrict attribute);

enum class rdwr_t { read, write, invalid };

struct clause_t {
  gi_t gi;
  vi_t vi;
  tl_t tl;
  valid_t valid;

  friend bool operator==(const clause_t &x, const clause_t &y) {
    return std::make_tuple(x.gi, x.vi, x.tl, x.valid) ==
           std::make_tuple(y.gi, y.vi, y.tl, y.valid);
  }
  friend bool operator<(const clause_t &x, const clause_t &y) {
    return std::make_tuple(x.gi, x.vi, x.tl, x.valid) <
           std::make_tuple(y.gi, y.vi, y.tl, y.valid);
  }

  friend std::ostream &operator<<(std::ostream &os, const clause_t &cl) {
    return os << "clause_t{gi:" << cl.gi << ",vi:" << cl.vi << ",tl:" << cl.tl
              << ",valid:" << cl.valid << "}";
  }
};

std::vector<clause_t> decode_clauses(const cFunctionData *restrict attribute,
                                     rdwr_t rdwr);

void check_storage_clauses(const cFunctionData *restrict attribute);

} // namespace CarpetX

#endif
