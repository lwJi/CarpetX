#include "schedule/clauses.hxx"
#include "hierarchy/driver.hxx"
#include <carpetx.hxx>

#include <cctk.h>
#include <cctk_Schedule.h>

#include <cassert>
#include <optional>
#include <ostream>
#include <vector>

namespace CarpetX {

mode_t decode_mode(const cFunctionData *restrict attribute) {
  bool local_mode = attribute->local;
  bool level_mode = attribute->level;
  bool global_mode = attribute->global;
  bool meta_mode = attribute->meta;
  assert(int(local_mode) + int(level_mode) + int(global_mode) +
             int(meta_mode) <=
         1);
  if (attribute->local)
    return mode_t::local;
  if (attribute->level)
    return mode_t::level;
  if (attribute->global)
    return mode_t::global;
  if (attribute->meta)
    return mode_t::meta;
  return mode_t::local;
}

std::ostream &operator<<(std::ostream &os, const rdwr_t rdwr) {
  switch (rdwr) {
  case rdwr_t::read:
    return os << "read";
  case rdwr_t::write:
    return os << "write";
  case rdwr_t::invalid:
    return os << "invalid";
  default:
    assert(0);
  }
}

std::vector<clause_t> decode_clauses(const cFunctionData *restrict attribute,
                                     const rdwr_t rdwr) {
  std::vector<clause_t> result;
  result.reserve(attribute->n_RDWR);
  for (int n = 0; n < attribute->n_RDWR; ++n) {
    const RDWR_entry &restrict RDWR = attribute->RDWR[n];
    const auto [gi, vi] = variable(var_t{RDWR.varindex});
    const tl_t tl{RDWR.timelevel};
    int where;
    switch (rdwr) {
    case rdwr_t::read:
      where = RDWR.where_rd;
      break;
    case rdwr_t::write:
      where = RDWR.where_wr;
      break;
    case rdwr_t::invalid:
      where = RDWR.where_inv;
      break;
    default:
      assert(0);
    }
    if (where == 0)
      continue;
    valid_t valid;
    valid.valid_int = where & CCTK_VALID_INTERIOR;
    valid.valid_outer = where & CCTK_VALID_BOUNDARY;
    valid.valid_ghosts = where & CCTK_VALID_GHOSTS;
    result.push_back({gi, vi, tl, valid});
  }
  return result;
}

void check_storage_clauses(const cFunctionData *restrict attribute) {
  int offender_count = 0;

  const auto group_inactive = [](const gi_t gi) {
    return !driver().record(gi).has_storage();
  };

  const auto report = [&](const char *clause_kind, const gi_t gi,
                          const std::optional<vi_t> vi, const tl_t tl) {
    ++offender_count;
    const char *const groupname = CCTK_FullGroupName(gi.index());
    if (vi) {
      const char *const varname =
          CCTK_VarName(driver().record(gi).var(*vi).index());
      CCTK_VWARN(CCTK_WARN_ALERT,
                 "Storage check: %s::%s (%s) clause %s references "
                 "group %s variable %s timelevel %d without active STORAGE",
                 attribute->thorn, attribute->routine, attribute->where,
                 clause_kind, groupname ? groupname : "<unknown>",
                 varname ? varname : "<unknown>", tl.index());
    } else {
      CCTK_VWARN(CCTK_WARN_ALERT,
                 "Storage check: %s::%s (%s) clause %s references "
                 "group %s without active STORAGE",
                 attribute->thorn, attribute->routine, attribute->where,
                 clause_kind, groupname ? groupname : "<unknown>");
    }
  };

  for (int n = 0; n < attribute->n_RDWR; ++n) {
    const RDWR_entry &entry = attribute->RDWR[n];
    if (entry.where_rd == 0 && entry.where_wr == 0 && entry.where_inv == 0)
      continue;
    const auto [gi, vi] = variable(var_t{entry.varindex});
    if (!group_inactive(gi))
      continue;
    const tl_t tl{entry.timelevel};
    if (entry.where_rd != 0)
      report("READS", gi, vi, tl);
    if (entry.where_wr != 0)
      report("WRITES", gi, vi, tl);
    if (entry.where_inv != 0)
      report("INVALIDATES", gi, vi, tl);
  }

  for (int n = 0; n < attribute->n_SyncGroups; ++n) {
    const gi_t gi{attribute->SyncGroups[n]};
    if (!group_inactive(gi))
      continue;
    report("SYNC", gi, std::nullopt, tl_t{0});
  }

  if (offender_count > 0)
    CCTK_VERROR("Storage check: %s::%s (%s) has %d schedule clause%s "
                "referencing groups without active STORAGE; aborting",
                attribute->thorn, attribute->routine, attribute->where,
                offender_count, offender_count == 1 ? "" : "s");
}

} // namespace CarpetX
