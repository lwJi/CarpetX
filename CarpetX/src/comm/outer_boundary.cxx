#include "comm/outer_boundary.hxx"

#include "hierarchy/loops.hxx"
#include "hierarchy/model.hxx"
#include "comm/boundary_conditions_inside.hxx"
#include "comm/group_selection_inside.hxx"
#include "comm/timelevels.hxx"
#include "util/refuse.hxx"
#include <carpetx.hxx>
#include "hierarchy/scopes.hxx"

#include <cctk.h>

#include <cassert>
#include <optional>
#include <source_location>
#include <span>
#include <vector>

namespace CarpetX {

namespace {
bool in_thorn_callout = false;

struct thorn_callout_t {
  explicit thorn_callout_t(
      const std::source_location site = std::source_location::current()) {
    error_if_sync_nested(site);
    in_thorn_callout = true;
  }
  ~thorn_callout_t() { in_thorn_callout = false; }
  thorn_callout_t(const thorn_callout_t &) = delete;
  thorn_callout_t &operator=(const thorn_callout_t &) = delete;
};
} // namespace

void error_if_sync_nested(const std::source_location site) {
  if (in_thorn_callout)
    refuse(site, "a sync is already open: a sync cannot be started inside "
                 "one. A routine that a sync calls, such as a thorn's "
                 "MultiPatch_Interpolate, must not sync grid functions");
}

static void sync_multipatch_boundaries(const std::span<const gi_t> groups) {
  const GHExt &ext = driver();
  static const bool have_multipatch_boundaries =
      CCTK_IsFunctionAliased("MultiPatch_Interpolate");

  if (have_multipatch_boundaries) {
    std::vector<CCTK_INT> cactusvarinds;
    for (const gi_t gi : groups) {
      const auto &group = ext.record(gi);
      for (const vi_t vi : group.variables())
        cactusvarinds.push_back(group.var(vi).index());
    }
    const thorn_callout_t callout;
    MultiPatch_Interpolate(ext.get_global_cctkGH(), cactusvarinds.size(),
                           cactusvarinds.data());
  } else {
    assert(ext.num_patches() == 1);
  }
}

static valid_t outer_fills(const window_t window, const gi_t gi) {
  std::optional<bool> covered;
  loop_serially(window, [&](auto &restrict leveldata) {
    const bool level_covered =
        leveldata.group(gi).all_faces_have_symmetries_or_boundaries();
    assert(!covered || *covered == level_covered);
    covered = level_covered;
  });
  return covered.value_or(false) ? make_valid_outer() : valid_t();
}

void fill_outer_boundary(const window_t window,
                         const std::span<const gi_t> groups,
                         const timelevels_t tls, const char *const event) {
  const GHExt &ext = driver();
  write_scopes_t scopes;
  for (const gi_t gi : groups) {
    const valid_t fills = outer_fills(window, gi);
    for (const tl_t tl : tls.range(ext.record(gi))) {
      require(window, {.gi = gi,
                       .tl = tl,
                       .where = make_valid_int() | make_valid_ghosts(),
                       .event = event});
      scopes.open(window, {.gi = gi,
                           .tl = tl,
                           .fills = fills,
                           .keeps = make_valid_all(),
                           .event = event});
    }
  }
  for (const gi_t gi : groups)
    loop_serially(window, [&](auto &restrict leveldata) {
      auto &restrict groupdata = leveldata.group(gi);
      for (const tl_t tl : tls.range(groupdata.record))
        apply_boundary_conditions(groupdata,
                                  scopes.mfab(leveldata.level, gi, tl));
    });
  synchronize();
  sync_multipatch_boundaries(groups);
  scopes.close();
}

void ApplyOuterBoundary(const window_t window,
                        const std::span<const gi_t> groups0,
                        const timelevels_t tls) {
  error_if_sync_nested();

  static Timer timer("ApplyOuterBoundary");
  Interval interval(timer);

  sync_log_groups("ApplyOuterBoundary", groups0);

  const std::vector<gi_t> groups = sync_filter_groups(groups0);
  fill_outer_boundary(window, groups, tls,
                      "CarpetX::apply_outer_boundary reads the interior and "
                      "both ghost regions");
}

} // namespace CarpetX
