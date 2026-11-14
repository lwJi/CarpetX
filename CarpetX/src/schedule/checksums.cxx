#include "schedule/checksums.hxx"

#include "hierarchy/model.hxx"
#include "hierarchy/loops.hxx"
#include "hierarchy/scopes.hxx"
#include <carpetx.hxx>

#include <loop_device.hxx>

#include <cctk.h>
#include <cctk_Parameters.h>

#include <cassert>
#include <functional>
#include <string>
#include <vector>

namespace CarpetX {

using Loop::where_t;

checksums_t calculate_checksums(
    const window_t window,
    const std::vector<std::vector<std::vector<valid_t> > > &will_write) {
  DECLARE_CCTK_PARAMETERS;

  checksums_t checksums;

  if (!poison_undefined_values)
    return checksums;

  static Timer timer("calculate_checksums");
  Interval interval(timer);

  loop_parallel(window, [&](const GHExt::PatchData::LevelData &leveldata,
                            const tile_t tile,
                            const cGH *restrict const cctkGH) {
    for (const auto &groupdata : leveldata.groups()) {
      const Loop::GridDescBaseDevice grid(cctkGH);
      const Loop::GF3D2layout layout(cctkGH, groupdata.grid_function.indextype);

      for (const vi_t vi : groupdata.record.variables()) {
        for (const tl_t tl : groupdata.record.timelevels()) {
          const gi_t gi = groupdata.record.groupindex;
          const tiletag_t tiletag{leveldata.level, tile, gi, vi, tl};

          const valid_t valid = groupdata.timelevel(tl).valid(vi);
          if (tl.index() >=
              int(will_write.at(gi.index()).at(vi.index()).size()))
            continue;
          const auto &wr =
              will_write.at(gi.index()).at(vi.index()).at(tl.index());
          valid_t to_check = valid & ~wr;

          if (!(wr.valid_any() && to_check.valid_any()))
            continue;

          const Loop::GF3D2<const CCTK_REAL> gf(
              layout,
              static_cast<const CCTK_REAL *>(CCTK_VarDataPtrI(
                  cctkGH, tl.index(), groupdata.record.var(vi).index())));

          checksum_t checksum(to_check);
          checksum.add(tiletag);
          const auto add_point = [&](const Loop::PointDesc &p) {
            checksum.add(gf(p.I));
          };

          if (to_check.valid_int)
            grid.loop_idx(where_t::interior, groupdata.grid_function.indextype,
                          groupdata.grid_function.nghostzones, add_point);

          if (to_check.valid_outer)
            grid.loop_idx(where_t::boundary, groupdata.grid_function.indextype,
                          groupdata.grid_function.nghostzones, add_point);

          loop_ghost_halves(grid, leveldata, groupdata, tile.component(),
                            to_check, add_point);

#pragma omp critical(CarpetX_calculate_checksums)
          checksums[tiletag] = checksum;
        }
      }
    }
  });

  return checksums;
}

void check_checksums(const window_t window, const checksums_t &checksums,
                     const std::function<std::string()> &where) {
  DECLARE_CCTK_PARAMETERS;

  if (!poison_undefined_values)
    return;
  if (checksums.empty())
    return;

  static Timer timer("check_checksums");
  Interval interval(timer);

  loop_parallel(window, [&](const GHExt::PatchData::LevelData &leveldata,
                            const tile_t tile,
                            const cGH *restrict const cctkGH) {
    for (const auto &groupdata : leveldata.groups()) {
      const Loop::GridDescBaseDevice grid(cctkGH);
      const Loop::GF3D2layout layout(cctkGH, groupdata.grid_function.indextype);

      for (const vi_t vi : groupdata.record.variables()) {
        for (const tl_t tl : groupdata.record.timelevels()) {
          const gi_t gi = groupdata.record.groupindex;
          const tiletag_t tiletag{leveldata.level, tile, gi, vi, tl};

          if (!checksums.count(tiletag))
            continue;

          const auto &old_checksum = checksums.at(tiletag);
          const auto &did_check = old_checksum.where;
          assert(did_check.valid_any());

          const Loop::GF3D2<const CCTK_REAL> gf(
              layout,
              static_cast<const CCTK_REAL *>(CCTK_VarDataPtrI(
                  cctkGH, tl.index(), groupdata.record.var(vi).index())));

          checksum_t checksum(did_check);
          checksum.add(tiletag);
          const auto add_point = [&](const Loop::PointDesc &p) {
            checksum.add(gf(p.I));
          };

          if (did_check.valid_int)
            grid.loop_idx(where_t::interior, groupdata.grid_function.indextype,
                          groupdata.grid_function.nghostzones, add_point);

          if (did_check.valid_outer)
            grid.loop_idx(where_t::boundary, groupdata.grid_function.indextype,
                          groupdata.grid_function.nghostzones, add_point);

          loop_ghost_halves(grid, leveldata, groupdata, tile.component(),
                            did_check, add_point);

          if (checksum != old_checksum)
#pragma omp critical
            CCTK_VERROR(
                "%s: Checksum mismatch: variable %s, tile %s, "
                "int:%d,outer:%d,same_level:%d,coarse_fine:%d, old checksum "
                "%s, new checksum %s",
                where().c_str(),
                CCTK_FullVarName(groupdata.record.var(tiletag.vi).index()),
                std::string(tiletag).c_str(), int(did_check.valid_int),
                int(did_check.valid_outer), int(did_check.valid_same_level),
                int(did_check.valid_coarse_fine),
                std::string(old_checksum).c_str(),
                std::string(checksum).c_str());
        }
      }
    }
  });
}

} // namespace CarpetX
