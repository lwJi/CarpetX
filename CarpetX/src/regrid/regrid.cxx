#include "hierarchy/driver.hxx"
#include "hierarchy/loops.hxx"
#include "hierarchy/operations.hxx"
#include "comm/fillpatch.hxx"
#include <loop_device.hxx>

#include <cctk_Parameters.h>

#include <AMReX_TagBox.H>

#include <cstddef>
#include <sstream>

namespace CarpetX {

CactusAmrCore::CactusAmrCore(const patch_t patch, const amrex::RealBox *rb,
                             int max_level_in,
                             const amrex::Vector<int> &n_cell_in, int coord,
                             amrex::Vector<amrex::IntVect> ref_ratios,
                             const int *is_per)
    : amrex::AmrCore(rb, max_level_in, n_cell_in, coord, ref_ratios, is_per),
      patch(patch) {}

CactusAmrCore::CactusAmrCore(const patch_t patch, const amrex::RealBox &rb,
                             int max_level_in,
                             const amrex::Vector<int> &n_cell_in, int coord,
                             amrex::Vector<amrex::IntVect> const &ref_ratios,
                             amrex::Array<int, AMREX_SPACEDIM> const &is_per)
    : amrex::AmrCore(rb, max_level_in, n_cell_in, coord, ref_ratios, is_per),
      patch(patch) {}

CactusAmrCore::~CactusAmrCore() {}

void CactusAmrCore::mark_initialized() { cactus_is_initialized = true; }

bool CactusAmrCore::is_initialized() const { return cactus_is_initialized; }

void CactusAmrCore::reset_level_modified(const int num_levels) {
  assert(num_levels >= 0);
  level_modified.assign(num_levels, false);
}

const std::vector<bool> &CactusAmrCore::modified_levels() const {
  return level_modified;
}

void CactusAmrCore::mark_level_modified(const level_t level) {
  assert(level.patch() == patch);
  level_modified.at(level.index()) = true;
}

void CactusAmrCore::ErrorEst(const int lev, amrex::TagBoxArray &tags,
                             const amrex::Real time, const int ngrow) {
  DECLARE_CCTK_PARAMETERS;
  const level_t level{patch, lev};

  if (!is_initialized())
    return;

  if (verbose)
#pragma omp critical
    CCTK_VINFO("ErrorEst patch %d level %d", patch.index(), lev);

  const gi_t gi{CCTK_GroupIndex("CarpetXRegrid::regrid_error")};
  assert(gi.index() >= 0);
  const vi_t vi{0};
  const tl_t tl{0};

  auto &restrict leveldata = driver().level(level);
  auto &restrict groupdata = leveldata.group(gi);
  require(window_t::one_level(level), {.gi = gi,
                                       .tl = tl,
                                       .vi = vi,
                                       .where = make_valid_int(),
                                       .event = "ErrorEst"});
  std::size_t npoints_set = 0, npoints_clear = 0, npoints_total = 0;
  auto mfitinfo = amrex::MFItInfo().SetDynamic(true).EnableTiling();
#pragma omp parallel
  for (amrex::MFIter mfi(*leveldata.fab, mfitinfo); mfi.isValid(); ++mfi) {
    GridPtrDesc1 grid(leveldata, groupdata, mfi);

    const amrex::Array4<const CCTK_REAL> &err_array4 =
        groupdata.timelevel(tl).mfab->array(mfi);
    const GF3D1<const CCTK_REAL> &err_ = grid.gf3d(err_array4, vi);
    const amrex::Array4<char> &tags_array4 = tags.array(mfi);

    std::size_t npoints_set_local = 0, npoints_clear_local = 0,
                npoints_total_local = 0;
    grid.loop_idx(
        where_t::interior, groupdata.indextype, [&](const Loop::PointDesc &p) {
          const int tag = err_(p.I) >= regrid_error_threshold
                              ? amrex::TagBox::SET
                              : amrex::TagBox::CLEAR;
          npoints_set_local += tag == amrex::TagBox::SET;
          npoints_clear_local += tag == amrex::TagBox::CLEAR;
          ++npoints_total_local;
          tags_array4(grid.cactus_offset.x + p.i, grid.cactus_offset.y + p.j,
                      grid.cactus_offset.z + p.k) = tag;
        });

#pragma omp atomic update
    npoints_set += npoints_set_local;
#pragma omp atomic update
    npoints_clear += npoints_clear_local;
#pragma omp atomic update
    npoints_total += npoints_total_local;
  }

  if (verbose)
#pragma omp critical
    CCTK_VINFO("ErrorEst patch %d level %d done. "
               "Set/clear/total=%td/%td/%td=%.0f%%/%.0f%%/%.0f%%",
               patch.index(), lev, npoints_set, npoints_clear, npoints_total,
               100.0 * npoints_set / npoints_total,
               100.0 * npoints_clear / npoints_total, 100.0);
}

void CactusAmrCore::SetupLevel(const level_t level, const amrex::BoxArray &ba,
                               const amrex::DistributionMapping &dm,
                               const std::function<std::string()> &why) {
  DECLARE_CCTK_PARAMETERS;
  assert(level.patch() == patch);
  const int lev = level.index();

  if (verbose)
#pragma omp critical
    CCTK_VINFO("SetupLevel patch %d level %d", patch.index(), lev);

  add_level(level, ba, dm, why);

  if (lev >= int(level_modified.size()))
    level_modified.resize(lev + 1, true);
  mark_level_modified(level);
  carpetx_epoch.fetch_add(1, std::memory_order_relaxed);

  if (verbose)
#pragma omp critical
    CCTK_VINFO("SetupLevel patch %d level %d done.", patch.index(), lev);
}

amrex::BoxArray CactusAmrCore::RechopLevel(const level_t recovered_level,
                                           amrex::BoxArray ba) const {
  assert(recovered_level.patch() == patch);
  if (recovered_level.index() == 0)
    return MakeBaseGrids();

  amrex::BoxList bl = ba.boxList();
  while (bl.simplify())
    ;
  ba = amrex::BoxArray(std::move(bl));
  ba.maxSize(maxGridSize(recovered_level.index()));
  if (refine_grid_layout)
    ChopGrids(recovered_level.index(), ba, amrex::ParallelDescriptor::NProcs());
  return ba;
}

void CactusAmrCore::make_level(const level_t level, const amrex::BoxArray &ba,
                               const amrex::DistributionMapping &dm,
                               const std::string &why) {
  SetupLevel(level, ba, dm, [why]() { return why; });

  const auto &leveldata = driver().level(level);
  const window_t window = window_t::one_level(level);
  for (const auto &groupdata : leveldata.groups()) {
    for (const tl_t tl : groupdata.record.timelevels())
      invalidate(window, {.gi = groupdata.record.groupindex,
                          .tl = tl,
                          .where = make_valid_all(),
                          .event = why});
  }
}

void CactusAmrCore::MakeNewLevelFromScratch(
    int lev, amrex::Real time, const amrex::BoxArray &ba,
    const amrex::DistributionMapping &dm) {
  DECLARE_CCTK_PARAMETERS;
  const level_t level{patch, lev};

  if (verbose)
#pragma omp critical
    CCTK_VINFO("MakeNewLevelFromScratch patch %d level %d", patch.index(), lev);

  make_level(level, ba, dm, "MakeNewLevelFromScratch");

  if (verbose)
#pragma omp critical
    CCTK_VINFO("MakeNewLevelFromScratch patch %d level %d done.", patch.index(),
               lev);
}

static tl_range regrid_prolongate_tls(const GHExt::GroupRecord &record) {
  const int ntls = record.num_timelevels();
  const int count = record.do_evolve       ? (ntls > 1 ? ntls - 1 : ntls)
                    : record.do_checkpoint ? ntls
                                           : 0;
  return tl_range{tl_t{0}, tl_t{count}};
}

static bool
regrid_source_valid(const GHExt::PatchData::LevelData::GroupData &groupdata,
                    const tl_t tl) {
  for (const vi_t vi : groupdata.record.variables())
    if (!groupdata.timelevel(tl).valid(vi).valid_all())
      return false;
  return true;
}

static valid_t
regrid_fills(const GHExt::PatchData::LevelData::GroupData &groupdata) {
  return make_valid_int() | make_valid_ghosts() |
         (groupdata.all_faces_have_symmetries_or_boundaries()
              ? make_valid_outer()
              : valid_t());
}

void CactusAmrCore::MakeNewLevelFromCoarse(
    const int lev, const amrex::Real time, const amrex::BoxArray &ba,
    const amrex::DistributionMapping &dm) {
  DECLARE_CCTK_PARAMETERS;
  const level_t level{patch, lev};

  if (verbose)
#pragma omp critical
    CCTK_VINFO("MakeNewLevelFromCoarse patch %d level %d", patch.index(), lev);

  assert(lev > 0);

  SetupLevel(level, ba, dm, []() { return "MakeNewLevelFromCoarse"; });

  const auto &patchdata = driver().patch(patch);
  const auto &leveldata = patchdata.level(level);
  const level_t coarse = level.parent();
  const auto &coarseleveldata = patchdata.level(coarse);
  const window_t fine_levels = window_t::one_level(level);
  const window_t coarse_levels = window_t::one_level(coarse);

  if (leveldata.iteration != coarseleveldata.iteration) {
    std::ostringstream msg;
    msg << "Coarse (rl=" << coarse.index()
        << ", it=" << coarseleveldata.iteration << ") and fine (rl=" << lev
        << ", it=" << leveldata.iteration
        << ") grid do not align in time when regridding";
#pragma omp critical
    CCTK_VERROR(msg.str().c_str());
  }

  for (auto &groupdata : leveldata.groups()) {
    const gi_t gi = groupdata.record.groupindex;
    auto &coarsegroupdata = coarseleveldata.group(gi);
    assert(&coarsegroupdata.record == &groupdata.record);
    amrex::Interpolater *const interpolator = groupdata.interpolator;

    const tl_range prolongate_tls = regrid_prolongate_tls(groupdata.record);
    const valid_t fills = regrid_fills(groupdata);

    for (const tl_t tl : groupdata.record.timelevels()) {
      if (prolongate_tls.contains(tl)) {
        const bool do_fill = groupdata.record.do_evolve ||
                             regrid_source_valid(coarsegroupdata, tl);
        if (do_fill) {
          require(coarse_levels,
                  {.gi = gi,
                   .tl = tl,
                   .where = make_valid_all(),
                   .event = "MakeNewLevelFromCoarse before prolongation"});
          const write_scope_t scope = open_write(
              fine_levels,
              {.gi = gi,
               .tl = tl,
               .fills = fills,
               .keeps = {},
               .event = "MakeNewLevelFromCoarse after prolongation"});
          FillPatch_NewLevel(
              groupdata, coarsegroupdata, *groupdata.timelevel(tl).mfab,
              *coarsegroupdata.timelevel(tl).mfab,
              patchdata.amrcore->Geom(coarse.index()),
              patchdata.amrcore->Geom(lev), interpolator, groupdata.bcrecs);
        } else {
          invalidate(fine_levels,
                     {.gi = gi,
                      .tl = tl,
                      .where = make_valid_all(),
                      .event = "MakeNewLevelFromCoarse: not prolongated "
                               "because source was invalid at regrid time"});
        }
      } else {
        invalidate(fine_levels,
                   {.gi = gi,
                    .tl = tl,
                    .where = make_valid_all(),
                    .event = groupdata.record.do_evolve
                                 ? "MakeNewLevelFromCoarse: oldest time level "
                                   "is not prolongated at regrid"
                                 : "MakeNewLevelFromCoarse: not prolongated "
                                   "because variable is not evolved"});
      }
    }
  }

  if (verbose)
#pragma omp critical
    CCTK_VINFO("MakeNewLevelFromCoarse patch %d level %d done.", patch.index(),
               lev);
}

void CactusAmrCore::RemakeLevel(const int lev, const amrex::Real time,
                                const amrex::BoxArray &ba,
                                const amrex::DistributionMapping &dm) {
  DECLARE_CCTK_PARAMETERS;
  const level_t level{patch, lev};

  if (verbose)
#pragma omp critical
    CCTK_VINFO("RemakeLevel patch %d level %d", patch.index(), lev);

  const auto &patchdata = driver().patch(patch);
  const auto &leveldata = patchdata.level(level);
  assert(lev > 0);
  const level_t coarse = level.parent();
  const auto &coarseleveldata = patchdata.level(coarse);
  const window_t fine_levels = window_t::one_level(level);
  const window_t coarse_levels = window_t::one_level(coarse);

  if (leveldata.iteration != coarseleveldata.iteration) {
    std::ostringstream msg;
    msg << "Coarse (rl=" << coarse.index()
        << ", it=" << coarseleveldata.iteration << ") and fine (rl=" << lev
        << ", it=" << leveldata.iteration
        << ") grid do not align in time when regridding";
#pragma omp critical
    CCTK_VERROR(msg.str().c_str());
  }

  const auto transports =
      [](const GHExt::PatchData::LevelData::GroupData &coarsegroupdata,
         const GHExt::PatchData::LevelData::GroupData &oldgroupdata,
         const tl_t tl) {
        return regrid_prolongate_tls(oldgroupdata.record).contains(tl) &&
               (oldgroupdata.record.do_evolve ||
                (regrid_source_valid(coarsegroupdata, tl) &&
                 regrid_source_valid(oldgroupdata, tl)));
      };

  for (const auto &groupdata : leveldata.groups()) {
    const gi_t gi = groupdata.record.groupindex;
    const auto &coarsegroupdata = coarseleveldata.group(gi);
    assert(&coarsegroupdata.record == &groupdata.record);

    for (const tl_t tl : groupdata.record.timelevels()) {
      if (!transports(coarsegroupdata, groupdata, tl))
        continue;
      require(coarse_levels, {.gi = gi,
                              .tl = tl,
                              .where = make_valid_all(),
                              .event = "RemakeLevel before prolongation"});
      require(fine_levels, {.gi = gi,
                            .tl = tl,
                            .where = make_valid_all(),
                            .event = "RemakeLevel before prolongation"});
    }
  }

  assert(ba.ixType() == amrex::IndexType(amrex::IndexType::CELL,
                                         amrex::IndexType::CELL,
                                         amrex::IndexType::CELL));

  assert(leveldata.iteration == coarseleveldata.iteration);

  const GHExt::PatchData::LevelData oldleveldata =
      remake_level(level, ba, dm, []() { return "RemakeLevel"; });

  for (const auto &groupdata : leveldata.groups()) {
    const gi_t gi = groupdata.record.groupindex;
    const auto &oldgroupdata = oldleveldata.group(gi);
    const auto &coarsegroupdata = coarseleveldata.group(gi);
    assert(&coarsegroupdata.record == &groupdata.record);
    amrex::Interpolater *const interpolator = groupdata.interpolator;

    const valid_t fills = regrid_fills(groupdata);
    assert(fills == make_valid_all());

    const tl_range prolongate_tls = regrid_prolongate_tls(groupdata.record);

    for (const tl_t tl : groupdata.record.timelevels()) {
      if (transports(coarsegroupdata, oldgroupdata, tl)) {
        const write_scope_t scope = open_write(
            fine_levels, {.gi = gi,
                          .tl = tl,
                          .fills = fills,
                          .keeps = {},
                          .event = "RemakeLevel after prolongation"});
        FillPatch_RemakeLevel(
            groupdata, coarsegroupdata, *groupdata.timelevel(tl).mfab,
            *coarsegroupdata.timelevel(tl).mfab,
            *oldgroupdata.timelevel(tl).mfab,
            patchdata.amrcore->Geom(coarse.index()),
            patchdata.amrcore->Geom(lev), interpolator, groupdata.bcrecs);
      } else {
        invalidate(fine_levels,
                   {.gi = gi,
                    .tl = tl,
                    .where = make_valid_all(),
                    .event = prolongate_tls.contains(tl)
                                 ? "RemakeLevel: not prolongated because "
                                   "source was invalid at regrid time"
                                 : "RemakeLevel"});
      }
    }
  }

  mark_level_modified(level);
  carpetx_epoch.fetch_add(1, std::memory_order_relaxed);

  if (verbose)
#pragma omp critical
    CCTK_VINFO("RemakeLevel patch %d level %d done.", patch.index(), lev);
}

void CactusAmrCore::ClearLevel(const int lev) {
  DECLARE_CCTK_PARAMETERS;
  const level_t level{patch, lev};

  if (verbose)
#pragma omp critical
    CCTK_VINFO("ClearLevel patch %d level %d", patch.index(), lev);
  erase_levels_from(level);

  level_modified.resize(lev);
  carpetx_epoch.fetch_add(1, std::memory_order_relaxed);

  if (verbose)
#pragma omp critical
    CCTK_VINFO("ClearLevel patch %d level %d done.", patch.index(), lev);
}

} // namespace CarpetX
