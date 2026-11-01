#include "comm/reflux.hxx"

#include "hierarchy/driver.hxx"
#include "hierarchy/group_tags.hxx"
#include "hierarchy/valid.hxx"
#include "comm/level_pairs.hxx"
#include <carpetx.hxx>

#include <cctk.h>
#include <cctk_Parameters.h>

#include <AMReX_FluxRegister.H>
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>

#include <array>
#include <cassert>
#include <memory>
#include <optional>
#include <sstream>

namespace CarpetX {

static std::array<CCTK_REAL, dim> face_areas(const amrex::Geometry &geom) {
  const CCTK_REAL *const dx = geom.CellSize();
  std::array<CCTK_REAL, dim> area;
  for (int d = 0; d < dim; ++d) {
    area[d] = 1;
    for (int j = 0; j < dim; ++j)
      if (j != d)
        area[d] *= dx[j];
  }
  return area;
}

void EnsureFluxGeometry(const level_t level) {
  const auto &patchdata = driver().patch(level.patch());
  const auto &leveldata = patchdata.level(level);

  [[maybe_unused]] const auto have_all_or_none = [&]() {
    const bool have = bool(leveldata.cell_volume);
    bool good = true;
    for (int d = 0; d < dim; ++d)
      good = good && bool(leveldata.face_area[d]) == have;
    return good;
  };
  assert(have_all_or_none());
  if (leveldata.cell_volume)
    return;

  const amrex::Geometry &geom = patchdata.amrcore->Geom(level.index());
  assert(leveldata.fab);
  const amrex::BoxArray &ba = leveldata.fab->boxArray();
  const amrex::DistributionMapping &dm = leveldata.fab->DistributionMap();
  assert(ba.ixType().cellCentered());

  for (int d = 0; d < dim; ++d) {
    leveldata.face_area[d] = std::make_unique<amrex::MultiFab>();
    geom.GetFaceArea(*leveldata.face_area[d], ba, dm, d, 0);
  }
  leveldata.cell_volume = std::make_unique<amrex::MultiFab>();
  geom.GetVolume(*leveldata.cell_volume, ba, dm, 0);

  assert(have_all_or_none());
}

void AccumulateFluxes(const level_t level, const int stage,
                      const CCTK_REAL weight) {
  const GHExt &ext = driver();
  if (!ext.do_reflux)
    return;
  assert(stage >= 1 && stage <= num_rk_stages());

  static Timer timer("AccumulateFluxes");
  Interval interval(timer);

  const auto &patchdata = ext.patch(level.patch());
  const auto &leveldata = patchdata.level(level);
  const std::optional<level_t> child = level.child();
  const auto *const childleveldata = child ? &patchdata.level(*child) : nullptr;
  const tl_t tl{0};
  const window_t this_level = window_t::one_level(level);

  for (const auto &groupdata : leveldata.groups()) {
    if (!groupdata.fluxes)
      continue;
    const gi_t gi = groupdata.record.groupindex;

    amrex::FluxRegister *const child_freg =
        childleveldata ? childleveldata->group(gi).freg.get() : nullptr;
    amrex::FluxRegister *const own_freg = groupdata.freg.get();
    if (!child_freg && !own_freg)
      continue;

    EnsureFluxGeometry(level);

    for (int d = 0; d < dim; ++d) {
      std::ostringstream buf;
      buf << "AccumulateFluxes: flux of " << groupdata.record.groupname
          << " in direction " << d << " at RK stage " << stage;
      require(this_level, {.gi = groupdata.fluxes->at(d),
                           .tl = tl,
                           .where = make_valid_int(),
                           .event = buf.str()});
    }

    if (child_freg) {
      amrex::FluxRegister *const child_scratch =
          childleveldata->group(gi).freg_scratch.get();
      assert(child_scratch);
      if (stage == 1)
        child_freg->setVal(0);
      const int numvars = groupdata.record.numvars;
      const CCTK_REAL mult = -weight;
      const std::array<CCTK_REAL, dim> area =
          face_areas(patchdata.amrcore->Geom(level.index()));
      for (int d = 0; d < dim; ++d) {
        const amrex::MultiFab &flux =
            *leveldata.group(groupdata.fluxes->at(d)).timelevel(tl).mfab;
        const CCTK_REAL area_d = area[d];
        for (const auto side :
             {amrex::Orientation::low, amrex::Orientation::high}) {
          const amrex::Orientation face(d, side);
          amrex::FabSet &scratch = (*child_scratch)[face];
          amrex::FabSet &reg = (*child_freg)[face];
          assert(scratch.boxArray() == reg.boxArray());
          assert(scratch.DistributionMap() == reg.DistributionMap());
          assert(scratch.nComp() == numvars && reg.nComp() == numvars);
          scratch.setVal(0);
          scratch.copyFrom(flux, 0, 0, 0, numvars);
          for (amrex::FabSetIter fsi(scratch); fsi.isValid(); ++fsi) {
            const amrex::Box &bx = fsi.validbox();
            const auto sfab = scratch.array(fsi);
            amrex::ParallelFor(
                bx, numvars,
                [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) noexcept {
                  sfab(i, j, k, n) = sfab(i, j, k, n) * mult * area_d;
                });
          }
          reg.plusFrom(scratch, 0, 0, numvars);
        }
      }
    }

    if (own_freg) {
      for (int d = 0; d < dim; ++d) {
        const amrex::MultiFab &flux =
            *leveldata.group(groupdata.fluxes->at(d)).timelevel(tl).mfab;
        const amrex::MultiFab &area = *leveldata.face_area[d];
        assert(area.ixType() == flux.ixType());
        assert(area.DistributionMap() == flux.DistributionMap());
        own_freg->FineAdd(flux, area, d, 0, 0, groupdata.record.numvars,
                          +weight);
      }
    }
  }
}

static void reflux_level(const level_t level) {
  DECLARE_CCTK_PARAMETERS;

  static Timer timer("Reflux");
  Interval interval(timer);

  const std::optional<level_t> fine = level.child();
  if (!fine)
    return;
  const auto &patchdata = driver().patch(level.patch());
  const auto &leveldata = patchdata.level(level);
  const auto &fineleveldata = patchdata.level(*fine);
  const amrex::Geometry &geom = patchdata.amrcore->Geom(level.index());
  const window_t coarse_level = window_t::one_level(level);
  const tl_t tl{0};

  for (const auto &groupdata : leveldata.groups()) {
    const gi_t gi = groupdata.record.groupindex;
    const auto &finegroupdata = fineleveldata.group(gi);

    if (!finegroupdata.freg)
      continue;

    if (verbose)
      CCTK_VINFO("Reflux: applying the flux register of levels (%d, %d) to "
                 "%s at iteration %d",
                 level.index(), fine->index(),
                 groupdata.record.groupname.c_str(), current_iteration());

    require(coarse_level,
            {.gi = gi,
             .tl = tl,
             .where = make_valid_int(),
             .event = "Reflux before refluxing: Coarse level data"});

    EnsureFluxGeometry(level);

    const write_scope_t scope =
        open_write(coarse_level, {.gi = gi,
                                  .tl = tl,
                                  .fills = make_valid_int(),
                                  .keeps = make_valid_all(),
                                  .event = "Reflux"});
    amrex::MultiFab &state = *groupdata.timelevel(tl).mfab;
    const amrex::MultiFab &volume = *leveldata.cell_volume;
    assert(volume.ixType() == state.ixType());
    assert(volume.DistributionMap() == state.DistributionMap());
    finegroupdata.freg->Reflux(state, volume, 1.0, 0, 0,
                               groupdata.record.numvars, geom);
  }
}

void Reflux(const window_t window) {
  if (!driver().do_reflux)
    return;
  loop_pairs_fine_to_coarse(window, reflux_level);
}

} // namespace CarpetX
