#include "hierarchy/loops.hxx"
#include "hierarchy/driver.hxx"

#include <cctk.h>
#include <cctk_Parameters.h>

#include <AMReX_Box.H>
#include <AMReX_Geometry.H>

#include <cassert>

namespace CarpetX {

bool window_is_restricted(const window_t window) {
  const auto [min_level, max_level] = window.level_bounds();
  assert(min_level < max_level);
  return max_level == driver().num_levels() && max_level - min_level >= 2;
}

GridDesc::GridDesc(const GHExt::PatchData::LevelData &leveldata,
                   const MFPointer &mfp) {
  DECLARE_CCTK_PARAMETERS;

  nghostzones = {ghost_size >= 0 ? ghost_size : ghost_size_x,
                 ghost_size >= 0 ? ghost_size : ghost_size_y,
                 ghost_size >= 0 ? ghost_size : ghost_size_z};

  const auto &patchdata = driver().patch(leveldata.level.patch());
  const amrex::IntVect ng(nghostzones[0], nghostzones[1], nghostzones[2]);
  const amrex::Box &domain =
      patchdata.amrcore->Geom(leveldata.level.index()).Domain();
  const amrex::Box &vbx = mfp.validbox();
  const amrex::Box &fbx = mfp.fabbox(ng);
  const amrex::Box &gbx = mfp.growntilebox(ng);

  for (int d = 0; d < dim; ++d)
    assert(domain.type(d) == amrex::IndexType::CELL);

  level = leveldata.level.index();
  patch = leveldata.level.patch().index();
  component = mfp.index();

  for (int d = 0; d < dim; ++d)
    gsh[d] = domain[orient(d, 1)] + 1 - domain[orient(d, 0)] + 1 +
             2 * nghostzones[d];

  for (int d = 0; d < dim; ++d)
    lsh[d] = fbx[orient(d, 1)] - fbx[orient(d, 0)] + 1 + 1;

  for (int d = 0; d < dim; ++d)
    ash[d] = lsh[d];

  for (int d = 0; d < dim; ++d) {
    lbnd[d] = fbx[orient(d, 0)] + nghostzones[d];
    ubnd[d] = fbx[orient(d, 1)] + 1 + nghostzones[d];
  }

  for (int d = 0; d < dim; ++d)
    for (int f = 0; f < 2; ++f)
      bbox[f][d] = vbx[orient(d, f)] == domain[orient(d, f)];

  for (int d = 0; d < dim; ++d) {
    tmin[d] = gbx[orient(d, 0)] - fbx[orient(d, 0)];
    assert(gbx[orient(d, 1)] <= fbx[orient(d, 1)]);
    tmax[d] = gbx[orient(d, 1)] + 1 - fbx[orient(d, 0)] +
              (gbx[orient(d, 1)] == fbx[orient(d, 1)]);
  }

  const amrex::Geometry &geom = patchdata.amrcore->Geom(0);
  const CCTK_REAL *restrict const global_x0 = geom.ProbLo();
  const CCTK_REAL *restrict const global_dx = geom.CellSize();
  for (int d = 0; d < dim; ++d) {
    const int levfac = 1 << leveldata.level.index();
    const int levoff = (1 - levfac) * (1 - 2 * nghostzones[d]);
    const int levoffdenom = 2;
    const CCTK_REAL origin_space =
        global_x0[d] + (1 - 2 * nghostzones[d]) * global_dx[d] / 2;
    const CCTK_REAL delta_space = global_dx[d];
    dx[d] = delta_space / levfac;
    x0[d] = origin_space + dx[d] * levoff / levoffdenom;
  }

  for (int d = 0; d < dim; ++d) {
    assert(gsh[d] >= 0);

    assert(lbnd[d] >= 0);
    assert(lsh[d] >= 0);
    assert(lbnd[d] + lsh[d] <= gsh[d]);
    assert(ubnd[d] == lbnd[d] + lsh[d] - 1);

    assert(ash[d] >= 0);
    assert(ash[d] >= lsh[d]);

    assert(nghostzones[d] >= 0);
    assert(2 * nghostzones[d] <= lsh[d]);

    assert(tmin[d] >= 0);
    assert(tmin[d] <= tmax[d]);
    assert(tmax[d] <= lsh[d]);
  }
}

GridDesc::GridDesc(const GHExt::PatchData::LevelData &leveldata,
                   const int component) {
  const auto &patchdata = driver().patch(leveldata.level.patch());

  const amrex::FabArrayBase &fab = *leveldata.fab;

  const amrex::Box &fbx = fab.fabbox(component);
  const amrex::Box &vbx = fab.box(component);
  const amrex::Box &gbx = fbx;
  const amrex::Box &domain =
      patchdata.amrcore->Geom(leveldata.level.index()).Domain();

  for (int d = 0; d < dim; ++d)
    assert(domain.type(d) == amrex::IndexType::CELL);

  level = leveldata.level.index();
  patch = leveldata.level.patch().index();
  this->component = component;

  for (int d = 0; d < dim; ++d)
    nghostzones[d] = fab.nGrowVect()[d];

  for (int d = 0; d < dim; ++d)
    gsh[d] = domain[orient(d, 1)] + 1 - domain[orient(d, 0)] + 1 +
             2 * nghostzones[d];

  for (int d = 0; d < dim; ++d)
    lsh[d] = fbx[orient(d, 1)] - fbx[orient(d, 0)] + 1 + 1;

  for (int d = 0; d < dim; ++d)
    ash[d] = lsh[d];

  for (int d = 0; d < dim; ++d) {
    lbnd[d] = fbx[orient(d, 0)] + nghostzones[d];
    ubnd[d] = fbx[orient(d, 1)] + 1 + nghostzones[d];
  }

  const auto &symmetries = patchdata.symmetries;
  for (int d = 0; d < dim; ++d)
    for (int f = 0; f < 2; ++f)
      bbox[f][d] = vbx[orient(d, f)] == domain[orient(d, f)] &&
                   symmetries[f][d] != symmetry_t::none;

  for (int d = 0; d < dim; ++d) {
    tmin[d] = gbx[orient(d, 0)] - fbx[orient(d, 0)];
    assert(gbx[orient(d, 1)] <= fbx[orient(d, 1)]);
    tmax[d] = gbx[orient(d, 1)] + 1 - fbx[orient(d, 0)] +
              (gbx[orient(d, 1)] == fbx[orient(d, 1)]);
  }

  const amrex::Geometry &geom = patchdata.amrcore->Geom(0);
  const CCTK_REAL *restrict const global_x0 = geom.ProbLo();
  const CCTK_REAL *restrict const global_dx = geom.CellSize();
  for (int d = 0; d < dim; ++d) {
    const int levfac = 1 << leveldata.level.index();
    const int levoff = (1 - levfac) * (1 - 2 * nghostzones[d]);
    const int levoffdenom = 2;
    const CCTK_REAL origin_space =
        global_x0[d] + (1 - 2 * nghostzones[d]) * global_dx[d] / 2;
    const CCTK_REAL delta_space = global_dx[d];
    dx[d] = delta_space / levfac;
    x0[d] = origin_space + dx[d] * levoff / levoffdenom;
  }

  for (int d = 0; d < dim; ++d) {
    assert(gsh[d] >= 0);

    assert(lbnd[d] >= 0);
    assert(lsh[d] >= 0);
    assert(lbnd[d] + lsh[d] <= gsh[d]);
    assert(ubnd[d] == lbnd[d] + lsh[d] - 1);

    assert(ash[d] >= 0);
    assert(ash[d] >= lsh[d]);

    assert(nghostzones[d] >= 0);
    assert(2 * nghostzones[d] <= lsh[d]);

    assert(tmin[d] >= 0);
    assert(tmin[d] <= tmax[d]);
    assert(tmax[d] <= lsh[d]);
  }
}

GridPtrDesc::GridPtrDesc(const GHExt::PatchData::LevelData &leveldata,
                         const MFPointer &mfp)
    : GridDesc(leveldata, mfp) {
  const amrex::IntVect ng(nghostzones[0], nghostzones[1], nghostzones[2]);
  const amrex::Box &fbx = mfp.fabbox(ng);
  cactus_offset = lbound(fbx);
}

GridPtrDesc1::GridPtrDesc1(
    const GHExt::PatchData::LevelData &leveldata,
    const GHExt::PatchData::LevelData::GroupData &groupdata,
    const MFPointer &mfp)
    : GridDesc(leveldata, mfp) {
  DECLARE_CCTK_PARAMETERS;
  const amrex::IntVect ng(nghostzones[0], nghostzones[1], nghostzones[2]);
  const amrex::Box &fbx = mfp.fabbox(ng);
  cactus_offset = lbound(fbx);
  for (int d = 0; d < dim; ++d) {
    assert(groupdata.nghostzones.at(d) >= 0);
    assert(groupdata.nghostzones.at(d) <= nghostzones[d]);
  }
  for (int d = 0; d < dim; ++d)
    gimin[d] = nghostzones[d] - groupdata.nghostzones.at(d);
  for (int d = 0; d < dim; ++d)
    gimax[d] = lsh[d] - groupdata.indextype.at(d) -
               (nghostzones[d] - groupdata.nghostzones.at(d));
  for (int d = 0; d < dim; ++d)
    gash[d] = ash[d] - groupdata.indextype.at(d) -
              2 * (nghostzones[d] - groupdata.nghostzones.at(d));
}

GridPtrDesc1::GridPtrDesc1(
    const GHExt::PatchData::LevelData &leveldata,
    const GHExt::PatchData::LevelData::GroupData &groupdata,
    const int component)
    : GridDesc(leveldata, component) {
  const amrex::Box &fbx = leveldata.fab->fabbox(component);
  cactus_offset = lbound(fbx);
  for (int d = 0; d < dim; ++d) {
    assert(groupdata.nghostzones.at(d) >= 0);
    assert(groupdata.nghostzones.at(d) <= nghostzones[d]);
  }
  for (int d = 0; d < dim; ++d)
    gimin[d] = nghostzones[d] - groupdata.nghostzones.at(d);
  for (int d = 0; d < dim; ++d)
    gimax[d] = lsh[d] - groupdata.indextype.at(d) -
               (nghostzones[d] - groupdata.nghostzones.at(d));
  for (int d = 0; d < dim; ++d)
    gash[d] = ash[d] - groupdata.indextype.at(d) -
              2 * (nghostzones[d] - groupdata.nghostzones.at(d));
}

} // namespace CarpetX
