#include "hierarchy/modes.hxx"
#include "hierarchy/grid.hxx"
#include "hierarchy/model.hxx"

#include <cctk.h>
#include <cctk_Parameters.h>

#include <AMReX_Box.H>
#include <AMReX_Geometry.H>

#include <cassert>
#include <cmath>

namespace CarpetX {

using Loop::undefined;

static mode_t current_mode(const cGH *restrict cctkGH) {
  const bool have_local = cctkGH->cctk_component != undefined;
  const bool have_patch = cctkGH->cctk_patch != undefined;
  const bool have_level = cctkGH->cctk_level != undefined;
  const bool have_global = cctkGH->cctk_nghostzones[0] != undefined;
  if (have_local && have_patch && have_level && have_global)
    return mode_t::local;
  else if (!have_local && have_patch && have_level && have_global)
    return mode_t::patch;
  else if (!have_local && !have_patch && have_level && have_global)
    return mode_t::level;
  else if (!have_local && !have_patch && !have_level && have_global)
    return mode_t::global;
  else if (!have_local && !have_patch && !have_level && !have_global)
    return mode_t::meta;
  else
    assert(0);
}

bool in_local_mode(const cGH *restrict cctkGH) {
  return current_mode(cctkGH) == mode_t::local;
}

bool in_patch_mode(const cGH *restrict cctkGH) {
  return current_mode(cctkGH) == mode_t::patch;
}

bool in_level_mode(const cGH *restrict cctkGH) {
  return current_mode(cctkGH) == mode_t::level;
}

bool in_global_mode(const cGH *restrict cctkGH) {
  return current_mode(cctkGH) == mode_t::global;
}

bool in_meta_mode(const cGH *restrict cctkGH) {
  return current_mode(cctkGH) == mode_t::meta;
}

void setup_cctkGH(cGH *restrict cctkGH) {
  DECLARE_CCTK_PARAMETERS;

  cctkGH->cctk_dim = 3;

  cctkGH->cctk_alignment = 1;
  cctkGH->cctk_alignment_offset = 0;

  cctkGH->cctk_timefac = 1;

  cctkGH->cctk_npatches = driver().num_patches();

  cctkGH->cctk_convlevel = 0;

  for (int d = 0; d < dim; ++d) {
    cctkGH->cctk_origin_space[d] = NAN;
    cctkGH->cctk_delta_space[d] = NAN;
  }

  cctkGH->cctk_time = 0;
  cctkGH->cctk_delta_time = NAN;

  cctkGH->cctk_component = undefined;
  cctkGH->cctk_level = undefined;
  cctkGH->cctk_patch = undefined;
  cctkGH->cctk_nghostzones[0] = undefined;
  assert(in_meta_mode(cctkGH));
}

void update_cctkGH(cGH *const cctkGH, const cGH *const sourceGH) {
  if (cctkGH == sourceGH)
    return;
  cctkGH->cctk_iteration = sourceGH->cctk_iteration;
  cctkGH->cctk_timefac = sourceGH->cctk_timefac;
  cctkGH->cctk_time = sourceGH->cctk_time;
  cctkGH->cctk_delta_time = sourceGH->cctk_delta_time;
}

void update_cctkGHs(cGH *restrict const cctkGH) {
  driver().for_each_cctkGH(
      [&](cGH *const copy) { update_cctkGH(copy, cctkGH); });
}

void enter_global_mode(cGH *restrict cctkGH) {
  DECLARE_CCTK_PARAMETERS;
  assert(in_meta_mode(cctkGH));

  cctkGH->cctk_nghostzones[0] = ghost_size >= 0 ? ghost_size : ghost_size_x;
  cctkGH->cctk_nghostzones[1] = ghost_size >= 0 ? ghost_size : ghost_size_y;
  cctkGH->cctk_nghostzones[2] = ghost_size >= 0 ? ghost_size : ghost_size_z;

  assert(in_global_mode(cctkGH));
}
void leave_global_mode(cGH *restrict cctkGH) {
  assert(in_global_mode(cctkGH));

  for (int d = 0; d < dim; ++d)
    cctkGH->cctk_nghostzones[d] = undefined;

  assert(in_meta_mode(cctkGH));
}

void enter_level_mode(cGH *restrict cctkGH, const level_t level) {
  DECLARE_CCTK_PARAMETERS;
  assert(in_global_mode(cctkGH));

  cctkGH->cctk_level = level.index();
  for (int d = 0; d < dim; ++d) {
    const int levfac = 1 << level.index();
    cctkGH->cctk_levfac[d] = levfac;
    const int levoff = (1 - levfac) * (1 - 2 * cctkGH->cctk_nghostzones[d]);
    const int levoffdenom = 2;
    cctkGH->cctk_levoff[d] = levoff;
    cctkGH->cctk_levoffdenom[d] = levoffdenom;
  }

  assert(in_level_mode(cctkGH));
}
void leave_level_mode(cGH *restrict cctkGH, const level_t level) {
  assert(in_level_mode(cctkGH));
  cctkGH->cctk_level = undefined;
  for (int d = 0; d < dim; ++d)
    cctkGH->cctk_levfac[d] = undefined;
  for (int d = 0; d < dim; ++d) {
    cctkGH->cctk_levoff[d] = undefined;
    cctkGH->cctk_levoffdenom[d] = 0;
  }
  assert(in_global_mode(cctkGH));
}

void enter_patch_mode(cGH *restrict cctkGH,
                      const GHExt::PatchData::LevelData &restrict leveldata) {
  DECLARE_CCTK_PARAMETERS;
  assert(in_level_mode(cctkGH));

  const auto &patchdata = driver().patch(leveldata.level.patch());

  cctkGH->cctk_patch = leveldata.level.patch().index();
  const amrex::Box &domain =
      patchdata.amrcore->Geom(leveldata.level.index()).Domain();
  const amrex::Geometry &geom = patchdata.amrcore->Geom(0);
  const CCTK_REAL *restrict const global_x0 = geom.ProbLo();
  const CCTK_REAL *restrict const global_dx = geom.CellSize();
  for (int d = 0; d < dim; ++d) {
    assert(cctkGH->cctk_nghostzones[d] != undefined);
    assert(domain.type(d) == amrex::IndexType::CELL);
    cctkGH->cctk_gsh[d] = domain[orient(d, 1)] + 1 - domain[orient(d, 0)] + 1 +
                          2 * cctkGH->cctk_nghostzones[d];
    const CCTK_REAL origin_space =
        global_x0[d] + (1 - 2 * cctkGH->cctk_nghostzones[d]) * global_dx[d] / 2;
    const CCTK_REAL delta_space = global_dx[d];
    cctkGH->cctk_delta_space[d] = delta_space;
    cctkGH->cctk_origin_space[d] = origin_space;
  }

  assert(in_patch_mode(cctkGH));
}
void leave_patch_mode(cGH *restrict cctkGH,
                      const GHExt::PatchData::LevelData &restrict leveldata) {
  assert(in_patch_mode(cctkGH));
  cctkGH->cctk_patch = undefined;
  for (int d = 0; d < dim; ++d)
    cctkGH->cctk_gsh[d] = undefined;
  for (int d = 0; d < dim; ++d) {
    cctkGH->cctk_origin_space[d] = NAN;
    cctkGH->cctk_delta_space[d] = NAN;
  }
  assert(in_level_mode(cctkGH));
}

void enter_local_mode(cGH *restrict cctkGH,
                      const GHExt::PatchData::LevelData &restrict leveldata,
                      const MFPointer &mfp) {
  assert(in_patch_mode(cctkGH));
  const GridPtrDesc grid(leveldata, mfp);

  cctkGH->cctk_component = mfp.index().index();
  for (int d = 0; d < dim; ++d)
    cctkGH->cctk_lsh[d] = grid.lsh[d];
  for (int d = 0; d < dim; ++d)
    cctkGH->cctk_ash[d] = grid.ash[d];
  for (int d = 0; d < dim; ++d) {
    cctkGH->cctk_lbnd[d] = grid.lbnd[d];
    cctkGH->cctk_ubnd[d] = grid.ubnd[d];
  }
  for (int d = 0; d < dim; ++d) {
    cctkGH->cctk_tile_min[d] = grid.tmin[d];
    cctkGH->cctk_tile_max[d] = grid.tmax[d];
  }
  for (int d = 0; d < dim; ++d)
    for (int f = 0; f < 2; ++f)
      cctkGH->cctk_bbox[2 * d + f] = grid.bbox[f][d];

  for (int d = 0; d < dim; ++d) {
    assert(cctkGH->cctk_gsh[d] >= 0);

    assert(cctkGH->cctk_lbnd[d] >= 0);
    assert(cctkGH->cctk_lsh[d] >= 0);
    assert(cctkGH->cctk_lbnd[d] + cctkGH->cctk_lsh[d] <= cctkGH->cctk_gsh[d]);
    assert(cctkGH->cctk_ubnd[d] ==
           cctkGH->cctk_lbnd[d] + cctkGH->cctk_lsh[d] - 1);

    assert(cctkGH->cctk_tile_min[d] >= 0);
    assert(cctkGH->cctk_tile_min[d] <= cctkGH->cctk_tile_max[d]);
    assert(cctkGH->cctk_tile_max[d] <= cctkGH->cctk_lsh[d]);

    assert(cctkGH->cctk_ash[d] >= 0);
    assert(cctkGH->cctk_ash[d] >= cctkGH->cctk_lsh[d]);

    assert(cctkGH->cctk_nghostzones[d] >= 0);
    assert(2 * cctkGH->cctk_nghostzones[d] <= cctkGH->cctk_lsh[d]);
  }

  assert(in_local_mode(cctkGH));
}
void leave_local_mode(cGH *restrict cctkGH,
                      const GHExt::PatchData::LevelData &restrict leveldata,
                      const MFPointer &mfp) {
  assert(in_local_mode(cctkGH));
  cctkGH->cctk_component = undefined;
  for (int d = 0; d < dim; ++d)
    cctkGH->cctk_lsh[d] = undefined;
  for (int d = 0; d < dim; ++d)
    cctkGH->cctk_ash[d] = undefined;
  for (int d = 0; d < dim; ++d) {
    cctkGH->cctk_lbnd[d] = undefined;
    cctkGH->cctk_ubnd[d] = undefined;
  }
  for (int d = 0; d < dim; ++d) {
    cctkGH->cctk_tile_min[d] = undefined;
    cctkGH->cctk_tile_max[d] = undefined;
  }
  for (int d = 0; d < dim; ++d)
    for (int f = 0; f < 2; ++f)
      cctkGH->cctk_bbox[2 * d + f] = undefined;
  assert(in_patch_mode(cctkGH));
}

} // namespace CarpetX
