#include "hierarchy/loops.hxx"
#include "hierarchy/driver.hxx"

#include <cctk.h>
#include <cctk_Parameters.h>

#include <AMReX_Box.H>
#include <AMReX_Geometry.H>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <type_traits>

namespace CarpetX {

using Loop::undefined;

cGH *copy_cctkGH(const cGH *restrict const sourceGH) {
  cGH *restrict const cctkGH = new cGH;

  *cctkGH = *sourceGH;

  const auto copy_array = [](const auto *restrict const srcptr, const int sz) {
    using T = std::decay_t<decltype(*srcptr)>;
    T *restrict const ptr = new T[sz];
    std::copy(srcptr, srcptr + sz, ptr);
    return ptr;
  };
  cctkGH->cctk_gsh = copy_array(sourceGH->cctk_gsh, dim);
  cctkGH->cctk_lsh = copy_array(sourceGH->cctk_lsh, dim);
  cctkGH->cctk_lbnd = copy_array(sourceGH->cctk_lbnd, dim);
  cctkGH->cctk_ubnd = copy_array(sourceGH->cctk_ubnd, dim);
  cctkGH->cctk_tile_min = copy_array(sourceGH->cctk_tile_min, dim);
  cctkGH->cctk_tile_max = copy_array(sourceGH->cctk_tile_max, dim);
  cctkGH->cctk_ash = copy_array(sourceGH->cctk_ash, dim);
  cctkGH->cctk_to = copy_array(sourceGH->cctk_to, dim);
  cctkGH->cctk_from = copy_array(sourceGH->cctk_from, dim);
  cctkGH->cctk_delta_space = copy_array(sourceGH->cctk_delta_space, dim);
  cctkGH->cctk_origin_space = copy_array(sourceGH->cctk_origin_space, dim);
  cctkGH->cctk_bbox = copy_array(sourceGH->cctk_bbox, 2 * dim);
  cctkGH->cctk_levfac = copy_array(sourceGH->cctk_levfac, dim);
  cctkGH->cctk_levoff = copy_array(sourceGH->cctk_levoff, dim);
  cctkGH->cctk_levoffdenom = copy_array(sourceGH->cctk_levoffdenom, dim);
  cctkGH->cctk_nghostzones = copy_array(sourceGH->cctk_nghostzones, dim);

  const int numvars = CCTK_NumVars();
  cctkGH->data = new void **[numvars];
  for (int vi = 0; vi < numvars; ++vi)
    cctkGH->data[vi] =
        copy_array(sourceGH->data[vi], CCTK_DeclaredTimeLevelsVI(vi));

  return cctkGH;
}

void delete_cctkGH(cGH *cctkGH) {
  delete[] cctkGH->cctk_gsh;
  delete[] cctkGH->cctk_lsh;
  delete[] cctkGH->cctk_lbnd;
  delete[] cctkGH->cctk_ubnd;
  delete[] cctkGH->cctk_tile_min;
  delete[] cctkGH->cctk_tile_max;
  delete[] cctkGH->cctk_ash;
  delete[] cctkGH->cctk_to;
  delete[] cctkGH->cctk_from;
  delete[] cctkGH->cctk_delta_space;
  delete[] cctkGH->cctk_origin_space;
  delete[] cctkGH->cctk_bbox;
  delete[] cctkGH->cctk_levfac;
  delete[] cctkGH->cctk_levoff;
  delete[] cctkGH->cctk_levoffdenom;
  delete[] cctkGH->cctk_nghostzones;
  const int numvars = CCTK_NumVars();
  for (int vi = 0; vi < numvars; ++vi)
    delete[] cctkGH->data[vi];
  delete[] cctkGH->data;
#ifdef CCTK_DEBUG
  memset(cctkGH, 0, sizeof *cctkGH);
#endif
  delete cctkGH;
}

mode_t current_mode(const cGH *restrict cctkGH) {
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

void enter_global_mode(cGH *restrict cctkGH) {
  DECLARE_CCTK_PARAMETERS;
  assert(in_meta_mode(cctkGH));

  cctkGH->cctk_nghostzones[0] = ghost_size >= 0 ? ghost_size : ghost_size_x;
  cctkGH->cctk_nghostzones[1] = ghost_size >= 0 ? ghost_size : ghost_size_y;
  cctkGH->cctk_nghostzones[2] = ghost_size >= 0 ? ghost_size : ghost_size_z;

  for (const auto &group : driver().records()) {
    if (group.kind == group_kind::grid_function)
      continue;
    const auto &restrict arraygroupdata = group.array();
    for (const tl_t tl : group.timelevels()) {
      const auto &restrict vars = arraygroupdata.timelevel(tl).data;
      for (const vi_t vi : group.variables())
        cctkGH->data[group.var(vi).index()][tl.index()] = const_cast<void *>(
            vars.data_at(vi.index() * arraygroupdata.array_size));
    }
  }

  assert(in_global_mode(cctkGH));
}
void leave_global_mode(cGH *restrict cctkGH) {
  assert(in_global_mode(cctkGH));

  for (const auto &group : driver().records()) {
    if (group.kind == group_kind::grid_function)
      continue;
    for (const tl_t tl : group.timelevels())
      for (const vi_t vi : group.variables())
        cctkGH->data[group.var(vi).index()][tl.index()] = nullptr;
  }

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

  cctkGH->cctk_component = mfp.index();
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

  for (const auto &groupdata : leveldata.groups()) {
    const GridPtrDesc1 grid1(leveldata, groupdata, mfp);
    for (const tl_t tl : groupdata.record.timelevels()) {
      const amrex::Array4<CCTK_REAL> vars =
          groupdata.timelevel(tl).mfab->array(mfp.index());
      for (const vi_t vi : groupdata.record.variables())
        cctkGH->data[groupdata.record.var(vi).index()][tl.index()] =
            grid1.ptr(vars, vi);
    }
  }

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
  for (const auto &groupdata : leveldata.groups())
    for (const tl_t tl : groupdata.record.timelevels())
      for (const vi_t vi : groupdata.record.variables())
        cctkGH->data[groupdata.record.var(vi).index()][tl.index()] = nullptr;
  assert(in_patch_mode(cctkGH));
}

} // namespace CarpetX
