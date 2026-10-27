#include "schedule.hxx"
#include "driver.hxx"
#include "fillpatch.hxx"
#include "io.hxx"
#include "loop.hxx"
#include "subcycling.hxx"
#include "sync_restrict_internal.hxx"
#include "task_manager.hxx"
#include <carpetx.hxx>
#include "valid.hxx"

#include <cctk.h>
#include <cctk_Parameters.h>
#include <cctk_Schedule.h>
#include <cctki_GHExtensions.h>
#include <cctki_ScheduleBindings.h>
#include <cctki_WarnLevel.h>
#include <util_Table.h>

#include <AMReX_MultiFabUtil.H>

#if defined _OPENMP
#include <omp.h>
#elif defined __HIPCC__
#define omp_get_max_threads() 1
#define omp_get_num_threads() 1
#define omp_get_thread_num() 0
#define omp_in_parallel() 0
#else
static inline int omp_get_max_threads() { return 1; }
static inline int omp_get_num_threads() { return 1; }
static inline int omp_get_thread_num() { return 0; }
static inline int omp_in_parallel() { return 0; }
#endif

#include <sys/time.h>

#include <algorithm>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <vector>

namespace CarpetX {

#ifndef CCTK_HAVE_CGH_LEVEL
#error                                                                         \
    "The Cactus flesh does not support cctk_level in the cGH structure. Update the flesh."
#endif
#ifndef CCTK_HAVE_CGH_PATCH
#error                                                                         \
    "The Cactus flesh does not support cctk_patch in the cGH structure. Update the flesh."
#endif
#ifndef CCTK_HAVE_CGH_COMPONENT
#error                                                                         \
    "The Cactus flesh does not support cctk_component in the cGH structure. Update the flesh."
#endif
#ifndef CCTK_HAVE_CGH_TILE
#error                                                                         \
    "The Cactus flesh does not support cctk_tile_min etc. in the cGH structure. Update the flesh."
#endif

#if defined _OPENMP
#if !defined AMREX_USE_OMP
#error                                                                         \
    "Cactus is configured with OpenMP, but AMReX is configured without OpenMP. This does not work."
#endif
#endif

namespace {
double gettime() {
  timeval tv;
  gettimeofday(&tv, nullptr);
  return tv.tv_sec + tv.tv_usec / 1.0e+6;
}
} // namespace

namespace {
std::optional<window_t> active_levels;
} // namespace

const std::optional<window_t> &scheduled_levels() { return active_levels; }

int SyncGroupsOnActiveLevels(const cGH *restrict const cctkGH,
                             const int numgroups, const int *const groups,
                             const int *const directions) {
  assert(active_levels);
  const std::vector<gi_t> gis(groups, groups + numgroups);
  return SyncGroups(*active_levels, cctkGH, gis);
}

namespace {
// Convert a (direction, face) pair to an AMReX Orientation
amrex::Orientation orient(int d, int f) {
  return amrex::Orientation(d, amrex::Orientation::Side(f));
}
int GroupStorageCrease(const cGH *cctkGH, int n_groups, const int *groups,
                       const int *requested_tls, int *status, const bool inc);

level_t WidenMinLevel(level_t level, const rat64 target_iteration) {
  while (level.index() > 0 &&
         ghext->level(level.parent()).iteration == target_iteration)
    level = level.parent();
  return level;
}

window_t ClockGroupBelow(const int max_level) {
  const level_t finest = window_t::levels(0, max_level).finest();
  const level_t coarsest =
      WidenMinLevel(finest, ghext->level(finest).iteration);
  return window_t::levels(coarsest.index(), max_level);
}

std::vector<window_t> ClockGroups() {
  std::vector<window_t> groups;
  for (int max_level = ghext->num_levels(); max_level > 0;
       max_level = groups.back().level_bounds().min)
    groups.push_back(ClockGroupBelow(max_level));
  std::reverse(groups.begin(), groups.end());
  assert((ghext->use_subcycling || groups.size() <= 1) &&
         "without subcycling every level shares one clock");
  return groups;
}

window_t FinestClockGroup() {
  const window_t group = ClockGroupBelow(ghext->num_levels());
  assert((ghext->use_subcycling || group.level_bounds().min == 0) &&
         "without subcycling the end-of-step window holds every level");
  return group;
}

int LastStepIteration(const level_t level, const int current_iteration) {
  const auto &finest = ghext->level(all_levels().finest());
  const auto &leveldata = ghext->level(level);
  const rat64 steps_since =
      (finest.iteration - (leveldata.iteration - leveldata.delta_iteration)) /
      finest.delta_iteration;
  assert(steps_since.den == 1 && steps_since.num >= 1);
  const int iteration = current_iteration + 1 - int(steps_since.num);
  assert(iteration >= 0 && iteration <= current_iteration);
  return iteration;
}

// A level clock in coarse steps, e.g. "2" or "3/2"
std::string format_clock(const rat64 &clock) {
  std::ostringstream buf;
  buf << clock.num;
  if (clock.den != 1)
    buf << "/" << clock.den;
  return buf.str();
}
} // namespace

bool window_is_restricted(const window_t window) {
  const auto [min_level, max_level] = window.level_bounds();
  assert(min_level < max_level);
  return max_level == ghext->num_levels() && max_level - min_level >= 2;
}

////////////////////////////////////////////////////////////////////////////////

GridDesc::GridDesc(const GHExt::PatchData::LevelData &leveldata,
                   const MFPointer &mfp) {
  DECLARE_CCTK_PARAMETERS;

  nghostzones = {ghost_size >= 0 ? ghost_size : ghost_size_x,
                 ghost_size >= 0 ? ghost_size : ghost_size_y,
                 ghost_size >= 0 ? ghost_size : ghost_size_z};

  const auto &patchdata = ghext->patch(leveldata.level.patch());
  const amrex::IntVect ng(nghostzones[0], nghostzones[1], nghostzones[2]);
  const amrex::Box &domain =
      patchdata.amrcore->Geom(leveldata.level.index()).Domain();
  const amrex::Box &vbx = mfp.validbox(); // interior region (without ghosts)
  const amrex::Box &fbx = mfp.fabbox(ng); // allocated array
  const amrex::Box &gbx = mfp.growntilebox(ng); // current region (with ghosts)

  for (int d = 0; d < dim; ++d)
    assert(domain.type(d) == amrex::IndexType::CELL);

  level = leveldata.level.index();
  patch = leveldata.level.patch().index();
  component = mfp.index();

  // Global shape
  for (int d = 0; d < dim; ++d)
    gsh[d] = domain[orient(d, 1)] + 1 - domain[orient(d, 0)] + 1 +
             2 * nghostzones[d];

  // Local shape
  for (int d = 0; d < dim; ++d)
    lsh[d] = fbx[orient(d, 1)] - fbx[orient(d, 0)] + 1 + 1;

  // Allocated shape
  for (int d = 0; d < dim; ++d)
    ash[d] = lsh[d];

  // Local extent
  for (int d = 0; d < dim; ++d) {
    lbnd[d] = fbx[orient(d, 0)] + nghostzones[d];
    ubnd[d] = fbx[orient(d, 1)] + 1 + nghostzones[d];
  }

  // Boundaries
  for (int d = 0; d < dim; ++d)
    for (int f = 0; f < 2; ++f)
      bbox[f][d] = vbx[orient(d, f)] == domain[orient(d, f)];

  // Thread tile box
  for (int d = 0; d < dim; ++d) {
    tmin[d] = gbx[orient(d, 0)] - fbx[orient(d, 0)];
    // For vertex centred grids, the allocated box is 1 vertex larger
    // than the number of cells, and AMReX assigns this extra vertex
    // to the final tile
    assert(gbx[orient(d, 1)] <= fbx[orient(d, 1)]);
    tmax[d] = gbx[orient(d, 1)] + 1 - fbx[orient(d, 0)] +
              (gbx[orient(d, 1)] == fbx[orient(d, 1)]);
  }

  const amrex::Geometry &geom = patchdata.amrcore->Geom(0);
  const CCTK_REAL *restrict const global_x0 = geom.ProbLo();
  const CCTK_REAL *restrict const global_dx = geom.CellSize();
  for (int d = 0; d < dim; ++d) {
    const int levfac = 1 << leveldata.level.index();
    // Offset between this level's and the coarsest level's origin as
    // multiple of the grid spacing
    const int levoff = (1 - levfac) * (1 - 2 * nghostzones[d]);
    const int levoffdenom = 2;
    // Vertex-centred coordinates on coarse level
    const CCTK_REAL origin_space =
        global_x0[d] + (1 - 2 * nghostzones[d]) * global_dx[d] / 2;
    const CCTK_REAL delta_space = global_dx[d];
    // Vertex-centred coordinates on current level
    dx[d] = delta_space / levfac;
    x0[d] = origin_space + dx[d] * levoff / levoffdenom;
  }

  // Check constraints
  for (int d = 0; d < dim; ++d) {
    // Domain size
    assert(gsh[d] >= 0);

    // Local size
    assert(lbnd[d] >= 0);
    assert(lsh[d] >= 0);
    assert(lbnd[d] + lsh[d] <= gsh[d]);
    assert(ubnd[d] == lbnd[d] + lsh[d] - 1);

    // Internal representation
    assert(ash[d] >= 0);
    assert(ash[d] >= lsh[d]);

    // Ghost zones
    assert(nghostzones[d] >= 0);
    assert(2 * nghostzones[d] <= lsh[d]);

    // Tiles
    assert(tmin[d] >= 0);
    assert(tmin[d] <= tmax[d]);
    assert(tmax[d] <= lsh[d]);
  }
}

GridDesc::GridDesc(const GHExt::PatchData::LevelData &leveldata,
                   const int component) {
  // `global_component` is the global component index.
  // There is no tiling.

  const auto &patchdata = ghext->patch(leveldata.level.patch());

  const amrex::FabArrayBase &fab = *leveldata.fab;

  const amrex::Box &fbx = fab.fabbox(component); // allocated array
  const amrex::Box &vbx =
      fab.box(component);      // interior region (without ghosts)
  const amrex::Box &gbx = fbx; // current region (with ghosts)
  const amrex::Box &domain =
      patchdata.amrcore->Geom(leveldata.level.index()).Domain();

  for (int d = 0; d < dim; ++d)
    assert(domain.type(d) == amrex::IndexType::CELL);

  level = leveldata.level.index();
  patch = leveldata.level.patch().index();
  this->component = component;

  // The number of ghostzones in each direction
  for (int d = 0; d < dim; ++d)
    nghostzones[d] = fab.nGrowVect()[d];

  // Global shape
  for (int d = 0; d < dim; ++d)
    gsh[d] = domain[orient(d, 1)] + 1 - domain[orient(d, 0)] + 1 +
             2 * nghostzones[d];

  // Local shape
  for (int d = 0; d < dim; ++d)
    lsh[d] = fbx[orient(d, 1)] - fbx[orient(d, 0)] + 1 + 1;

  // Allocated shape
  for (int d = 0; d < dim; ++d)
    ash[d] = lsh[d];

  // Local extent
  for (int d = 0; d < dim; ++d) {
    lbnd[d] = fbx[orient(d, 0)] + nghostzones[d];
    ubnd[d] = fbx[orient(d, 1)] + 1 + nghostzones[d];
  }

  const auto &symmetries = patchdata.symmetries;
  for (int d = 0; d < dim; ++d)
    for (int f = 0; f < 2; ++f)
      bbox[f][d] = vbx[orient(d, f)] == domain[orient(d, f)] &&
                   symmetries[f][d] != symmetry_t::none;

  // Thread tile box
  for (int d = 0; d < dim; ++d) {
    tmin[d] = gbx[orient(d, 0)] - fbx[orient(d, 0)];
    // For vertex centred grids, the allocated box is 1 vertex larger
    // than the number of cells, and AMReX assigns this extra vertex
    // to the final tile
    assert(gbx[orient(d, 1)] <= fbx[orient(d, 1)]);
    tmax[d] = gbx[orient(d, 1)] + 1 - fbx[orient(d, 0)] +
              (gbx[orient(d, 1)] == fbx[orient(d, 1)]);
  }

  const amrex::Geometry &geom = patchdata.amrcore->Geom(0);
  const CCTK_REAL *restrict const global_x0 = geom.ProbLo();
  const CCTK_REAL *restrict const global_dx = geom.CellSize();
  for (int d = 0; d < dim; ++d) {
    const int levfac = 1 << leveldata.level.index();
    // Offset between this level's and the coarsest level's origin as
    // multiple of the grid spacing
    const int levoff = (1 - levfac) * (1 - 2 * nghostzones[d]);
    const int levoffdenom = 2;
    // Vertex-centred coordinates on coarse level
    const CCTK_REAL origin_space =
        global_x0[d] + (1 - 2 * nghostzones[d]) * global_dx[d] / 2;
    const CCTK_REAL delta_space = global_dx[d];
    // Vertex-centred coordinates on current level
    dx[d] = delta_space / levfac;
    x0[d] = origin_space + dx[d] * levoff / levoffdenom;
  }

  // Check constraints
  for (int d = 0; d < dim; ++d) {
    // Domain size
    assert(gsh[d] >= 0);

    // Local size
    assert(lbnd[d] >= 0);
    assert(lsh[d] >= 0);
    assert(lbnd[d] + lsh[d] <= gsh[d]);
    assert(ubnd[d] == lbnd[d] + lsh[d] - 1);

    // Internal representation
    assert(ash[d] >= 0);
    assert(ash[d] >= lsh[d]);

    // Ghost zones
    assert(nghostzones[d] >= 0);
    assert(2 * nghostzones[d] <= lsh[d]);

    // Tiles
    assert(tmin[d] >= 0);
    assert(tmin[d] <= tmax[d]);
    assert(tmax[d] <= lsh[d]);
  }
}

GridPtrDesc::GridPtrDesc(const GHExt::PatchData::LevelData &leveldata,
                         const MFPointer &mfp)
    : GridDesc(leveldata, mfp) {
  const amrex::IntVect ng(nghostzones[0], nghostzones[1], nghostzones[2]);
  const amrex::Box &fbx = mfp.fabbox(ng); // allocated array
  cactus_offset = lbound(fbx);
}

GridPtrDesc1::GridPtrDesc1(
    const GHExt::PatchData::LevelData &leveldata,
    const GHExt::PatchData::LevelData::GroupData &groupdata,
    const MFPointer &mfp)
    : GridDesc(leveldata, mfp) {
  DECLARE_CCTK_PARAMETERS;
  const amrex::IntVect ng(nghostzones[0], nghostzones[1], nghostzones[2]);
  const amrex::Box &fbx = mfp.fabbox(ng); // allocated array
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
  const amrex::Box &fbx = leveldata.fab->fabbox(component); // allocated array
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

////////////////////////////////////////////////////////////////////////////////

cGH *copy_cctkGH(const cGH *restrict const sourceGH) {
  cGH *restrict const cctkGH = new cGH;

  // Copy all fields by default
  *cctkGH = *sourceGH;

  // Allocate most pointers anew
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

enum class mode_t { unknown, local, patch, level, global, meta };

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

// Initialize cctkGH entries
void setup_cctkGH(cGH *restrict cctkGH) {
  DECLARE_CCTK_PARAMETERS;

  // Dimensions
  cctkGH->cctk_dim = 3;

  // Grid function alignment
  // TODO: Check whether AMReX guarantees a particular alignment
  cctkGH->cctk_alignment = 1;
  cctkGH->cctk_alignment_offset = 0;

  // The refinement factor in time over the top level (coarsest) grid
  cctkGH->cctk_timefac = 1; // no subcycling

  // The total number of patches
  cctkGH->cctk_npatches = ghext->num_patches();

  // The convergence level (numbered from zero upwards)
  cctkGH->cctk_convlevel = 0; // no convergence tests

  // Initialize grid spacing
  for (int d = 0; d < dim; ++d) {
    cctkGH->cctk_origin_space[d] = NAN;
    cctkGH->cctk_delta_space[d] = NAN;
  }

  // Initialize time stepping
  cctkGH->cctk_time = 0;
  cctkGH->cctk_delta_time = NAN;

  // init into meta mode
  cctkGH->cctk_component = undefined;
  cctkGH->cctk_level = undefined;
  cctkGH->cctk_patch = undefined;
  cctkGH->cctk_nghostzones[0] = undefined;
  assert(in_meta_mode(cctkGH));
}

// Update fields that carry state and change over time
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

  // The number of ghostzones in each direction
  // TODO: Get this from mfab (mfab.fb_ghosts)
  cctkGH->cctk_nghostzones[0] = ghost_size >= 0 ? ghost_size : ghost_size_x;
  cctkGH->cctk_nghostzones[1] = ghost_size >= 0 ? ghost_size : ghost_size_y;
  cctkGH->cctk_nghostzones[2] = ghost_size >= 0 ? ghost_size : ghost_size_z;

  for (const auto &group : ghext->records()) {
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

  for (const auto &group : ghext->records()) {
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
    // Offset between this level's and the coarsest level's origin as multiple
    // of the grid spacing
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

  const auto &patchdata = ghext->patch(leveldata.level.patch());

  cctkGH->cctk_patch = leveldata.level.patch().index();
  const amrex::Box &domain =
      patchdata.amrcore->Geom(leveldata.level.index()).Domain();
  const amrex::Geometry &geom = patchdata.amrcore->Geom(0);
  const CCTK_REAL *restrict const global_x0 = geom.ProbLo();
  const CCTK_REAL *restrict const global_dx = geom.CellSize();
  for (int d = 0; d < dim; ++d) {
    // Global shape
    assert(cctkGH->cctk_nghostzones[d] != undefined);
    assert(domain.type(d) == amrex::IndexType::CELL);
    cctkGH->cctk_gsh[d] = domain[orient(d, 1)] + 1 - domain[orient(d, 0)] + 1 +
                          2 * cctkGH->cctk_nghostzones[d];
    // Vertex-centred coarse level coordinates
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

  // Check constraints
  for (int d = 0; d < dim; ++d) {
    // Domain size
    assert(cctkGH->cctk_gsh[d] >= 0);

    // Local size
    assert(cctkGH->cctk_lbnd[d] >= 0);
    assert(cctkGH->cctk_lsh[d] >= 0);
    assert(cctkGH->cctk_lbnd[d] + cctkGH->cctk_lsh[d] <= cctkGH->cctk_gsh[d]);
    assert(cctkGH->cctk_ubnd[d] ==
           cctkGH->cctk_lbnd[d] + cctkGH->cctk_lsh[d] - 1);

    // Tile box
    assert(cctkGH->cctk_tile_min[d] >= 0);
    assert(cctkGH->cctk_tile_min[d] <= cctkGH->cctk_tile_max[d]);
    assert(cctkGH->cctk_tile_max[d] <= cctkGH->cctk_lsh[d]);

    // Internal representation
    assert(cctkGH->cctk_ash[d] >= 0);
    assert(cctkGH->cctk_ash[d] >= cctkGH->cctk_lsh[d]);

    // Ghost zones
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

// Should this be passed in `cGH`?
int CallFunction_count = -1;
extern "C" CCTK_INT CarpetX_GetCallFunctionCount() {
#ifndef AMREX_USE_GPU
  // CPU: use hardware thread index
  return omp_get_thread_num();
#else
  // GPU: count tiles
  assert(CallFunction_count >= 0);
  return CallFunction_count;
#endif
}

static void assert_consistent_iterations(const window_t window) {
  const auto levels = window.level_bounds();
  const auto patches = window.patch_bounds();
  rat64 good_iteration = -1;
  for (int level = levels.min; level < levels.max; ++level) {
    for (int patch = patches.min; patch < patches.max; ++patch) {
      const auto &patchdata = ghext->patch(patch_t{patch});
      if (level < patchdata.num_levels()) {
        const auto &leveldata =
            patchdata.level(level_t{patchdata.patch, level});
        const auto &iteration = leveldata.iteration;
        if (good_iteration == -1)
          good_iteration = iteration;
        assert(iteration == good_iteration);
      }
    }
  }
}

void loop_coarse_to_fine(
    const window_t window,
    const std::function<void(GHExt::PatchData::LevelData &leveldata)> &kernel) {
  assert(omp_get_num_threads() == 1);
  assert_consistent_iterations(window);
  const auto levels = window.level_bounds();
  const auto patches = window.patch_bounds();
  for (int level = levels.min; level < levels.max; ++level) {
    for (int patch = patches.min; patch < patches.max; ++patch) {
      auto &patchdata = ghext->patch(patch_t{patch});
      if (level < patchdata.num_levels())
        kernel(patchdata.level(level_t{patchdata.patch, level}));
    }
  }
}

void loop_fine_to_coarse(
    const window_t window,
    const std::function<void(GHExt::PatchData::LevelData &leveldata)> &kernel) {
  assert(omp_get_num_threads() == 1);
  assert_consistent_iterations(window);
  const auto levels = window.level_bounds();
  const auto patches = window.patch_bounds();
  for (int level = levels.max - 1; level >= levels.min; --level) {
    for (int patch = patches.min; patch < patches.max; ++patch) {
      auto &patchdata = ghext->patch(patch_t{patch});
      if (level < patchdata.num_levels())
        kernel(patchdata.level(level_t{patchdata.patch, level}));
    }
  }
}

void loop_parallel(
    const window_t window,
    const std::function<void(GHExt::PatchData::LevelData &leveldata, int index,
                             int component, const cGH *cctkGH)> &kernel) {
  assert(omp_get_num_threads() == 1);
  task_manager tasks;

  loop_coarse_to_fine(window, [&](auto &restrict leveldata) {
    int component = 0;
    const auto mfitinfo = amrex::MFItInfo().DisableDeviceSync().EnableTiling();
    const auto &fab0 = *leveldata.fab;
    for (amrex::MFIter mfi(fab0, mfitinfo); mfi.isValid(); ++mfi, ++component) {
      const int index = mfi.index();
      tasks.submit_serially([&kernel, &leveldata, index, component]() {
        cGH *restrict const localGH = leveldata.get_local_cctkGH(component);
        kernel(leveldata, index, component, localGH);
      });
    }
  });

  // Run all tasks
  tasks.run_tasks();
  // There is an implicit OpenMP barrier here.
}

void loop_serially(
    const window_t window,
    const std::function<void(GHExt::PatchData::LevelData &leveldata, int index,
                             int component, const cGH *cctkGH)> &kernel) {
  loop_coarse_to_fine(window, [&](auto &restrict leveldata) {
    int component = 0;
    const auto mfitinfo = amrex::MFItInfo().DisableDeviceSync().EnableTiling();
    const auto &fab0 = *leveldata.fab;
    for (amrex::MFIter mfi(fab0, mfitinfo); mfi.isValid(); ++mfi, ++component) {
      const int index = mfi.index();
      cGH *restrict const localGH = leveldata.get_local_cctkGH(component);
      kernel(leveldata, index, component, localGH);
    }
  });
}

void synchronize() {
#ifdef AMREX_USE_GPU
  // TODO: Synchronize only if GPU kernels were actually launched
  amrex::Gpu::streamSynchronizeAll();
  AMREX_GPU_ERROR_CHECK();
#endif
}

void update_cctkGHs(cGH *restrict const cctkGH) {
  update_cctkGH(ghext->global_cctkGH.get(), cctkGH);
  for (auto &restrict level_cctkGH : ghext->level_cctkGHs) {
    update_cctkGH(level_cctkGH.get(), cctkGH);
  }
  for (auto &patch : ghext->patches()) {
    for (auto &restrict level : patch.levels()) {
      update_cctkGH(level.patch_cctkGH.get(), cctkGH);
    }
  }
  for (auto &patch : ghext->patches()) {
    for (auto &restrict level : patch.levels()) {
      for (auto &restrict local_cctkGH : level.local_cctkGHs) {
        update_cctkGH(local_cctkGH.get(), cctkGH);
      }
    }
  }
}

////////////////////////////////////////////////////////////////////////////////

extern "C" void CarpetX_GetLoopBoxAll(const void *restrict const cctkGH_,
                                      const CCTK_INT size,
                                      CCTK_INT *restrict const loop_min,
                                      CCTK_INT *restrict const loop_max) {
  const cGH *restrict const cctkGH = static_cast<const cGH *>(cctkGH_);
  assert(size == dim);
  const GridDescBase grid(cctkGH);
  vect<int, dim> imin, imax;
  // Assumd cell centred box
  grid.box_all<1, 1, 1>(grid.nghostzones, imin, imax);
  for (int d = 0; d < dim; ++d)
    loop_min[d] = imin[d];
  for (int d = 0; d < dim; ++d)
    loop_max[d] = imax[d];
}

extern "C" void CarpetX_GetLoopBoxInt(const void *restrict const cctkGH_,
                                      const CCTK_INT size,
                                      CCTK_INT *restrict const loop_min,
                                      CCTK_INT *restrict const loop_max) {
  const cGH *restrict const cctkGH = static_cast<const cGH *>(cctkGH_);
  assert(size == dim);
  const GridDescBase grid(cctkGH);
  vect<int, dim> imin, imax;
  // Assumd cell centred box
  grid.box_int<1, 1, 1>(grid.nghostzones, imin, imax);
  for (int d = 0; d < dim; ++d)
    loop_min[d] = imin[d];
  for (int d = 0; d < dim; ++d)
    loop_max[d] = imax[d];
}

////////////////////////////////////////////////////////////////////////////////

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
  return mode_t::local; // default
}

enum class rdwr_t { read, write, invalid };
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
    // Ignore clauses that have no effect
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

CCTK_REAL get_coarse_mindx() {
  CCTK_REAL mindx = std::numeric_limits<CCTK_REAL>::infinity();
  for (const auto &patchdata : ghext->patches()) {
    const amrex::Geometry &geom = patchdata.amrcore->Geom(0);
    const CCTK_REAL *restrict const dx = geom.CellSize();
    for (int d = 0; d < dim; ++d)
      mindx = fmin(mindx, dx[d]);
  }
  return mindx;
}

CCTK_REAL get_finest_mindx() {
  CCTK_REAL mindx = std::numeric_limits<CCTK_REAL>::infinity();
  for (const auto &patchdata : ghext->patches()) {
    const amrex::Geometry &geom = patchdata.amrcore->Geom(0);
    const CCTK_REAL *restrict const dx = geom.CellSize();
    CCTK_REAL mindx1 = std::numeric_limits<CCTK_REAL>::infinity();
    for (int d = 0; d < dim; ++d)
      mindx1 = fmin(mindx1, dx[d]);
    mindx1 = ldexp(mindx1, -(patchdata.num_levels() - 1));
    mindx = fmin(mindx, mindx1);
  }
  return mindx;
}

static void SetDeltaTime(cGH *restrict const cctkGH) {
  DECLARE_CCTK_PARAMETERS;
  if (CCTK_EQUALS(timestep_choice, "timestep")) {
    cctkGH->cctk_delta_time = timestep;
  } else if (CCTK_EQUALS(timestep_choice, "dtfac")) {
    cctkGH->cctk_delta_time =
        dtfac *
        (ghext->use_subcycling ? get_coarse_mindx() : get_finest_mindx());
  } else {
    CCTK_ERROR("Unexpected value for 'CarpetX::timestep_choice'");
    abort();
  }
  using std::isfinite;
  assert(isfinite(cctkGH->cctk_delta_time));
  update_cctkGHs(cctkGH);
}

// The time that a window sees once `level` has stepped to `clock` (in coarse
// steps). Evolve sets it for each batch before CCTK_PRESTEP, recovery for
// each clock group before CCTK_POST_RECOVER_VARIABLES, with `level` the
// group's finest level, and FinalizeWindow for the finest clock group before
// the end of step, with `level` the finest level, so the three cannot drift
// apart: recovery from an initial checkpoint (one clock group, every clock 0)
// sees what the fresh run saw. cctk_iteration is not
// a function of the clock, so each caller sets it itself. Without subcycling
// cctk_time is not one either (Evolve accumulates it, recovery keeps the
// checkpointed value), so only cctk_timefac is set.
static void SetStepTime(cGH *restrict const cctkGH, const int level,
                        const rat64 clock) {
  if (ghext->use_subcycling) {
    cctkGH->cctk_timefac = 1 << level;
    cctkGH->cctk_time = cctkGH->cctk_delta_time * double(clock);
  } else {
    cctkGH->cctk_timefac = 1;
  }
}

static void RestoreLevelClocks(const cGH *restrict const cctkGH,
                               const checkpoint_record_t &record) {
  const int checkpoint_iteration = cctkGH->cctk_iteration;

  const int iteration_ratio = 1 << (ghext->num_levels() - 1);
  bool have_last_step_iterations = true;
  for (auto &patchdata : ghext->patches()) {
    for (auto &leveldata : patchdata.levels()) {
      const level_record_t &recovered =
          record.levels.at(leveldata.level.patch().index())
              .at(leveldata.level.index());
      if (recovered.iteration)
        leveldata.iteration = *recovered.iteration;
      else
        leveldata.iteration = rat64(checkpoint_iteration) / iteration_ratio;
      if (recovered.last_step_iteration)
        leveldata.last_step_iteration = *recovered.last_step_iteration;
      else
        have_last_step_iterations = false;
    }
  }

  const auto clock_groups = ClockGroups();
  const int num_groups = clock_groups.size();

  std::ostringstream clocks_buf;
  for (const auto &leveldata : ghext->patch(patch_t{0}).levels())
    clocks_buf << (leveldata.level.index() == 0 ? "" : ", ") << "L"
               << leveldata.level.index() << " "
               << format_clock(leveldata.iteration);
  const std::string clocks = clocks_buf.str();
#pragma omp critical
  CCTK_VINFO("Level clocks (coarse steps): %s -> %d clock group%s",
             clocks.c_str(), num_groups, num_groups == 1 ? "" : "s");

  if (!have_last_step_iterations) {
    for (const window_t &group : clock_groups) {
      const auto [min_level, max_level] = group.level_bounds();
      const int last_step_iteration =
          LastStepIteration(group.finest(), checkpoint_iteration);
      for (auto &patchdata : ghext->patches())
        for (auto &leveldata : patchdata.levels())
          if (leveldata.level.index() >= min_level &&
              leveldata.level.index() < max_level)
            leveldata.last_step_iteration = last_step_iteration;
    }
    if (num_groups >= 2 && CCTK_MyProc(nullptr) == 0)
      CCTK_VWARN(CCTK_WARN_ALERT,
                 "Checkpoint at iteration %d predates per-level last-step "
                 "iterations; derived them from the %d distinct level clocks "
                 "(%s), assuming that no regrid has changed the number of "
                 "levels since the coarsest clock group last stepped.",
                 checkpoint_iteration, num_groups, clocks.c_str());
  }

  for (const window_t &group : clock_groups) {
    const auto [min_level, max_level] = group.level_bounds();
    const int group_last_step_iteration =
        ghext->level(group.finest()).last_step_iteration;
    for (const auto &patchdata : ghext->patches())
      for (const auto &leveldata : patchdata.levels())
        if (leveldata.level.index() >= min_level &&
            leveldata.level.index() < max_level &&
            leveldata.last_step_iteration != group_last_step_iteration)
          CCTK_VERROR("Checkpoint at iteration %d is inconsistent: patch %d "
                      "level %d has last-step iteration %d, but its "
                      "clock group [%d,%d) has %d",
                      checkpoint_iteration, leveldata.level.patch().index(),
                      leveldata.level.index(), leveldata.last_step_iteration,
                      min_level, max_level, group_last_step_iteration);
  }

  const int finest_last_step_iteration =
      ghext->level(all_levels().finest()).last_step_iteration;
  if (finest_last_step_iteration != checkpoint_iteration)
    CCTK_VERROR("Checkpoint at iteration %d is inconsistent: its finest clock "
                "group has last-step iteration %d",
                checkpoint_iteration, finest_last_step_iteration);
}

static void RecoverHierarchy(cGH *restrict const cctkGH) {
  const checkpoint_record_t record = RecoverGridStructure(cctkGH);

  assert(!active_levels);
  active_levels = all_levels();

  CCTK_Traverse(cctkGH, "CCTK_BASEGRID");

  const char *recovery_mode = *static_cast<const char *const *>(
      CCTK_ParameterGet("recovery_mode", "Cactus", nullptr));
  if (!CCTK_Equals(recovery_mode, "strict")) {
    // Set up initial conditions
    CCTK_Traverse(cctkGH, "CCTK_INITIAL");
    CCTK_Traverse(cctkGH, "CCTK_POSTINITIAL");
    CCTK_Traverse(cctkGH, "CCTK_POSTPOSTINITIAL");
  }

  RecoverGH(cctkGH, record);
  CCTK_Traverse(cctkGH, "CCTK_RECOVER_VARIABLES");

  active_levels = std::nullopt;

  for (const auto &patchdata : ghext->patches())
    for (const auto &leveldata : patchdata.levels())
      assert(leveldata.iteration == 0 &&
             "recovery runs BASEGRID and RECOVER_VARIABLES at clock 0");

  RestoreLevelClocks(cctkGH, record);

  for (auto &patchdata : ghext->patches())
    patchdata.amrcore->cactus_is_initialized = true;

  SetDeltaTime(cctkGH);
}

namespace {
// Marks the per-group traversals of CCTK_POST_RECOVER_VARIABLES
// (GHExt::in_recovery_traversal) for error messages, like mark_sync_active
struct recovery_traversal_scope {
  recovery_traversal_scope() {
    assert(!ghext->in_recovery_traversal);
    ghext->in_recovery_traversal = true;
  }
  ~recovery_traversal_scope() { ghext->in_recovery_traversal = false; }
  recovery_traversal_scope(const recovery_traversal_scope &) = delete;
  recovery_traversal_scope &
  operator=(const recovery_traversal_scope &) = delete;
};

// Marks the driver's own traversal of a bin on the end-of-step window
// (GHExt::tail_bin): CCTK_POSTSTEP and CCTK_ANALYSIS in Evolve and
// Initialise, CCTK_CHECKPOINT in Evolve, and CCTK_TERMINATE in Shutdown. Under
// subcycling, SyncGroupsByDirISubcycling aborts on a sync while it is set,
// unless the window holds every level.
struct tail_traversal_scope {
  explicit tail_traversal_scope(const char *const bin) {
    assert(bin);
    assert(!ghext->tail_bin);
    ghext->tail_bin = bin;
  }
  ~tail_traversal_scope() { ghext->tail_bin = nullptr; }
  tail_traversal_scope(const tail_traversal_scope &) = delete;
  tail_traversal_scope &operator=(const tail_traversal_scope &) = delete;
};

// Traverse a bin on the end-of-step window, marked (tail_traversal_scope)
void TraverseTailBin(cGH *restrict const cctkGH, const char *const bin) {
  const tail_traversal_scope scope(bin);
  CCTK_Traverse(cctkGH, bin);
}

void FinalizeWindow(const window_t window, cGH *restrict const cctkGH,
                    const bool reflux) {
  DECLARE_CCTK_PARAMETERS;

  assert(FinestClockGroup() == window);
  const auto &finest = ghext->level(window.finest());
  SetStepTime(cctkGH, finest.level.index(), finest.iteration);
  loop_serially(window, [&](auto &restrict leveldata) {
    assert(leveldata.last_step_iteration == cctkGH->cctk_iteration);
  });

  const int min_level = window.level_bounds().min;
  if (!window_is_restricted(window)) {
    // Only the finest level is at the current time: nothing to restrict
    if (verbose)
#pragma omp critical
      CCTK_VINFO("Skipping restriction at iteration %d: only level %d is "
                 "at the current time",
                 cctkGH->cctk_iteration, min_level);
  } else if (!restrict_during_sync) {
    // Pre-restriction hook, traversed over the window before the reflux and
    // the restriction below, so that it sees the state and its dependents as
    // the last RK stage's ODESolvers_PostStep left them, consistent
    // everywhere
    CCTK_Traverse(cctkGH, "CarpetX_PreRestrict");
    if (reflux)
      Reflux(cctkGH, window);
    assert(do_restrict);
    const std::vector<gi_t> groups = collect_restrictable_groups();
    if (ghext->use_subcycling)
      RestrictNoPoison(cctkGH, window, groups);
    else
      Restrict(cctkGH, window, groups);
    ProlongateAfterRestriction(window, cctkGH);
    CCTK_Traverse(cctkGH, "CCTK_POSTRESTRICT");
  }
}
} // namespace

static void RecomputePerClockGroup(cGH *restrict const cctkGH) {
  const recovery_traversal_scope scope;

  for (const window_t &group : ClockGroups()) {
    const auto [min_level, max_level] = group.level_bounds();
    const auto &finest = ghext->level(group.finest());

    assert(!active_levels);
    active_levels = group;

    cctkGH->cctk_iteration = finest.last_step_iteration;
    SetStepTime(cctkGH, finest.level.index(), finest.iteration);

#pragma omp critical
    CCTK_VINFO("POST_RECOVER_VARIABLES on levels [%d,%d): iteration %d, "
               "time %g, timefac %d",
               min_level, max_level, cctkGH->cctk_iteration,
               double(cctkGH->cctk_time), cctkGH->cctk_timefac);

    RecoverRKBoundary(group, cctkGH);
    CCTK_Traverse(cctkGH, "CCTK_POST_RECOVER_VARIABLES");

    active_levels = std::nullopt;
  }

#pragma omp critical
  CCTK_VINFO("Recovery does not rerun restriction, POSTSTEP or ANALYSIS");
}

static void RecoverFromCheckpoint(cGH *restrict const cctkGH) {
  DECLARE_CCTK_PARAMETERS;

#pragma omp critical
  CCTK_VINFO("Recovering from checkpoint...");

  RecoverHierarchy(cctkGH);

  RecomputePerClockGroup(cctkGH);

  assert(cctkGH->cctk_iteration == recovered_iteration());

#pragma omp critical
  CCTK_VINFO("Iteration: %d   time: %g   delta_time: %g",
             cctkGH->cctk_iteration, double(cctkGH->cctk_time),
             double(cctkGH->cctk_delta_time));

  assert(!active_levels);
  active_levels = FinestClockGroup();

  CCTK_Traverse(cctkGH, "CCTK_CPINITIAL");
  if (out_initial_data)
    CCTK_OutputGH(cctkGH);

  active_levels = std::nullopt;
}

int Initialise(tFleshConfig *config) {
  DECLARE_CCTK_PARAMETERS;

  static Timer timer("Initialise");
  Interval interval(timer);

  cGH *restrict const cctkGH = CCTK_SetupGH(config, 0);
  CCTKi_AddGH(config, 0, cctkGH);

  // Check presync mode
  if (!CCTK_EQUALS(presync_mode, "mixed-error") &&
      !CCTK_EQUALS(presync_mode, "presync-only"))
    CCTK_ERROR("CarpetX currently requires Cactus::presync_mode = "
               "\"mixed-error\" or \"presync-only\"");

  // Check restrict_during_sync when use_subcycling is on
  if (use_subcycling && restrict_during_sync)
    CCTK_ERROR("CarpetX requires CarpetX::restrict_during_sync = "
               "\"no\" when CarpetX::use_subcycling = \"yes\"");

  // Check presync_mode when use_subcycling is on: in presync-only mode
  // CallFunction and CycleTimelevels sync through SyncGroupsByDirI, which
  // bypasses the subcycling sync (SyncGroupsByDirISubcycling), its time
  // interpolation and its end-of-step check
  if (use_subcycling && CCTK_EQUALS(presync_mode, "presync-only"))
    CCTK_ERROR("CarpetX requires Cactus::presync_mode = \"mixed-error\" when "
               "CarpetX::use_subcycling = \"yes\"");

  // Initialise iteration and time
  cctkGH->cctk_iteration = 0;
  cctkGH->cctk_time = *static_cast<const CCTK_REAL *>(
      CCTK_ParameterGet("cctk_initial_time", "Cactus", nullptr));

  // Initialise schedule
  CCTKi_ScheduleGHInit(cctkGH);

  ghext->freeze_storage();

  // Initialise all grid extensions
  CCTKi_InitGHExtensions(cctkGH);

  // Set up cctkGH
  setup_cctkGH(cctkGH);
  enter_global_mode(cctkGH);
  ghext->global_cctkGH = GHExt::cctkGHptr(copy_cctkGH(cctkGH));

  for (const auto &patchdata : ghext->patches())
    assert(patchdata.num_levels() == 0);
  assert(!active_levels);
  active_levels = window_t::levels(0, 0);

  CCTK_Traverse(cctkGH, "CCTK_WRAGH");
  CCTK_Traverse(cctkGH, "CCTK_PARAMCHECK");
  CCTKi_FinaliseParamWarn();
  warn_inert_flux_tags();

  active_levels = std::nullopt;

  CCTK_VINFO("Setting initial values for max_grid_size values for all levels");
  for (const auto &patchdata : ghext->patches()) {
    amrex::Vector<amrex::IntVect> max_grid_sizes_vec(20);
    for (int level = 0; level < 20; ++level) {
      int size_x = max_grid_sizes_x[level];
      if (size_x == -1)
        size_x = max_grid_size_x;
      int size_y = max_grid_sizes_y[level];
      if (size_y == -1)
        size_y = max_grid_size_y;
      int size_z = max_grid_sizes_z[level];
      if (size_z == -1)
        size_z = max_grid_size_z;
      const amrex::IntVect max_grid_size_vec{size_x, size_y, size_z};
      max_grid_sizes_vec.at(level) = max_grid_size_vec;
    }
    patchdata.amrcore->SetMaxGridSize(max_grid_sizes_vec);
  }

  if (config->recovered) {
    // Recovery runs its own sequence and skips the fresh-start tail below
    RecoverFromCheckpoint(cctkGH);
    return 0;

  } else {
    // Set up initial conditions
#pragma omp critical
    CCTK_VINFO("Setting up initial conditions...");

    {
      static Timer timer("InitialiseRegrid [coarse]");
      Interval interval(timer);

      const CCTK_REAL time = 0; // dummy time
      for (const auto &patchdata : ghext->patches())
        patchdata.amrcore->MakeNewGrids(time);

      SetDeltaTime(cctkGH);
#pragma omp critical
      CCTK_VINFO("Iteration: %d   time: %g   delta_time: %g",
                 cctkGH->cctk_iteration, double(cctkGH->cctk_time),
                 double(cctkGH->cctk_delta_time));

      assert(!active_levels);
      active_levels = window_t::levels(0, 1);
      CCTK_Traverse(cctkGH, "CCTK_BASEGRID");
      active_levels = std::nullopt;
    }

    if (CCTK_MyProc(nullptr) == 0) {
      for (const auto &patchdata : ghext->patches()) {
        const cGH *const patchGH =
            patchdata.level(level_t{patchdata.patch, 0}).get_patch_cctkGH();
        const int *restrict const gsh = patchGH->cctk_gsh;
        const int *restrict const nghostzones = patchGH->cctk_nghostzones;
        CCTK_REAL x0[dim], x1[dim], dx[dim];
        for (int d = 0; d < dim; ++d) {
          dx[d] = patchGH->cctk_delta_space[d];
          x0[d] = patchGH->cctk_origin_space[d] +
                  (2 * nghostzones[d] - 1) * dx[d] / 2;
          x1[d] = x0[d] + (gsh[d] - 2 * nghostzones[d] - 1) * dx[d];
        }
#pragma omp critical
        {
          CCTK_VINFO("Patch %d:", patchdata.patch.index());
          CCTK_VINFO("  Grid extent:");
          CCTK_VINFO("    gsh=[%d,%d,%d]", gsh[0], gsh[1], gsh[2]);
          const auto &bf = patchdata.amrcore->blockingFactor(0);
          CCTK_VINFO("    blocking_factor=[%d,%d,%d]", bf[0], bf[1], bf[2]);
          const auto &mgs = patchdata.amrcore->maxGridSize(0);
          CCTK_VINFO("    max_grid_size=[%d,%d,%d]", mgs[0], mgs[1], mgs[2]);
          const auto mfitinfo = amrex::MFItInfo().EnableTiling();
          const auto &ts = mfitinfo.tilesize;
          CCTK_VINFO("    max_tile_size=[%d,%d,%d]", ts[0], ts[1], ts[2]);
          CCTK_VINFO("  Domain extent:");
          CCTK_VINFO("    xmin=[%.17g,%.17g,%.17g]", double(x0[0]),
                     double(x0[1]), double(x0[2]));
          CCTK_VINFO("    xmax=[%.17g,%.17g,%.17g]", double(x1[0]),
                     double(x1[1]), double(x1[2]));
          CCTK_VINFO("    base dx=[%.17g,%.17g,%.17g]", double(dx[0]),
                     double(dx[1]), double(dx[2]));
        }
      }
    }

    for (auto &patchdata : ghext->patches())
      patchdata.amrcore->cactus_is_initialized = true;

    for (;;) {
      const int level = ghext->num_levels() - 1;
#pragma omp critical
      CCTK_VINFO("Initializing level %d...", level);

      for (const auto &patchdata : ghext->patches())
        assert(patchdata.amrcore->finestLevel() <= level);

      assert(!active_levels);
      active_levels = all_levels();

      InputGH(cctkGH);
      CCTK_Traverse(cctkGH, "CCTK_INITIAL");
      CCTK_Traverse(cctkGH, "CCTK_POSTINITIAL");
      CCTK_Traverse(cctkGH, "CCTK_POSTPOSTINITIAL");

      active_levels = std::nullopt;

      bool did_modify_any_level;
      {
#pragma omp critical
        CCTK_VINFO("Regridding...");
        static Timer timer("InitialiseRegrid [refined]");
        Interval interval(timer);

        for (const auto &patchdata : ghext->patches()) {

          const int old_numlevels = patchdata.amrcore->finestLevel() + 1;
          patchdata.amrcore->level_modified.clear();
          patchdata.amrcore->level_modified.resize(old_numlevels, false);
          const CCTK_REAL time = 0; // dummy time
          patchdata.amrcore->regrid(0, time);

          const int new_numlevels = patchdata.amrcore->finestLevel() + 1;
          const int max_numlevels = patchdata.amrcore->maxLevel() + 1;
          assert(new_numlevels >= 0 && new_numlevels <= max_numlevels);
          assert(new_numlevels == old_numlevels ||
                 new_numlevels == old_numlevels + 1);

#pragma omp critical
          {
            const double pts0 = patchdata.level(level_t{patchdata.patch, 0})
                                    .fab->boxArray()
                                    .d_numPts();
            for (const auto &leveldata : patchdata.levels()) {
              const int sz = leveldata.fab->size();
              const double pts = leveldata.fab->boxArray().d_numPts();
              const int level = leveldata.level.index();
              if (level == 0) {
                CCTK_VINFO(
                    "  level %d: %d boxes, %.0f cells (%.4g%%)", level, sz, pts,
                    100 * pts / (ldexp(CCTK_REAL(1), dim * level) * pts0));
              } else {
                const double ptsc = patchdata.level(leveldata.level.parent())
                                        .fab->boxArray()
                                        .d_numPts();
                CCTK_VINFO("  level %d: %d boxes, %.0f cells (%.4g%%, %.0f%%)",
                           level, sz, pts,
                           100 * pts /
                               (ldexp(CCTK_REAL(1), dim * level) * pts0),
                           100 * pts / (ldexp(CCTK_REAL(1), dim) * ptsc));
              }
            }
          } // omp critical
        } // for patchdata

        int min_modified_level = INT_MAX;
        int max_modified_level = 0;
        for (const auto &patchdata : ghext->patches()) {
          for (int lev = 0; lev < int(patchdata.amrcore->level_modified.size());
               ++lev) {
            if (patchdata.amrcore->level_modified.at(lev)) {
              using std::max, std::min;
              min_modified_level = min(min_modified_level, lev);
              max_modified_level = max(max_modified_level, lev + 1);
            }
          }
        }
        did_modify_any_level = min_modified_level < max_modified_level;

        if (did_modify_any_level) {
          SetDeltaTime(cctkGH);
#pragma omp critical
          CCTK_VINFO("Iteration: %d   time: %g   delta_time: %g",
                     cctkGH->cctk_iteration, double(cctkGH->cctk_time),
                     double(cctkGH->cctk_delta_time));

          assert(!active_levels);
          active_levels =
              window_t::levels(min_modified_level, max_modified_level);
          CCTK_Traverse(cctkGH, "CCTK_BASEGRID");
          CCTK_Traverse(cctkGH, "CCTK_POSTREGRID");
          active_levels = std::nullopt;
        }
      } // Regrid

      if (!did_modify_any_level)
        break;
    } // for level
  }
#pragma omp critical
  CCTK_VINFO("Initialized %d levels", ghext->num_levels());

  assert(!active_levels);
  const window_t window = FinestClockGroup();
  active_levels = window;
  FinalizeWindow(window, cctkGH, false);

  // Checkpoint, analysis, output. The end-of-step bins are marked as in
  // Evolve; their window here holds every level, so a sync in them is legal
  // also under subcycling.
  TraverseTailBin(cctkGH, "CCTK_POSTSTEP");
  CCTK_Traverse(cctkGH, "CCTK_CPINITIAL");
  TraverseTailBin(cctkGH, "CCTK_ANALYSIS");
  if (out_initial_data) {
    CCTK_OutputGH(cctkGH);
  }

  active_levels = std::nullopt;

  return 0;
} // namespace CarpetX

bool EvolutionIsDone(cGH *restrict const cctkGH) {
  DECLARE_CCTK_PARAMETERS;

  const bool max_iteration_reached = cctkGH->cctk_iteration >= cctk_itlast;

  const bool max_simulation_time_reached =
      cctk_initial_time < cctk_final_time
          ? cctkGH->cctk_time >= cctk_final_time
          : cctkGH->cctk_time <= cctk_final_time;

  const int runtime = CCTK_RunTime();
  const bool max_runtime_reached = runtime >= 60 * max_runtime;

  // Note: Some MPI implementations have been built without support
  // for `MPI_CXX_BOOL`, so we use `int` instead
  int we_are_done;
  if (terminate_next || CCTK_TerminationReached(cctkGH))
    we_are_done = true;
  else if (CCTK_Equals(terminate, "never"))
    we_are_done = false;
  else if (CCTK_Equals(terminate, "iteration"))
    we_are_done = max_iteration_reached;
  else if (CCTK_Equals(terminate, "time"))
    we_are_done = max_simulation_time_reached;
  else if (CCTK_Equals(terminate, "runtime"))
    we_are_done = max_runtime_reached;
  else if (CCTK_Equals(terminate, "any"))
    we_are_done = max_iteration_reached || max_simulation_time_reached ||
                  max_runtime_reached;
  else if (CCTK_Equals(terminate, "all"))
    we_are_done = max_iteration_reached && max_simulation_time_reached &&
                  max_runtime_reached;
  else if (CCTK_Equals(terminate, "either"))
    we_are_done = max_iteration_reached || max_simulation_time_reached;
  else if (CCTK_Equals(terminate, "both"))
    we_are_done = max_iteration_reached && max_simulation_time_reached;
  else
    CCTK_ERROR("internal error");

  // Ensure all processes make the same decision
  MPI_Allreduce(MPI_IN_PLACE, &we_are_done, 1, MPI_INT, MPI_LOR,
                MPI_COMM_WORLD);

  return we_are_done;
}

namespace {

struct seed_run_t {
  window_t levels;
  valid_t seed;
};

std::vector<seed_run_t> seed_runs(const window_t window, const gi_t gi,
                                  const vi_t vi) {
  const auto levels = window.level_bounds();
  std::vector<std::optional<valid_t> > seeds(levels.max - levels.min);
  loop_serially(window, [&](auto &restrict leveldata) {
    std::optional<valid_t> &seed =
        seeds.at(leveldata.level.index() - levels.min);
    const valid_t level_seed = leveldata.group(gi).timelevel(tl_t{1}).valid(vi);
    assert(!seed || *seed == level_seed);
    seed = level_seed;
  });
  std::vector<seed_run_t> runs;
  for (int lo = levels.min; lo < levels.max;) {
    const std::optional<valid_t> seed = seeds.at(lo - levels.min);
    int hi = lo + 1;
    while (hi < levels.max && seeds.at(hi - levels.min) == seed)
      ++hi;
    if (seed)
      runs.push_back({window.narrow(lo, hi), *seed});
    lo = hi;
  }
  return runs;
}

} // namespace

void CycleTimelevels(const window_t window, cGH *restrict const cctkGH) {
  DECLARE_CCTK_PARAMETERS;

  static Timer timer("CycleTimelevels");
  Interval interval(timer);

  update_cctkGHs(cctkGH);

  const int num_groups = CCTK_NumGroups();
  const bool presync_only = CCTK_EQUALS(presync_mode, "presync-only");
  std::vector<gi_t> presync_groups;
  for (int gi0 = 0; gi0 < num_groups; ++gi0) {
    const gi_t gi{gi0};
    const auto &group = ghext->record(gi);
    const int ntls0 = group.num_timelevels();
    if (ntls0 == 0)
      continue;

    if (group.kind == group_kind::grid_function) {
      loop_serially(window, [&](auto &restrict leveldata) {
        auto &restrict groupdata = leveldata.group(gi);
        const int ntls = groupdata.num_timelevels();
        assert(ntls == ntls0);
        if (ntls > 1) {
          groupdata.rotate_timelevels();
          for (auto &restrict local_cctkGH_ptr : leveldata.local_cctkGHs) {
            cGH *const local_cctkGH = local_cctkGH_ptr.get();
            const int component = local_cctkGH->cctk_component;
            const GridPtrDesc1 grid1(leveldata, groupdata, component);
            for (const tl_t tl : groupdata.record.timelevels()) {
              const amrex::Array4<CCTK_REAL> vars =
                  groupdata.timelevel(tl).mfab->array(component);
              for (const vi_t vi : groupdata.record.variables())
                local_cctkGH
                    ->data[groupdata.record.var(vi).index()][tl.index()] =
                    grid1.ptr(vars, vi);
            }
          }
        }
      });
      if (ntls0 > 1) {
        if (group.do_evolve) {
          invalidate(window, {.gi = gi,
                              .tl = tl_t{0},
                              .where = make_valid_all(),
                              .event = "CycleTimelevels"});
        } else {
          std::vector<write_scope_t> scopes;
          for (const vi_t vi : group.variables())
            for (const seed_run_t &run : seed_runs(window, gi, vi))
              scopes.push_back(open_write(
                  run.levels,
                  {.gi = gi,
                   .tl = tl_t{0},
                   .vi = vi,
                   .fills = run.seed,
                   .keeps = {},
                   .event = "CycleTimelevels (seed from old time level)"}));
          loop_serially(window, [&](auto &restrict leveldata) {
            auto &restrict groupdata = leveldata.group(gi);
            amrex::MultiFab::Copy(
                *groupdata.timelevel(tl_t{0}).mfab,
                *groupdata.timelevel(tl_t{1}).mfab, 0, 0,
                groupdata.record.numvars,
                groupdata.timelevel(tl_t{0}).mfab->nGrowVect());
          });
        }
      }
      if (group.do_evolve) {
        for (const tl_t tl : group.timelevels()) {
          if (ntls0 > 1 && tl == tl_t{0})
            continue;
          if (tl == tl_t{0} && presync_only)
            presync_groups.push_back(gi);
          else
            require(window, {.gi = gi,
                             .tl = tl,
                             .where = make_valid_all(),
                             .event = "CycleTimelevels for the state vector"});
        }
      }
    } else {
      if (ntls0 > 1) {
        ghext->record(gi).array().rotate_timelevels();
        invalidate_array({.gi = gi,
                          .tl = tl_t{0},
                          .where = make_valid_int(),
                          .event = "CycleTimelevels"});
      }
    }

  } // for gi
  if (!presync_groups.empty()) {
    SyncGroupsByDirI(window, cctkGH, presync_groups);
  }
}

int Evolve(tFleshConfig *config) {
  DECLARE_CCTK_PARAMETERS;

  static Timer timer("Evolve");
  Interval interval(timer);

  assert(config);
  cGH *restrict const cctkGH = config->GH[0];
  assert(cctkGH);

#pragma omp critical
  CCTK_VINFO("Starting evolution...");

  double total_evolution_time = 0;
  double total_evolution_output_time = 0;
  int total_iterations = 0;
  double total_cell_updates = 0;
  int average_iteration_time_iterations = 0;
  double average_iteration_time = 0;

  std::ofstream performance_file;
  if (out_performance && CCTK_MyProc(NULL) == 0) {
    const int every =
        out_performance_every == -1 ? out_every : out_performance_every;
    if (every > 0) {
      std::ostringstream buf;
      buf << out_dir << "/performance.yaml";
      const std::string filename = buf.str();
      performance_file.open(filename);
      performance_file << "performance:\n" << std::flush;
    }
  }

  while (!EvolutionIsDone(cctkGH)) {

    const double start_time = gettime();

    assert(!active_levels);

    rat64 iteration = ghext->patches().front().levels().front().iteration;
    using std::min;
    for (const auto &patchdata : ghext->patches())
      for (const auto &leveldata : patchdata.levels())
        iteration = min(iteration, leveldata.iteration);

    if (regrid_every > 0 && cctkGH->cctk_iteration % regrid_every == 0) {
#pragma omp critical
      CCTK_VINFO("Regridding...");
      static Timer timer("EvolveRegrid");
      Interval interval(timer);

      for (const auto &patchdata : ghext->patches()) {

        int min_active_level = -1;
        for (int level = patchdata.num_levels() - 1; level >= 0; --level) {
          if (patchdata.level(level_t{patchdata.patch, level}).iteration !=
              iteration) {
            break;
          }
          min_active_level = level;
        }
        assert(min_active_level != -1);

        const int old_numlevels = patchdata.amrcore->finestLevel() + 1;
        patchdata.amrcore->level_modified.clear();
        patchdata.amrcore->level_modified.resize(old_numlevels, false);
        const CCTK_REAL time = 0; // dummy time

        // Set the value of max_grid_size before regridding
        CCTK_VINFO(
            "Setting max_grid_size values for all levels before regridding");
        amrex::Vector<amrex::IntVect> max_grid_sizes_vec(20);
        for (int level = 0; level < 20; ++level) {
          int size_x = max_grid_sizes_x[level];
          if (size_x == -1)
            size_x = max_grid_size_x;
          int size_y = max_grid_sizes_y[level];
          if (size_y == -1)
            size_y = max_grid_size_y;
          int size_z = max_grid_sizes_z[level];
          if (size_z == -1)
            size_z = max_grid_size_z;
          const amrex::IntVect max_grid_size_vec{size_x, size_y, size_z};
          max_grid_sizes_vec.at(level) = max_grid_size_vec;
        }
        patchdata.amrcore->SetMaxGridSize(max_grid_sizes_vec);

        patchdata.amrcore->regrid(min_active_level, time);

        const int new_numlevels = patchdata.amrcore->finestLevel() + 1;
        const int max_numlevels = patchdata.amrcore->maxLevel() + 1;
        assert(new_numlevels >= 0 && new_numlevels <= max_numlevels);

#pragma omp critical
        {
          CCTK_VINFO("  old levels %d, new levels %d", old_numlevels,
                     new_numlevels);
          double pts0 = patchdata.level(level_t{patchdata.patch, 0})
                            .fab->boxArray()
                            .d_numPts();
          assert(!active_levels);
          for (const auto &leveldata : patchdata.levels()) {
            const int sz = leveldata.fab->size();
            const double pts = leveldata.fab->boxArray().d_numPts();
            const int level = leveldata.level.index();
            if (level == 0) {
              CCTK_VINFO("  level %d: %d boxes, %.0f cells (%.4g%%)", level, sz,
                         pts,
                         100 * pts / (ldexp(CCTK_REAL(1), dim * level) * pts0));
            } else {
              const double ptsc = patchdata.level(leveldata.level.parent())
                                      .fab->boxArray()
                                      .d_numPts();
              CCTK_VINFO("  level %d: %d boxes, %.0f cells (%.4g%%, %.0f%%)",
                         level, sz, pts,
                         100 * pts / (ldexp(CCTK_REAL(1), dim * level) * pts0),
                         100 * pts / (ldexp(CCTK_REAL(1), dim) * ptsc));
            }
          }
        } // omp critical
      } // for patchdata

      int min_modified_level = INT_MAX;
      int max_modified_level = 0;
      for (const auto &patchdata : ghext->patches()) {
        for (int lev = 0; lev < int(patchdata.amrcore->level_modified.size());
             ++lev) {
          if (patchdata.amrcore->level_modified.at(lev)) {
            using std::max, std::min;
            min_modified_level = min(min_modified_level, lev);
            max_modified_level = max(max_modified_level, lev + 1);
          }
        }
      }
      const bool did_modify_any_level = min_modified_level < max_modified_level;

      if (did_modify_any_level) {
        SetDeltaTime(cctkGH);
#pragma omp critical
        CCTK_VINFO("Iteration: %d   time: %g   delta_time: %g",
                   cctkGH->cctk_iteration, double(cctkGH->cctk_time),
                   double(cctkGH->cctk_delta_time));

        assert(!active_levels);
        active_levels =
            window_t::levels(min_modified_level, max_modified_level);
        CCTK_Traverse(cctkGH, "CCTK_BASEGRID");
        CCTK_Traverse(cctkGH, "CCTK_POSTREGRID");
        active_levels = std::nullopt;
      }
    } // Regrid

    rat64 delta_iteration =
        ghext->patches().front().levels().front().delta_iteration;
    for (const auto &patchdata : ghext->patches())
      for (const auto &leveldata : patchdata.levels())
        delta_iteration = min(delta_iteration, leveldata.delta_iteration);

    cctkGH->cctk_iteration += 1;

    for (int min_level = 0, max_level; min_level < ghext->num_levels();
         min_level = max_level) {
      max_level = min_level;
      while (++max_level < ghext->num_levels()) {
        bool level_is_subcycling_level = false;
        for (const auto &patchdata : ghext->patches())
          if (max_level < patchdata.num_levels())
            level_is_subcycling_level |=
                patchdata.level(level_t{patchdata.patch, max_level})
                    .is_subcycling_level;
        if (level_is_subcycling_level)
          break;
      }

      rat64 level_iteration = -1;
      rat64 level_delta_iteration = -1;
      for (const auto &patchdata : ghext->patches()) {
        if (min_level < patchdata.num_levels()) {
          const auto &leveldata =
              patchdata.level(level_t{patchdata.patch, min_level});
          level_iteration = leveldata.iteration;
          level_delta_iteration = leveldata.delta_iteration;
          break;
        }
      }
      assert(level_iteration != -1);
      assert(level_delta_iteration != -1);
      // Skip those evolved coarse levels while evolving the fine levels to
      // catch up
      if (level_iteration > iteration)
        continue;

      const window_t window = window_t::levels(min_level, max_level);
      active_levels = window;

      level_iteration += level_delta_iteration;
      loop_serially(window, [&](auto &restrict leveldata) {
        leveldata.iteration += leveldata.delta_iteration;
        assert(level_iteration == leveldata.iteration);
        assert(level_delta_iteration == leveldata.delta_iteration);
      });

      CycleTimelevels(window, cctkGH);

      // Without subcycling cctk_time accumulates; SetStepTime leaves it alone
      if (!ghext->use_subcycling)
        cctkGH->cctk_time += cctkGH->cctk_delta_time;
      SetStepTime(cctkGH, min_level, level_iteration);

      CCTK_Traverse(cctkGH, "CCTK_PRESTEP");
      CCTK_Traverse(cctkGH, "CCTK_EVOL");
      active_levels = std::nullopt;

      const window_t widened = window_t::levels(
          WidenMinLevel(window.coarsest(), level_iteration).index(), max_level);
      loop_serially(widened, [&](auto &restrict leveldata) {
        leveldata.last_step_iteration = cctkGH->cctk_iteration;
      });
    } // for min_level, max_level

    assert(!active_levels);
    const window_t window = FinestClockGroup();
    active_levels = window;
    FinalizeWindow(window, cctkGH, ghext->use_subcycling);
    TraverseTailBin(cctkGH, "CCTK_POSTSTEP");
    TraverseTailBin(cctkGH, "CCTK_ANALYSIS");

    loop_serially(window, [&](auto &restrict leveldata) {
      leveldata.last_finalized_iteration = cctkGH->cctk_iteration;
    });

    TraverseTailBin(cctkGH, "CCTK_CHECKPOINT");
    const double output_start_time = gettime();
    CCTK_OutputGH(cctkGH);
    const double output_finish_time = gettime();
    total_evolution_output_time += output_finish_time - output_start_time;

    active_levels = std::nullopt;

    const double waiting_start_time = gettime();
    MPI_Barrier(MPI_COMM_WORLD);
    const double waiting_finish_time = gettime();

    const double finish_time = gettime();
    double num_cells = 0;
    for (const auto &patch : ghext->patches())
      for (const auto &level : patch.levels())
        num_cells += level.fab->boxArray().d_numPts();
    total_cell_updates += num_cells;
    ++total_iterations;
    const double iteration_time = finish_time - start_time;
    total_evolution_time += iteration_time;

    // Calculate statistics
    if (average_iteration_time_iterations < 10)
      ++average_iteration_time_iterations;
    // Calculate exponential moving average
    average_iteration_time =
        ((average_iteration_time_iterations - 1) * average_iteration_time +
         iteration_time) /
        average_iteration_time_iterations;

    const double iterations_per_second = 1 / average_iteration_time;
    const double cell_updates_per_second = num_cells * iterations_per_second;
    CCTK_VINFO("Simulation time: %g   "
               "Iterations per second: %g   "
               "Simulation time per second: %g",
               double(cctkGH->cctk_time), iterations_per_second,
               double(cctkGH->cctk_delta_time * iterations_per_second)

    );
    // This is the same as H-AMR's "cell updates per second":
    CCTK_VINFO("Grid cells: %g   "
               "Grid cell updates per second: %g",
               num_cells, cell_updates_per_second);

    const double total_evolution_compute_time =
        total_evolution_time - total_evolution_output_time;
    CCTK_VINFO("Performance:");
    CCTK_VINFO("  total evolution time:            %g sec",
               total_evolution_time);
    CCTK_VINFO("  total evolution compute time:    %g sec",
               total_evolution_compute_time);
    CCTK_VINFO("  total evolution output time:     %g sec",
               total_evolution_output_time);
    CCTK_VINFO("  total iterations:                %d", total_iterations);
    CCTK_VINFO("  total cells updated:             %g", total_cell_updates);
    CCTK_VINFO("  average iterations per second: %g",
               total_iterations / total_evolution_time);
    CCTK_VINFO("  average cell updates per second: %g",
               total_cell_updates / total_evolution_time);
    // TODO: Output this in a proper I/O method
    if (out_performance && CCTK_MyProc(NULL) == 0) {
      const int every =
          out_performance_every == -1 ? out_every : out_performance_every;
      if (every > 0 && cctkGH->cctk_iteration % every == 0)
        performance_file << "  " << total_iterations << ":\n"
                         << "    evolution-seconds: " << total_evolution_time
                         << "\n"
                         << "    evolution-compute-seconds: "
                         << total_evolution_compute_time << "\n"
                         << "    evolution-output-seconds: "
                         << total_evolution_output_time << "\n"
                         << "    evolution-cell-updates: " << total_cell_updates
                         << "\n"
                         << "    evolution-iterations: " << total_iterations
                         << "\n"
                         << std::flush;
    }

  } // main loop

  if (out_performance && CCTK_MyProc(NULL) == 0)
    performance_file.close();

  return 0;
} // namespace CarpetX

int Shutdown(tFleshConfig *config) {
  assert(config);
  cGH *restrict const cctkGH = config->GH[0];
  assert(cctkGH);

  static Timer timer("Shutdown");
  Interval interval(timer);

#pragma omp critical
  CCTK_VINFO("Shutting down...");

  assert(!active_levels);
  active_levels = FinestClockGroup();

  TraverseTailBin(cctkGH, "CCTK_TERMINATE");

  active_levels = std::nullopt;
  active_levels = window_t::levels(0, 0);

  CCTK_Traverse(cctkGH, "CCTK_SHUTDOWN");

  active_levels = std::nullopt;
  assert(!ghext);

  return 0;
}

static void check_storage_clauses(const cFunctionData *restrict attribute) {
  int offender_count = 0;

  const auto group_inactive = [](const gi_t gi) {
    return !ghext->record(gi).has_storage();
  };

  const auto report = [&](const char *clause_kind, const gi_t gi,
                          const std::optional<vi_t> vi, const tl_t tl) {
    ++offender_count;
    const char *const groupname = CCTK_FullGroupName(gi.index());
    if (vi) {
      const char *const varname =
          CCTK_VarName(ghext->record(gi).var(*vi).index());
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

int CallFunction(void *function, cFunctionData *restrict attribute,
                 void *data) {
  DECLARE_CCTK_PARAMETERS;

  assert(function);
  assert(attribute);
  assert(data);

  cGH *restrict const cctkGH = static_cast<cGH *>(data);

  if (verbose)
#pragma omp critical
    CCTK_VINFO("CallFunction iteration %d %s: %s::%s", cctkGH->cctk_iteration,
               attribute->where, attribute->thorn, attribute->routine);

  // For the end-of-step sync check's message. ghext is gone once
  // CarpetX_Shutdown has run, while meta-mode routines are still called.
  if (ghext)
    ghext->last_called_function = attribute;

  const std::string event = std::string("CallFunction ") + attribute->where +
                            ": " + attribute->thorn + "::" + attribute->routine;

  static std::map<cFunctionData *restrict, Timer> timers;

  std::map<cFunctionData *restrict, Timer>::iterator timer_iter;
#pragma omp critical(CarpetX_CallFunction)
  {
    timer_iter = timers.find(attribute);
    if (timer_iter == timers.end())
      timer_iter = std::get<0>(timers.emplace(attribute, event));
  }
  Timer &timer = timer_iter->second;
  Interval interval(timer);

  assert(active_levels);

  {
    static std::unordered_set<cFunctionData *> validated;
    cFunctionData *const key = attribute;
    bool needs_check;
#pragma omp critical(CarpetX_CallFunction)
    {
      needs_check = validated.insert(key).second;
    }
    if (needs_check)
      check_storage_clauses(attribute);
  }

  if (CCTK_EQUALS(presync_mode, "presync-only")) {
    const std::vector<clause_t> &reads =
        decode_clauses(attribute, rdwr_t::read);
    std::set<gi_t> sync_set;
    for (const auto &rd : reads) {
      if (CCTK_GroupTypeI(rd.gi.index()) == CCTK_GF) {

        loop_serially(*active_levels, [&](const auto &restrict leveldata) {
          const valid_t &need = rd.valid;
          const valid_t have =
              leveldata.group(rd.gi).timelevel(rd.tl).valid(rd.vi);
          if (need.valid_ghosts && !have.valid_ghosts && have.valid_int)
            sync_set.insert(rd.gi);
        });
      }
    }
    if (!sync_set.empty()) {
      const std::vector<gi_t> sync_vec(sync_set.begin(), sync_set.end());
      SyncGroupsByDirI(*active_levels, cctkGH, sync_vec);
    }
  }

  {
    std::set<gi_t> touched;
    for (const rdwr_t rdwr : {rdwr_t::read, rdwr_t::write})
      for (const auto &clause : decode_clauses(attribute, rdwr))
        touched.insert(clause.gi);
    for (int n = 0; n < attribute->n_SyncGroups; ++n)
      touched.insert(gi_t{attribute->SyncGroups[n]});
    for (const gi_t gi : touched) {
      const auto &group = ghext->record(gi);
      if (group.kind != group_kind::grid_function || !group.has_storage())
        continue;
      loop_serially(*active_levels, [&](const auto &restrict leveldata) {
        const auto &restrict groupdata = leveldata.group(gi);
        const int open_writes = groupdata.open_write_count();
        if (open_writes > 0)
          CCTK_VERROR("CallFunction iteration %d %s: %s::%s reads, writes or "
                      "syncs group %s, which has %d open write scope%s "
                      "(CarpetX::write) on patch %d, refinement level %d; "
                      "destroy the write_t before scheduling the function",
                      cctkGH->cctk_iteration, attribute->where,
                      attribute->thorn, attribute->routine,
                      groupdata.record.groupname.c_str(), open_writes,
                      open_writes == 1 ? "" : "s",
                      groupdata.level.patch().index(), groupdata.level.index());
      });
    }
  }

  {
    const std::vector<clause_t> &reads =
        decode_clauses(attribute, rdwr_t::read);
    for (const auto &rd : reads) {
      if (!ghext->record(rd.gi).has_storage())
        continue;
      const require_spec_t spec{.gi = rd.gi,
                                .tl = rd.tl,
                                .vi = rd.vi,
                                .where = rd.valid,
                                .event = event + " checking input"};
      if (CCTK_GroupTypeI(rd.gi.index()) == CCTK_GF)
        require(*active_levels, spec);
      else
        require_array(spec);
    }
  }

  std::vector<write_scope_t> write_scopes;
  std::vector<array_write_scope_t> array_write_scopes;
  {
    std::map<clause_t, valid_t> isread;
    const std::vector<clause_t> &reads =
        decode_clauses(attribute, rdwr_t::read);
    for (const auto &rd : reads) {
      clause_t cl = rd;
      cl.valid = valid_t();
      assert(isread.count(cl) == 0);
      isread[cl] = rd.valid;
    }
    const std::vector<clause_t> &writes =
        decode_clauses(attribute, rdwr_t::write);
    for (const auto &wr : writes) {
      const GHExt::GroupRecord &record = ghext->record(wr.gi);
      if (!record.has_storage())
        continue;
      if (!record.timelevels().contains(wr.tl))
        CCTK_VERROR("Accessing non-existent timelevel %d of variable %s",
                    wr.tl.index(), CCTK_FullVarName(record.var(wr.vi).index()));
      clause_t cl = wr;
      cl.valid = valid_t();
      valid_t need;
      if (isread.count(cl) > 0)
        need = isread[cl];
      const write_spec_t spec{.gi = wr.gi,
                              .tl = wr.tl,
                              .vi = wr.vi,
                              .fills = wr.valid,
                              .keeps = ~(wr.valid & ~need),
                              .event = event};
      if (CCTK_GroupTypeI(wr.gi.index()) == CCTK_GF)
        write_scopes.push_back(open_write(*active_levels, spec));
      else
        array_write_scopes.push_back(open_array_write(spec));
    }
  }

  checksums_t checksums;
  if (poison_undefined_values) {
    const std::vector<clause_t> &writes =
        decode_clauses(attribute, rdwr_t::write);
    const int numgroups = CCTK_NumGroups();
    std::vector<std::vector<std::vector<valid_t> > > gfs(numgroups);
    for (int gi = 0; gi < numgroups; ++gi) {
      const int numvars = CCTK_NumVarsInGroupI(gi);
      gfs.at(gi).resize(numvars);
      for (int vi = 0; vi < numvars; ++vi) {
        const int numtimelevels = 1; // is expanded later if necessary
        gfs.at(gi).at(vi).resize(numtimelevels);
      }
    }
    for (const auto &wr : writes) {
      if (wr.tl.index() >= int(gfs.at(wr.gi.index()).at(wr.vi.index()).size()))
        gfs.at(wr.gi.index()).at(wr.vi.index()).resize(wr.tl.index() + 1);
      gfs.at(wr.gi.index()).at(wr.vi.index()).at(wr.tl.index()) |= wr.valid;
    }

    checksums = calculate_checksums(*active_levels, gfs);
  }

  const mode_t mode = decode_mode(attribute);
  switch (mode) {
  case mode_t::local:
    loop_parallel(*active_levels, [&](GHExt::PatchData::LevelData &leveldata,
                                      int index, int component,
                                      const cGH *local_cctkGH) {
      update_cctkGH(const_cast<cGH *>(local_cctkGH), cctkGH);
      CCTK_CallFunction(function, attribute, const_cast<cGH *>(local_cctkGH));
    });
    synchronize();
    break;

  case mode_t::meta:
  case mode_t::global:
  case mode_t::level:
    // Call function just once
    // Note: meta mode scheduling must continue to work even after we
    // shut down ourselves!
    CCTK_CallFunction(function, attribute, cctkGH);
    break;

  default:
    assert(0);
  }

  if (poison_undefined_values)
    check_checksums(*active_levels, checksums,
                    [event, iteration = cctkGH->cctk_iteration]() {
                      return event + " checking output (iteration " +
                             std::to_string(iteration) + ")";
                    });

  write_scopes.clear();
  array_write_scopes.clear();

  {
    const std::vector<clause_t> &invalids =
        decode_clauses(attribute, rdwr_t::invalid);
    for (const auto &inv : invalids) {
      const invalidate_spec_t spec{.gi = inv.gi,
                                   .tl = inv.tl,
                                   .vi = inv.vi,
                                   .where = inv.valid,
                                   .event = event};
      if (CCTK_GroupTypeI(inv.gi.index()) == CCTK_GF)
        invalidate(*active_levels, spec);
      else
        invalidate_array(spec);
    }
  }

  // The flesh fires this routine's SYNC clause after we return. A nested
  // CallScheduleGroup may have overwritten last_called_function, so that a
  // sync check would name the wrong routine: set it again.
  if (ghext)
    ghext->last_called_function = attribute;

  constexpr int didsync = 0;
  return didsync;
}

namespace {
int GroupStorageCrease(const cGH *cctkGH, int n_groups, const int *groups,
                       const int *requested_tls, int *status, const bool inc) {
  DECLARE_CCTK_PARAMETERS;

  assert(cctkGH);
  assert(n_groups >= 0);
  assert(groups);
  assert(requested_tls);
  assert(ghext);
  assert(int(ghext->active_timelevels.size()) == CCTK_NumGroups());

  for (int n = 0; n < n_groups; ++n) {
    if (groups[n] < 0 or groups[n] >= CCTK_NumGroups()) {
      CCTK_VWARN(CCTK_WARN_ALERT, "Group index %d is illegal", groups[n]);
      return -1;
    }
    assert(requested_tls[n] >= 0 or requested_tls[n] == -1);
  }

  int min_num_timelevels = INT_MAX;
  for (int n = 0; n < n_groups; ++n) {
    const gi_t gi{groups[n]};
    const int gid = gi.index();
    const int declared_tls = CCTK_DeclaredTimeLevelsGI(gid);
    int ntls = requested_tls[n];
    if (ntls == -1)
      ntls = declared_tls;
    if (inc and declared_tls < 2 and ntls > declared_tls) {
      CCTK_VWARN(CCTK_WARN_ALERT,
                 "Attempting to activate %d timelevels for group '%s' which "
                 "only has a single timelevel declared in interface.ccl. "
                 "Please declare at least 2 timelevels in interface.ccl to "
                 "allow more timelevels to be created at runtime.",
                 ntls, CCTK_FullGroupName(gid));
      ntls = declared_tls;
    }
    if (ntls > declared_tls)
      ntls = declared_tls;

    const int previous = ghext->active_timelevels.at(gid);
    if (status)
      status[n] = previous;

    // Legacy flesh convention: CCTK_ActiveTimeLevels* routes through
    // CCTK_GroupStorageIncrease with requested_tls = 0 to query the current
    // number of active timelevels. Treat this as a pure read.
    if (inc and requested_tls[n] == 0) {
      min_num_timelevels = std::min(min_num_timelevels, previous);
      continue;
    }

    if (!ghext->storage_frozen) {
      ghext->active_timelevels.at(gid) = ntls;
    } else if (ntls != previous) {
      CCTK_VWARN(CCTK_WARN_ALERT,
                 "Attempting to %s storage for group '%s' from %d to %d "
                 "timelevels after the STORAGE registry has been frozen; "
                 "ignoring",
                 inc ? "increase" : "decrease", CCTK_FullGroupName(gid),
                 previous, ntls);
    }

    const int current = ghext->active_timelevels.at(gid);
    min_num_timelevels = std::min(min_num_timelevels, current);
  }
  if (min_num_timelevels == INT_MAX)
    min_num_timelevels = 0;

  return min_num_timelevels;
}
} // namespace

int GroupStorageIncrease(const cGH *cctkGH, int n_groups, const int *groups,
                         const int *tls, int *status) {
  DECLARE_CCTK_PARAMETERS;

  return GroupStorageCrease(cctkGH, n_groups, groups, tls, status, true);
}

int GroupStorageDecrease(const cGH *cctkGH, int n_groups, const int *groups,
                         const int *tls, int *status) {
  DECLARE_CCTK_PARAMETERS;

  return GroupStorageCrease(cctkGH, n_groups, groups, tls, status, false);
}

int QueryGroupStorageB(const cGH *cctkGH, int gi, const char *groupname) {
  assert(cctkGH);
  assert(ghext);
  if (gi < 0) {
    if (!groupname) {
      CCTK_WARN(CCTK_WARN_ALERT,
                "QueryGroupStorageB called with negative group index and "
                "no group name");
      return -1;
    }
    gi = CCTK_GroupIndex(groupname);
    if (gi < 0) {
      CCTK_VWARN(CCTK_WARN_ALERT, "Group '%s' not found", groupname);
      return -1;
    }
  }
  assert(gi < CCTK_NumGroups());
  return ghext->record(gi_t{gi}).has_storage() ? 1 : 0;
}

int EnableGroupStorage(const cGH *cctkGH, const char *groupname) {
  const int group = CCTK_GroupIndex(groupname);
  assert(group >= 0 and group < CCTK_NumGroups());
  // TODO: decide whether to use CCTK_MaxActiveTimeLevelsGI
  const int tls = CCTK_DeclaredTimeLevelsGI(group);
  int status;
  GroupStorageIncrease(cctkGH, 1, &group, &tls, &status);
  // Return whether storage was allocated previously
  return status;
}

int DisableGroupStorage(const cGH *cctkGH, const char *groupname) {
  const int group = CCTK_GroupIndex(groupname);
  assert(group >= 0 and group < CCTK_NumGroups());
  const int tls = 0;
  int status;
  GroupStorageDecrease(cctkGH, 1, &group, &tls, &status);
  // Return whether storage was allocated previously
  return status;
}

} // namespace CarpetX
