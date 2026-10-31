#include "hierarchy/driver.hxx"
#include "hierarchy/loops.hxx"
#include "schedule/schedule.hxx"
#include "ops/interp.hxx"
#include "ops/linsolve.hxx"
#include <carpetx.hxx>

#include <cctk.h>
#include <cctk_Parameters.h>
#include <util_ErrorCodes.h>
#include <util_Table.h>

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

#include <array>
#include <atomic>
#include <cassert>
#include <optional>
#include <vector>

namespace CarpetX {

using Arith::vect;
using Loop::GridDescBase;

extern "C" CCTK_INT
CarpetX_GetDomainSpecification(const CCTK_INT patch, const CCTK_INT size,
                               CCTK_REAL *restrict const physical_min,
                               CCTK_REAL *restrict const physical_max,
                               CCTK_REAL *restrict const interior_min,
                               CCTK_REAL *restrict const interior_max,
                               CCTK_REAL *restrict const exterior_min,
                               CCTK_REAL *restrict const exterior_max,
                               CCTK_REAL *restrict const spacing) {
  assert(size == dim);
  const auto &patchdata = driver().patch(patch_t{patch});
  const auto &fab = *patchdata.level(level_t{patchdata.patch, 0}).fab;
  const auto &geom = patchdata.amrcore->Geom(0);
  for (int d = 0; d < dim; ++d) {
    physical_min[d] = geom.ProbDomain().lo(d);
    physical_max[d] = geom.ProbDomain().hi(d);
    interior_min[d] =
        geom.ProbDomain().lo(d) + CCTK_REAL(0.5) * geom.CellSize(d);
    interior_max[d] =
        geom.ProbDomain().hi(d) - CCTK_REAL(0.5) * geom.CellSize(d);
    exterior_min[d] = geom.ProbDomain().lo(d) -
                      (fab.nGrow(d) - CCTK_REAL(0.5)) * geom.CellSize(d);
    exterior_max[d] = geom.ProbDomain().hi(d) +
                      (fab.nGrow(d) - CCTK_REAL(0.5)) * geom.CellSize(d);
    spacing[d] = geom.CellSize(d);
  }
  return 0;
}

extern "C" CCTK_INT CarpetX_GetBoundarySizesAndTypes(
    const void *const cctkGH_, const CCTK_INT patch, const CCTK_INT size,
    CCTK_INT *restrict const bndsize, CCTK_INT *restrict const is_ghostbnd,
    CCTK_INT *restrict const is_symbnd, CCTK_INT *restrict const is_physbnd) {
  DECLARE_CCTK_PARAMETERS;

  const cGH *restrict const cctkGH = static_cast<const cGH *>(cctkGH_);
  assert(size == 2 * dim);

  const std::array<std::array<bool, 3>, 2> is_periodic{{
      {{bool(periodic_x), bool(periodic_y), bool(periodic_z)}},
      {{bool(periodic_x), bool(periodic_y), bool(periodic_z)}},
  }};
  const std::array<std::array<bool, 3>, 2> is_reflection{{
      {{bool(reflection_x), bool(reflection_y), bool(reflection_z)}},
      {{bool(reflection_upper_x), bool(reflection_upper_y),
        bool(reflection_upper_z)}},
  }};
  const std::array<std::array<bool, 3>, 2> is_boundary{{
      {{
          bool(!CCTK_EQUALS(boundary_x, "none")),
          bool(!CCTK_EQUALS(boundary_y, "none")),
          bool(!CCTK_EQUALS(boundary_z, "none")),
      }},
      {{
          bool(!CCTK_EQUALS(boundary_upper_x, "none")),
          bool(!CCTK_EQUALS(boundary_upper_y, "none")),
          bool(!CCTK_EQUALS(boundary_upper_z, "none")),
      }},
  }};

  for (int d = 0; d < dim; ++d) {
    for (int f = 0; f < 2; ++f) {
      bndsize[2 * d + f] = cctkGH->cctk_nghostzones[d];
      is_ghostbnd[2 * d + f] = cctkGH->cctk_bbox[2 * d + f];
      is_symbnd[2 * d + f] =
          !is_ghostbnd[2 * d + f] &&
          (is_periodic[f][d] || is_reflection[f][d] || is_boundary[f][d]);
      is_physbnd[2 * d + f] = !is_ghostbnd[2 * d + f] && !is_symbnd[2 * d + f];
    }
  }
  return 0;
}

int CallFunction_count = -1;
extern "C" CCTK_INT CarpetX_GetCallFunctionCount() {
#ifndef AMREX_USE_GPU
  return omp_get_thread_num();
#else
  assert(CallFunction_count >= 0);
  return CallFunction_count;
#endif
}

extern "C" void CarpetX_GetLoopBoxAll(const void *restrict const cctkGH_,
                                      const CCTK_INT size,
                                      CCTK_INT *restrict const loop_min,
                                      CCTK_INT *restrict const loop_max) {
  const cGH *restrict const cctkGH = static_cast<const cGH *>(cctkGH_);
  assert(size == dim);
  const GridDescBase grid(cctkGH);
  vect<int, dim> imin, imax;
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
  grid.box_int<1, 1, 1>(grid.nghostzones, imin, imax);
  for (int d = 0; d < dim; ++d)
    loop_min[d] = imin[d];
  for (int d = 0; d < dim; ++d)
    loop_max[d] = imax[d];
}

extern "C" void CarpetX_CallScheduleGroup(void *cctkGH_,
                                          const char *groupname) {
  cGH *cctkGH = static_cast<cGH *>(cctkGH_);
  static Timer timer("CallScheduleGroup");
  Interval interval(timer);
  int ierr = CCTK_ScheduleTraverse(groupname, cctkGH, CallFunction);
  assert(!ierr);
}

extern "C" void CarpetX_SolvePoisson(const CCTK_INT gi_sol,
                                     const CCTK_INT gi_rhs,
                                     const CCTK_INT gi_res,
                                     const CCTK_REAL reltol,
                                     const CCTK_REAL abstol,
                                     CCTK_REAL *restrict const res_initial,
                                     CCTK_REAL *restrict const res_final) {
  solve_poisson(gi_t{gi_sol}, gi_t{gi_rhs},
                gi_res >= 0 ? std::optional<gi_t>(gi_t{gi_res}) : std::nullopt,
                reltol, abstol, res_initial, res_final);
}

extern "C" void CarpetX_Interpolate(const CCTK_POINTER_TO_CONST cctkGH_,
                                    const CCTK_INT npoints,
                                    const CCTK_REAL *restrict const globalsx,
                                    const CCTK_REAL *restrict const globalsy,
                                    const CCTK_REAL *restrict const globalsz,
                                    const CCTK_INT nvars,
                                    const CCTK_INT *restrict const varinds,
                                    const CCTK_INT *restrict const operations,
                                    const CCTK_INT allow_boundaries,
                                    const CCTK_POINTER resultptrs_) {
  const cGH *restrict const cctkGH = static_cast<const cGH *>(cctkGH_);

  const InterpolationSetup setup(cctkGH, npoints, globalsx, globalsy, globalsz);

  const int npatches = driver().num_patches();
  const vect<vect<bool, dim>, 2> uniform{
      {{bool(allow_boundaries), bool(allow_boundaries), bool(allow_boundaries)},
       {bool(allow_boundaries), bool(allow_boundaries),
        bool(allow_boundaries)}}};
  const std::vector<vect<vect<bool, dim>, 2> > allowed_boundaries(npatches,
                                                                  uniform);
  setup.Interpolate(cctkGH, nvars, varinds, operations, allowed_boundaries,
                    resultptrs_);
}

extern "C" CCTK_INT CarpetX_GetEpoch(void) {
  return carpetx_epoch.load(std::memory_order_relaxed);
}

extern "C" CCTK_INT CarpetX_DriverInterpolate(
    CCTK_POINTER_TO_CONST const cctkGH, CCTK_INT const N_dims,
    CCTK_INT const local_interp_handle, CCTK_INT const param_table_handle,
    CCTK_INT const coord_system_handle, CCTK_INT const N_interp_points,
    CCTK_INT const interp_coords_type_code,
    CCTK_POINTER_TO_CONST const coords[], CCTK_INT const N_input_arrays,
    CCTK_INT const input_array_variable_indices[],
    CCTK_INT const N_output_arrays, CCTK_INT const output_array_type_codes[],
    CCTK_POINTER const output_arrays[]) {
  DECLARE_CCTK_PARAMETERS;

  const int carpetx_interp_handle = CCTK_InterpHandle("CarpetX");
  assert(carpetx_interp_handle >= 0);
  if (carpetx_interp_handle != local_interp_handle) {
    CCTK_VERROR("Incorrect local interpolator handle provided, only 'CarpetX' "
                "is allowed: %d != %d",
                local_interp_handle, carpetx_interp_handle);
  }

  CCTK_INT order;
  int n_elems = Util_TableGetInt(param_table_handle, &order, "order");
  assert(n_elems == 1);
  assert(order == interpolation_order);

  std::vector<CCTK_INT> varinds;
  varinds.resize(N_output_arrays);
  n_elems = Util_TableGetIntArray(param_table_handle, N_output_arrays,
                                  varinds.data(), "operand_indices");
  if (n_elems == UTIL_ERROR_TABLE_NO_SUCH_KEY) {
    assert(N_input_arrays == N_output_arrays);
    for (int i = 0; i < N_input_arrays; i++) {
      varinds.at(i) = input_array_variable_indices[i];
    }
  } else if (n_elems == N_output_arrays) {
    for (int i = 0; i < n_elems; i++) {
      varinds.at(i) = input_array_variable_indices[varinds.at(i)];
    }
  } else {
    CCTK_VERROR("TableGetIntArray failed with error code %d", n_elems);
  }

  std::vector<CCTK_INT> operations;
  operations.resize(N_output_arrays, 0);
  n_elems = Util_TableGetIntArray(param_table_handle, N_output_arrays,
                                  operations.data(), "operation_codes");
  if (n_elems == UTIL_ERROR_TABLE_NO_SUCH_KEY) {
    assert(N_input_arrays == N_output_arrays);
  } else if (n_elems != N_output_arrays) {
    CCTK_ERROR("TableGetIntArray failed.");
  }

  const CCTK_POINTER resultptrs = (CCTK_POINTER)output_arrays;
  const bool allow_boundaries = true;
  CarpetX_Interpolate(
      cctkGH, N_interp_points, static_cast<const CCTK_REAL *>(coords[0]),
      static_cast<const CCTK_REAL *>(coords[1]),
      static_cast<const CCTK_REAL *>(coords[2]), N_output_arrays,
      varinds.data(), operations.data(), allow_boundaries, resultptrs);

  return 0;
}

} // namespace CarpetX
