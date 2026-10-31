#include "hierarchy/driver.hxx"
#include "hierarchy/operations.hxx"
#include "schedule/schedule.hxx"
#include "comm/comm.hxx"
#include "ops/interp.hxx"
#include "io/io.hxx"
#include "util/logo.hxx"
#include <carpetx.hxx>

#include <cctk.h>
#include <cctk_Parameters.h>

#include <AMReX.H>
#include <AMReX_ParmParse.H>

#include <mpi.h>

#include <cassert>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace CarpetX {

namespace {
int ghext_handle = -1;

amrex::AMReX *restrict pamrex = nullptr;
} // namespace

void *SetupGH(tFleshConfig *fc, int convLevel, cGH *cctkGH);
int InitGH(cGH *cctkGH);
int ScheduleTraverseGH(cGH *cctkGH, const char *where);

int MyProc(const cGH *cctkGH);
int nProcs(const cGH *cctkGH);
int Exit(cGH *cctkGH, int retval);
int Abort(cGH *cctkGH, int retval);
int Barrier(const cGH *cctkGHa);
const int *ArrayGroupSizeB(const cGH *cctkGH, int dir, int gi,
                           const char *groupname);
int GroupDynamicData(const cGH *cctkGH, int gi, cGroupDynamicData *data);

extern "C" CCTK_INT CarpetX_InterpGridArrays(
    cGH const *const cctkGH, int const N_dims, int const local_interp_handle,
    int const param_table_handle, int const coord_system_handle,
    int const N_interp_points, int const interp_coords_type_code,
    void const *const coords[], int const N_input_arrays,
    CCTK_INT const input_array_variable_indices[], int const N_output_arrays,
    CCTK_INT const output_array_type_codes[], void *const output_arrays[]);

extern "C" int CarpetX_Startup() {
  DECLARE_CCTK_PARAMETERS;

  if (verbose)
#pragma omp critical
    CCTK_VINFO("Startup");

  const std::vector<std::string> features{
#ifdef AMREX_USE_MPI
      "MPI",
#else
      "no MPI",
#endif
#ifdef AMREX_USE_OMP
      "OpenMP",
#else
      "no OpenMP",
#endif
#ifdef AMREX_USE_GPU
      "accelerators",
#else
      "no accelerators",
#endif
#ifdef AMREX_USE_ASSERTION
      "debug",
#else
      "optimized",
#endif
  };
  std::ostringstream buf;
  buf << logo();
  buf << "   AMR driver using AMReX " << amrex::Version() << "\n   ";
  auto sep = "";
  for (const auto &feature : features) {
    buf << sep << feature;
    sep = ", ";
  }
  int ierr = CCTK_RegisterBanner(buf.str().c_str());
  assert(!ierr);

  ghext_handle = CCTK_RegisterGHExtension("CarpetX");
  assert(ghext_handle >= 0);
  int iret = CCTK_RegisterGHExtensionSetupGH(ghext_handle, SetupGH);
  assert(iret);
  iret = CCTK_RegisterGHExtensionInitGH(ghext_handle, InitGH);
  assert(iret);
  iret = CCTK_RegisterGHExtensionScheduleTraverseGH(ghext_handle,
                                                    ScheduleTraverseGH);
  assert(iret);

  CCTK_OverloadInitialise(Initialise);
  CCTK_OverloadEvolve(Evolve);
  CCTK_OverloadShutdown(Shutdown);
  CCTK_OverloadOutputGH(OutputGH);

  CCTK_OverloadMyProc(MyProc);
  CCTK_OverloadnProcs(nProcs);
  CCTK_OverloadExit(Exit);
  CCTK_OverloadAbort(Abort);
  CCTK_OverloadBarrier(Barrier);

  CCTK_OverloadEnableGroupStorage(EnableGroupStorage);
  CCTK_OverloadDisableGroupStorage(DisableGroupStorage);
  CCTK_OverloadGroupStorageIncrease(GroupStorageIncrease);
  CCTK_OverloadGroupStorageDecrease(GroupStorageDecrease);
  CCTK_OverloadQueryGroupStorageB(QueryGroupStorageB);

  CCTK_OverloadSyncGroupsByDirI(SyncGroupsOnActiveLevels);

  CCTK_OverloadInterpGridArrays(CarpetX_InterpGridArrays);
  CCTK_OverloadArrayGroupSizeB(ArrayGroupSizeB);
  CCTK_OverloadGroupDynamicData(GroupDynamicData);

  iret = CCTK_InterpRegisterOpLocalUniform(InterpLocalUniform, "CarpetX",
                                           CCTK_THORNSTRING);
  assert(!iret && "CCTK_InterpRegisterOpLocalUniform failed");

  return 0;
}

void *SetupGH(tFleshConfig *fc, int convLevel, cGH *restrict cctkGH) {
  DECLARE_CCTK_PARAMETERS;

  if (verbose)
#pragma omp critical
    CCTK_VINFO("SetupGH");

  assert(fc);
  assert(convLevel == 0);
  assert(cctkGH);

  amrex::ParmParse pp;

  pp.add("amrex.signal_handling", 0);
  pp.add("amrex.throw_exception", 1);

#ifndef AMREX_USE_HIP
  pp.add("amrex.the_arena_is_managed", 1);
#endif

  amrex::Vector<int> max_grid_size_x_vec(20, -1), max_grid_size_y_vec(20, -1),
      max_grid_size_z_vec(20, -1);
  for (int level = 0; level < 20; ++level) {
    int size_x = max_grid_sizes_x[level];
    int size_y = max_grid_sizes_y[level];
    int size_z = max_grid_sizes_z[level];
    if (size_x == -1)
      size_x = max_grid_size_x;
    if (size_y == -1)
      size_y = max_grid_size_y;
    if (size_z == -1)
      size_z = max_grid_size_z;
    max_grid_size_x_vec.at(level) = size_x;
    max_grid_size_y_vec.at(level) = size_y;
    max_grid_size_z_vec.at(level) = size_z;
  }
  pp.addarr("amr.max_grid_size_x", max_grid_size_x_vec);
  pp.addarr("amr.max_grid_size_y", max_grid_size_y_vec);
  pp.addarr("amr.max_grid_size_z", max_grid_size_z_vec);

  pp.add("amr.refine_grid_layout", refine_grid_layout);
  pp.add("amr.blocking_factor_x", blocking_factor_x);
  pp.add("amr.blocking_factor_y", blocking_factor_y);
  pp.add("amr.blocking_factor_z", blocking_factor_z);
  pp.add("amr.grid_eff", grid_efficiency);
  if (poison_undefined_values) {
    pp.add("fab.do_initval", true);
    pp.add("fab.init_snan", true);
    amrex::FArrayBox::set_do_initval(true);
    amrex::FArrayBox::set_initval(
        std::numeric_limits<amrex::Real>::signaling_NaN());
  }
  pp.addarr(
      "fabarray.mfiter_tile_size",
      std::vector<int>{max_tile_size_x, max_tile_size_y, max_tile_size_z});

  std::vector<char *> args;
  for (std::size_t n = 0; n < 100; ++n)
    if (amrex_parameters[n][0])
      args.push_back(strdup(amrex_parameters[n]));
  int argc = args.size();
  args.push_back(nullptr);
  char **argv = args.data();
  if (verbose) {
#pragma omp critical
    {
      CCTK_VINFO("  Run-time AMReX parameters:");
      for (int n = 0; n < argc; ++n)
        CCTK_VINFO("    [%d]: %s", n, argv[n]);
    }
  }
  pamrex = amrex::Initialize(argc, argv, true, MPI_COMM_WORLD);
  for (auto arg : args)
    free(arg);
  args.clear();

  if (CCTK_IsFunctionAliased("MultiPatch_Setup")) {
    const int ierr = MultiPatch_Setup();
    assert(!ierr);
  }

  return &create_hierarchy(use_subcycling, do_reflux);
}

int InitGH(cGH *restrict cctkGH) {
  DECLARE_CCTK_PARAMETERS;

  if (verbose)
#pragma omp critical
    CCTK_VINFO("InitGH");

  assert(cctkGH);

  assert(driver().num_patches() == 0);
  CCTK_INT num_patches = 1;
  if (CCTK_IsFunctionAliased("MultiPatch_GetSystemSpecification")) {
    const int ierr = MultiPatch_GetSystemSpecification(&num_patches);
    assert(!ierr);
  }
  for (int patch = 0; patch < num_patches; ++patch)
    add_patch(patch_t{patch});

  return 0;
}

int ScheduleTraverseGH(cGH *restrict cctkGH, const char *where) {
  DECLARE_CCTK_PARAMETERS;

  if (verbose)
#pragma omp critical
    CCTK_VINFO("ScheduleTraverseGH iteration %d %s", cctkGH->cctk_iteration,
               where);

  int ierr = CCTK_ScheduleTraverse(where, cctkGH, CallFunction);
  assert(ierr == 0 || ierr == 2);

  return 0;
}

void ShutdownADIOS2();
void ShutdownOpenPMD();
extern "C" int CarpetX_Shutdown() {
  DECLARE_CCTK_PARAMETERS;

  if (verbose)
#pragma omp critical
    CCTK_VINFO("Shutdown");

  ShutdownADIOS2();
  ShutdownOpenPMD();

  if (false) {
    int iret = CCTK_UnregisterGHExtension("CarpetX");
    assert(iret == 0);
  }

  destroy_hierarchy();

  amrex::Finalize(pamrex);
  pamrex = nullptr;

  return 0;
}

int MyProc(const cGH *restrict cctkGH) {
  if (pamrex) {
    return amrex::ParallelDescriptor::MyProc();
  } else {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    return rank;
  }
}

int nProcs(const cGH *restrict cctkGH) {
  if (pamrex) {
    return amrex::ParallelDescriptor::NProcs();
  } else {
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    return size;
  }
}

int Exit(cGH *cctkGH, int retval) {
  if (pamrex) {
    amrex::ParallelDescriptor::Abort(retval);
  } else {
    MPI_Abort(MPI_COMM_WORLD, retval);
  }
  return 0;
}

int Abort(cGH *cctkGH, int retval) {
  if (pamrex) {
    amrex::ParallelDescriptor::Abort(retval);
  } else {
    MPI_Abort(MPI_COMM_WORLD, retval);
  }
  return 0;
}

int Barrier(const cGH *restrict cctkGH) {
  assert(pamrex);
  amrex::ParallelDescriptor::Barrier();
  return 0;
}

const int *ArrayGroupSizeB(const cGH *cctkGH, int dir, int gi,
                           const char *groupname) {
  if (groupname) {
    gi = CCTK_GroupIndex(groupname);
  }
  assert(gi >= 0 and gi < CCTK_NumGroups());

  cGroup group;
  int ierr = CCTK_GroupData(gi, &group);
  assert(!ierr);
  assert(dir >= 0 && dir < group.dim);
  if (group.grouptype == CCTK_GF) {
    return &cctkGH->cctk_ash[dir];
  } else if (group.grouptype == CCTK_SCALAR or group.grouptype == CCTK_ARRAY) {
    return &driver().record(gi_t{gi}).array().ash[dir];
  } else {
    CCTK_VERROR("Internal error: unexpected group type %d for group '%s'",
                (int)group.grouptype, CCTK_FullGroupName(gi));
  }
}

int GroupDynamicData(const cGH *cctkGH, int gi, cGroupDynamicData *data) {
  if (not cctkGH)
    return -3;
  if (not(gi >= 0 and gi < CCTK_NumGroups()))
    return -78;
  if (not data)
    return -1;
  cGroup group;
  int ierr = CCTK_GroupData(gi, &group);
  assert(!ierr);
  if (group.grouptype == CCTK_GF) {
    data->dim = group.dim;
    data->lsh = cctkGH->cctk_lsh;
    data->ash = cctkGH->cctk_ash;
    data->gsh = cctkGH->cctk_gsh;
    data->lbnd = cctkGH->cctk_lbnd;
    data->ubnd = cctkGH->cctk_ubnd;
    data->tile_min = cctkGH->cctk_tile_min;
    data->tile_max = cctkGH->cctk_tile_max;
    data->bbox = cctkGH->cctk_bbox;
    data->nghostzones = cctkGH->cctk_nghostzones;
    data->activetimelevels = CCTK_ActiveTimeLevelsGI(cctkGH, gi);
  } else if (group.grouptype == CCTK_SCALAR or group.grouptype == CCTK_ARRAY) {
    const auto &arraygroupdata = driver().record(gi_t{gi}).array();
    data->dim = arraygroupdata.dimension;
    data->lsh = arraygroupdata.lsh;
    data->ash = arraygroupdata.ash;
    data->gsh = arraygroupdata.gsh;
    data->lbnd = arraygroupdata.lbnd;
    data->ubnd = arraygroupdata.ubnd;
    data->bbox = arraygroupdata.bbox;
    data->nghostzones = arraygroupdata.nghostzones;
    data->activetimelevels = arraygroupdata.record.num_timelevels();
  } else {
    CCTK_VERROR("Internal error: unexpected group type %d for group '%s'",
                (int)group.grouptype, CCTK_FullGroupName(gi));
  }
  return 0;
}

extern "C" CCTK_INT CarpetX_InterpGridArrays(
    cGH const *const cctkGH, int const N_dims, int const local_interp_handle,
    int const param_table_handle, int const coord_system_handle,
    int const N_interp_points, int const interp_coords_type_code,
    void const *const coords[], int const N_input_arrays,
    CCTK_INT const input_array_variable_indices[], int const N_output_arrays,
    CCTK_INT const output_array_type_codes[], void *const output_arrays[]) {
  return CarpetX_DriverInterpolate(
      cctkGH, N_dims, local_interp_handle, param_table_handle,
      coord_system_handle, N_interp_points, interp_coords_type_code, coords,
      N_input_arrays, input_array_variable_indices, N_output_arrays,
      output_array_type_codes, output_arrays);
}

} // namespace CarpetX
