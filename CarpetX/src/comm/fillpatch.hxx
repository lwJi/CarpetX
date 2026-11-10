#ifndef CARPETX_CARPETX_COMM_FILLPATCH_HXX
#define CARPETX_CARPETX_COMM_FILLPATCH_HXX

#include "internal.hxx"

#include "hierarchy/driver.hxx"
#include "util/task_manager.hxx"

#include <AMReX_BCRec.H>
#include <AMReX_Interpolater.H>

#include <functional>

namespace CarpetX {

void FillPatch_Sync(task_manager &tasks2, amrex::MultiFab &mfab,
                    const amrex::Geometry &geom);

void FillPatch_Prolongate(
    task_manager &tasks2, task_manager &tasks3,
    const GHExt::PatchData::LevelData::GroupData &groupdata,
    const GHExt::PatchData::LevelData::GroupData &coarsegroupdata,
    amrex::MultiFab &mfab, const amrex::MultiFab &cmfab,
    const amrex::Geometry &fgeom, const amrex::Geometry &cgeom, bool do_sync,
    const amrex::MultiFab *cmfab_old = nullptr, CCTK_REAL w_new = 1);

inline void FillPatch_ProlongateGhosts(
    task_manager &tasks2, task_manager &tasks3,
    const GHExt::PatchData::LevelData::GroupData &groupdata,
    const GHExt::PatchData::LevelData::GroupData &coarsegroupdata,
    amrex::MultiFab &mfab, const amrex::MultiFab &cmfab,
    const amrex::Geometry &fgeom, const amrex::Geometry &cgeom,
    const amrex::MultiFab *cmfab_old = nullptr, CCTK_REAL w_new = 1) {
  FillPatch_Prolongate(tasks2, tasks3, groupdata, coarsegroupdata, mfab, cmfab,
                       fgeom, cgeom, true, cmfab_old, w_new);
}

inline void FillPatch_ProlongateOnly(
    task_manager &tasks2, task_manager &tasks3,
    const GHExt::PatchData::LevelData::GroupData &groupdata,
    const GHExt::PatchData::LevelData::GroupData &coarsegroupdata,
    amrex::MultiFab &mfab, const amrex::MultiFab &cmfab,
    const amrex::Geometry &fgeom, const amrex::Geometry &cgeom) {
  FillPatch_Prolongate(tasks2, tasks3, groupdata, coarsegroupdata, mfab, cmfab,
                       fgeom, cgeom, false);
}

// The back half of a coarse-to-fine ghost fill, shared by
// `FillPatch_Prolongate` (temporary buffers) and the subcycling RK fill
// `FillRKBoundary` (persistent buffers). The caller owns both buffers and has
// already filled `crse_patch` (zero ghosts, FPinfo::ba_crse_patch on
// FPinfo::dm_patch) with the coarse state; `fine_patch` (zero ghosts,
// FPinfo::ba_fine_patch on the same distribution) is overwritten. Both must
// stay alive until `Prolongate_Finish` has returned.
//
// Coarse boundary conditions on `crse_patch`, spatial interpolation into
// `fine_patch` with `mapper`, then the start of the copy into the ghosts of
// `mfab`.
void Prolongate_Start(
    const GHExt::PatchData::LevelData::GroupData &groupdata,
    const GHExt::PatchData::LevelData::GroupData &coarsegroupdata,
    amrex::MultiFab &mfab, amrex::MultiFab &crse_patch,
    amrex::MultiFab &fine_patch, const amrex::Geometry &fgeom,
    const amrex::Geometry &cgeom, amrex::Interpolater *mapper,
    const amrex::Vector<amrex::BCRec> &bcrecs);

void Prolongate_Finish(amrex::MultiFab &mfab);

void FillPatch_NewLevel(
    const GHExt::PatchData::LevelData::GroupData &groupdata,
    const GHExt::PatchData::LevelData::GroupData &coarsegroupdata,
    amrex::MultiFab &mfab, const amrex::MultiFab &cmfab,
    const amrex::Geometry &cgeom, const amrex::Geometry &fgeom);

void FillPatch_RemakeLevel(
    const GHExt::PatchData::LevelData::GroupData &groupdata,
    const GHExt::PatchData::LevelData::GroupData &coarsegroupdata,
    amrex::MultiFab &mfab, const amrex::MultiFab &cmfab,
    const amrex::MultiFab &fmfab, const amrex::Geometry &cgeom,
    const amrex::Geometry &fgeom);

} // namespace CarpetX

#endif
