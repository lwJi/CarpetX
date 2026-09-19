#ifndef CARPETX_CARPETX_FILLPATCH_HXX
#define CARPETX_CARPETX_FILLPATCH_HXX

#include "driver.hxx"
#include "task_manager.hxx"

#include <functional>

namespace CarpetX {

// Allocate a temporary MultiFab on the fill path. Every temporary in
// fillpatch.cxx / subcycling.cxx that is created per fill goes through here,
// so that the subcycling counter report sees it (one `charge_temp_buffer()`
// per call; free when the report is off).
amrex::MultiFab
make_temp_mfab(const amrex::BoxArray &ba, const amrex::DistributionMapping &dm,
               int ncomps, int nghosts,
               const amrex::FabFactory<amrex::FArrayBox> &factory =
                   amrex::DefaultFabFactory<amrex::FArrayBox>());

// Sync
void FillPatch_Sync(task_manager &tasks2,
                    const GHExt::PatchData::LevelData::GroupData &groupdata,
                    amrex::MultiFab &mfab, const amrex::Geometry &geom);

// Prolongate ghosts from coarse level, optionally with same-level sync.
// When do_sync=true, also performs FillBoundary (same-level ghost exchange).
// When do_sync=false, only performs coarse-to-fine interpolation.
//
// Optional time-blend: when cmfab_old != nullptr and w_new != 1, the coarse
// patch is filled as w_new*cmfab + (1-w_new)*cmfab_old before interpolation.
// This is used to time-interpolate the coarse source when the fine level is
// mid-subcycle (misaligned in time with the coarse level).
void FillPatch_Prolongate(
    task_manager &tasks2, task_manager &tasks3,
    const GHExt::PatchData::LevelData::GroupData &groupdata,
    const GHExt::PatchData::LevelData::GroupData &coarsegroupdata,
    amrex::MultiFab &mfab, const amrex::MultiFab &cmfab,
    const amrex::Geometry &fgeom, const amrex::Geometry &cgeom,
    amrex::Interpolater *mapper, const amrex::Vector<amrex::BCRec> &bcrecs,
    bool do_sync, const amrex::MultiFab *cmfab_old = nullptr,
    CCTK_REAL w_new = 1);

// Prolongate and sync ghosts (same-level exchange + coarse-to-fine
// interpolation)
inline void FillPatch_ProlongateGhosts(
    task_manager &tasks2, task_manager &tasks3,
    const GHExt::PatchData::LevelData::GroupData &groupdata,
    const GHExt::PatchData::LevelData::GroupData &coarsegroupdata,
    amrex::MultiFab &mfab, const amrex::MultiFab &cmfab,
    const amrex::Geometry &fgeom, const amrex::Geometry &cgeom,
    amrex::Interpolater *mapper, const amrex::Vector<amrex::BCRec> &bcrecs,
    const amrex::MultiFab *cmfab_old = nullptr, CCTK_REAL w_new = 1) {
  FillPatch_Prolongate(tasks2, tasks3, groupdata, coarsegroupdata, mfab, cmfab,
                       fgeom, cgeom, mapper, bcrecs, /*do_sync=*/true,
                       cmfab_old, w_new);
}

// Prolongate only (coarse-to-fine interpolation, no same-level exchange)
inline void FillPatch_ProlongateOnly(
    task_manager &tasks2, task_manager &tasks3,
    const GHExt::PatchData::LevelData::GroupData &groupdata,
    const GHExt::PatchData::LevelData::GroupData &coarsegroupdata,
    amrex::MultiFab &mfab, const amrex::MultiFab &cmfab,
    const amrex::Geometry &fgeom, const amrex::Geometry &cgeom,
    amrex::Interpolater *mapper, const amrex::Vector<amrex::BCRec> &bcrecs) {
  FillPatch_Prolongate(tasks2, tasks3, groupdata, coarsegroupdata, mfab, cmfab,
                       fgeom, cgeom, mapper, bcrecs, /*do_sync=*/false);
}

// The back half of every coarse-to-fine fill, shared by `FillPatch_Prolongate`
// (temporary buffers) and the subcycling RK fill `FillRKBoundary` (persistent
// buffers). The caller owns both buffers and has already filled `crse_patch`
// (zero ghosts, FPinfo::ba_crse_patch on FPinfo::dm_patch) with the coarse
// state; `fine_patch` (zero ghosts, FPinfo::ba_fine_patch on the same
// distribution) is overwritten. Both must stay alive until `Prolongate_Finish`
// has returned.
//
// Coarse boundary conditions on `crse_patch`, spatial interpolation into
// `fine_patch` with `mapper`, then the start of the copy into the ghosts of
// `mfab`.
//
// `streams` is where the boundary-condition kernels of both functions go (see
// `bc_streams_t`); there is deliberately no default. With `round_robin` the
// caller must have waited for all streams before each of the two calls, and
// must wait again before the result is used. With `default_stream` the caller
// orders its kernels by issue order on the default stream and needs one wait,
// after `Prolongate_Finish`; the rules for that are written down at
// `FillRKBoundary`.
void Prolongate_Start(
    const GHExt::PatchData::LevelData::GroupData &groupdata,
    const GHExt::PatchData::LevelData::GroupData &coarsegroupdata,
    amrex::MultiFab &mfab, amrex::MultiFab &crse_patch,
    amrex::MultiFab &fine_patch, const amrex::Geometry &fgeom,
    const amrex::Geometry &cgeom, amrex::Interpolater *mapper,
    const amrex::Vector<amrex::BCRec> &bcrecs, bc_streams_t streams);

// Finish the copy into the ghosts of `mfab`, then apply the fine boundary
// conditions (after the prolongation, because symmetry boundary conditions
// might require prolongated points).
void Prolongate_Finish(const GHExt::PatchData::LevelData::GroupData &groupdata,
                       amrex::MultiFab &mfab, bc_streams_t streams);

#warning "TODO: Restrict"

// Prolongate and sync interior. Expects coarse mfab prolongated and
// synced. ("InterpFromCoarseLevel")
void FillPatch_NewLevel(
    const GHExt::PatchData::LevelData::GroupData &groupdata,
    const GHExt::PatchData::LevelData::GroupData &coarsegroupdata,
    amrex::MultiFab &mfab, const amrex::MultiFab &cmfab,
    const amrex::Geometry &cgeom, const amrex::Geometry &fgeom,
    amrex::Interpolater *mapper, const amrex::Vector<amrex::BCRec> &bcrecs);

// ("FillPatchTwoLevels")
void FillPatch_RemakeLevel(
    const GHExt::PatchData::LevelData::GroupData &groupdata,
    const GHExt::PatchData::LevelData::GroupData &coarsegroupdata,
    amrex::MultiFab &mfab, const amrex::MultiFab &cmfab,
    const amrex::MultiFab &fmfab, const amrex::Geometry &cgeom,
    const amrex::Geometry &fgeom, amrex::Interpolater *mapper,
    const amrex::Vector<amrex::BCRec> &bcrecs);

} // namespace CarpetX

#endif // #ifndef CARPETX_CARPETX_FILLPATCH_HXX
