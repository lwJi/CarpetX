#include "fillpatch.hxx"
#include "schedule.hxx"

#include <utility>

#include <AMReX_FillPatchUtil.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_Version.H>

namespace CarpetX {

using namespace amrex;
#if AMREX_RELEASE_NUMBER >= 240500
using namespace amrex::detail;
#endif

void FillPatch_Sync(task_manager &tasks2,
                    const GHExt::PatchData::LevelData::GroupData &groupdata,
                    MultiFab &mfab, const Geometry &geom) {
  assert(!groupdata.mfab.empty());
  mfab.FillBoundary_nowait(0, mfab.nComp(), mfab.nGrowVect(),
                           geom.periodicity());
  tasks2.submit_serially([&groupdata, &mfab]() {
    mfab.FillBoundary_finish();
    groupdata.apply_boundary_conditions(mfab);
  });
}

void FillPatch_Prolongate(
    task_manager &tasks2, task_manager &tasks3,
    const GHExt::PatchData::LevelData::GroupData &groupdata,
    const GHExt::PatchData::LevelData::GroupData &coarsegroupdata,
    MultiFab &mfab, const MultiFab &cmfab, const Geometry &fgeom,
    const Geometry &cgeom, Interpolater *const mapper,
    const Vector<BCRec> &bcrecs, const bool do_sync,
    const MultiFab *const cmfab_old, const CCTK_REAL w_new) {
  assert(!groupdata.mfab.empty());
  assert(!coarsegroupdata.mfab.empty());
  const IntVect &nghosts = mfab.nGrowVect();
  if (nghosts.max() == 0)
    return;

  const bool do_blend = cmfab_old && w_new != CCTK_REAL(1);

  const int ncomps = mfab.nComp();
  const IntVect ratio{2, 2, 2};
  const EB2::IndexSpace *const index_space = nullptr;

  const InterpolaterBoxCoarsener &coarsener = mapper->BoxCoarsener(ratio);

  const FabArrayBase::FPinfo &fpc = FabArrayBase::TheFPinfo(
      mfab, mfab, nghosts, coarsener, fgeom, cgeom, index_space);

  if (do_sync)
    mfab.FillBoundary_nowait(0, mfab.nComp(), mfab.nGrowVect(),
                             fgeom.periodicity());

  if (fpc.ba_crse_patch.empty()) {
    if (do_sync) {
      tasks2.submit_serially([&groupdata, &mfab]() {
        mfab.FillBoundary_finish();
        groupdata.apply_boundary_conditions(mfab);
      });
    }
    return;
  }

  MultiFab *const mfab_crse_patch_ptr =
      new MultiFab(make_mf_crse_patch<MultiFab>(fpc, ncomps));
  MultiFab &mfab_crse_patch = *mfab_crse_patch_ptr;
  mf_set_domain_bndry(mfab_crse_patch, cgeom);

  mfab_crse_patch.ParallelCopy_nowait(cmfab, 0, 0, ncomps, IntVect{0},
                                      mfab_crse_patch.nGrowVect(),
                                      cgeom.periodicity());

  MultiFab *mfab_crse_patch_old_ptr = nullptr;
  if (do_blend) {
    mfab_crse_patch_old_ptr =
        new MultiFab(make_mf_crse_patch<MultiFab>(fpc, ncomps));
    mf_set_domain_bndry(*mfab_crse_patch_old_ptr, cgeom);
    mfab_crse_patch_old_ptr->ParallelCopy_nowait(
        *cmfab_old, 0, 0, ncomps, IntVect{0},
        mfab_crse_patch_old_ptr->nGrowVect(), cgeom.periodicity());
  }

  tasks2.submit_serially([&tasks3, &groupdata, &coarsegroupdata, &mfab, &cgeom,
                          &fgeom, mapper, &bcrecs, &fpc, mfab_crse_patch_ptr,
                          mfab_crse_patch_old_ptr, w_new, do_sync]() {
    const int ncomps = mfab.nComp();
    MultiFab &mfab_crse_patch = *mfab_crse_patch_ptr;

    if (do_sync)
      mfab.FillBoundary_finish();

    mfab_crse_patch.ParallelCopy_finish();

    if (mfab_crse_patch_old_ptr) {
      MultiFab &mfab_crse_patch_old = *mfab_crse_patch_old_ptr;
      mfab_crse_patch_old.ParallelCopy_finish();
      const CCTK_REAL w_old = CCTK_REAL(1) - w_new;
      MultiFab::LinComb(mfab_crse_patch, w_new, mfab_crse_patch, 0, w_old,
                        mfab_crse_patch_old, 0, 0, ncomps, IntVect{0});
      delete mfab_crse_patch_old_ptr;
    }

    MultiFab *const mfab_fine_patch_ptr =
        new MultiFab(make_mf_fine_patch<MultiFab>(fpc, ncomps));
    MultiFab &mfab_fine_patch = *mfab_fine_patch_ptr;

    Prolongate_Start(groupdata, coarsegroupdata, mfab, mfab_crse_patch,
                     mfab_fine_patch, fgeom, cgeom, mapper, bcrecs);

    delete mfab_crse_patch_ptr;

    tasks3.submit_serially([&groupdata, &mfab, mfab_fine_patch_ptr]() {
      Prolongate_Finish(groupdata, mfab);

      delete mfab_fine_patch_ptr;
    });
  });
}

void Prolongate_Start(
    const GHExt::PatchData::LevelData::GroupData &groupdata,
    const GHExt::PatchData::LevelData::GroupData &coarsegroupdata,
    MultiFab &mfab, MultiFab &crse_patch, MultiFab &fine_patch,
    const Geometry &fgeom, const Geometry &cgeom, Interpolater *const mapper,
    const Vector<BCRec> &bcrecs) {
  assert(!groupdata.mfab.empty());
  assert(!coarsegroupdata.mfab.empty());
  const IntVect &nghosts = mfab.nGrowVect();
  const int ncomps = mfab.nComp();
  const IntVect ratio{2, 2, 2};
  assert(crse_patch.nComp() == ncomps);
  assert(fine_patch.nComp() == ncomps);
  assert(crse_patch.DistributionMap() == fine_patch.DistributionMap());
  assert(crse_patch.size() == fine_patch.size());

  coarsegroupdata.apply_boundary_conditions(crse_patch);

  FillPatchInterp(fine_patch, 0, crse_patch, 0, ncomps, IntVect{0}, cgeom,
                  fgeom, grow(convert(fgeom.Domain(), mfab.ixType()), nghosts),
                  ratio, mapper, bcrecs, 0);

  mfab.ParallelCopy_nowait(fine_patch, 0, 0, ncomps, IntVect{0}, nghosts);
}

void Prolongate_Finish(const GHExt::PatchData::LevelData::GroupData &groupdata,
                       MultiFab &mfab) {
  mfab.ParallelCopy_finish();

  groupdata.apply_boundary_conditions(mfab);
}

void FillPatch_NewLevel(
    const GHExt::PatchData::LevelData::GroupData &groupdata,
    const GHExt::PatchData::LevelData::GroupData &coarsegroupdata,
    MultiFab &mfab, const MultiFab &cmfab, const Geometry &cgeom,
    const Geometry &fgeom, Interpolater *const mapper,
    const Vector<BCRec> &bcrecs) {
  assert(!groupdata.mfab.empty());
  assert(!coarsegroupdata.mfab.empty());
  const int ncomps = mfab.nComp();
  const IntVect ratio{2, 2, 2};
  const IntVect &nghosts = mfab.nGrowVect();

  const InterpolaterBoxCoarsener &coarsener = mapper->BoxCoarsener(ratio);

  const BoxArray &ba = mfab.boxArray();
  const DistributionMapping &dm = mfab.DistributionMap();

  const IndexType &ixtype = ba.ixType();
  assert(ixtype == cmfab.boxArray().ixType());

  Box fdomain_g(amrex::convert(fgeom.Domain(), mfab.ixType()));
  for (int d = 0; d < dim; ++d)
    if (fgeom.isPeriodic(d))
      fdomain_g.grow(d, nghosts[d]);

  const int nboxes = ba.size();
  BoxArray cba_g(nboxes);
  for (int i = 0; i < nboxes; ++i) {
    Box box = amrex::convert(amrex::grow(ba[i], nghosts), ixtype);
    box &= fdomain_g;
    cba_g.set(i, coarsener.doit(box));
  }
  MultiFab cmfab_g(cba_g, dm, ncomps, 0);
  mf_set_domain_bndry(cmfab_g, cgeom);

  cmfab_g.ParallelCopy(cmfab, 0, 0, ncomps, cgeom.periodicity());

  coarsegroupdata.apply_boundary_conditions(cmfab_g);

  FillPatchInterp(mfab, 0, cmfab_g, 0, ncomps, nghosts, cgeom, fgeom, fdomain_g,
                  ratio, mapper, bcrecs, 0);

  groupdata.apply_boundary_conditions(mfab);
}

void FillPatch_RemakeLevel(
    const GHExt::PatchData::LevelData::GroupData &groupdata,
    const GHExt::PatchData::LevelData::GroupData &coarsegroupdata,
    MultiFab &mfab, const MultiFab &cmfab, const MultiFab &fmfab,
    const Geometry &cgeom, const Geometry &fgeom, Interpolater *const mapper,
    const Vector<BCRec> &bcrecs) {
  assert(!groupdata.mfab.empty());
  assert(!coarsegroupdata.mfab.empty());
  const int ncomps = mfab.nComp();
  const IntVect ratio{2, 2, 2};
  const IntVect &nghosts = mfab.nGrowVect();
  const EB2::IndexSpace *const index_space = nullptr;

  const InterpolaterBoxCoarsener &coarsener = mapper->BoxCoarsener(ratio);

  const FabArrayBase::FPinfo &fpc = FabArrayBase::TheFPinfo(
      fmfab, mfab, nghosts, coarsener, fgeom, cgeom, index_space);

  if (!fpc.ba_crse_patch.empty()) {
    MultiFab mfab_crse_patch = make_mf_crse_patch<MultiFab>(fpc, ncomps);
    mf_set_domain_bndry(mfab_crse_patch, cgeom);

    mfab_crse_patch.ParallelCopy(cmfab, 0, 0, ncomps, IntVect{0},
                                 mfab_crse_patch.nGrowVect(),
                                 cgeom.periodicity());
    coarsegroupdata.apply_boundary_conditions(mfab_crse_patch);

    MultiFab mfab_fine_patch = make_mf_fine_patch<MultiFab>(fpc, ncomps);

    FillPatchInterp(mfab_fine_patch, 0, mfab_crse_patch, 0, ncomps, IntVect{0},
                    cgeom, fgeom,
                    grow(convert(fgeom.Domain(), mfab.ixType()), nghosts),
                    ratio, mapper, bcrecs, 0);

    mfab.ParallelCopy_nowait(mfab_fine_patch, 0, 0, ncomps, IntVect{0},
                             nghosts);
    mfab.ParallelCopy_finish();
  }

  mfab.ParallelCopy(fmfab, 0, 0, ncomps, IntVect{0}, nghosts,
                    fgeom.periodicity());
  groupdata.apply_boundary_conditions(mfab);
}

} // namespace CarpetX
