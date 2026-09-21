#include "subcycling.hxx"

#include "driver.hxx"
#include "fillpatch.hxx"
#include "schedule.hxx"
#include "task_manager.hxx"

#include <AMReX_FabArray.H>     // MultiArray4, MultiFab::arrays()
#include <AMReX_FabArrayBase.H> // FabArrayBase::TheFPinfo
#include <AMReX_Geometry.H>
#include <AMReX_GpuContainers.H> // amrex::GpuArray
#include <AMReX_GpuDevice.H>     // amrex::Gpu::synchronize
#include <AMReX_IntVect.H>
#include <AMReX_Interpolater.H>
#include <AMReX_MFParallelFor.H> // amrex::ParallelFor(MF, IntVect, ncomp, F)
#include <AMReX_MultiFab.H>
#include <AMReX_Periodicity.H>

#include <array>
#include <cassert>
#include <memory>
#include <vector>

namespace CarpetX {

namespace {

using GroupData = GHExt::PatchData::LevelData::GroupData;

// crse_patch = u0 + dtc * P_stage(xsi; k_1..k_RKSTAGES), evaluated with one
// fused amrex::ParallelFor over the coarse source-band geometry. The
// coefficient tables are AMReX's FillPatcher::fillRK dense-output formulas
// (AMReX_FillPatcher.H: RK3 table around lines 474-527, RK4 table around
// 530-605), with the coarse-to-fine step ratio r fixed at 1/2.
//
// The kernel walks `crse_patch` and the bands box by box, so they must share
// one BoxArray and one DistributionMapping (see EnsureRKBuffers).
template <int RKSTAGES>
void rk_dense_output_impl(amrex::MultiFab &crse_patch,
                          const amrex::MultiFab &old_band,
                          const std::array<std::unique_ptr<amrex::MultiFab>,
                                           max_num_rk_stages> &ks_bands,
                          const int stage, const CCTK_REAL xsi,
                          const CCTK_REAL dtc) {
  static_assert(RKSTAGES == 3 || RKSTAGES == 4,
                "rk_dense_output only supports RKSTAGES == 3 or 4");
  assert(stage > 0 && stage <= RKSTAGES);

  // ratio between coarse and fine time step (2:1 refinement)
  constexpr CCTK_REAL r = 0.5;

  const int nvars = crse_patch.nComp();
  assert(old_band.nComp() == nvars);
  assert(old_band.boxArray() == crse_patch.boxArray());
  // The kernel below indexes all MultiFabs with one local box index
  assert(old_band.DistributionMap() == crse_patch.DistributionMap());

  const CCTK_REAL xsi2 = xsi * xsi;

  // Stage the coarse old state u(t_n); the polynomial is then accumulated in
  // place on the coarse patch.
  amrex::MultiFab::Copy(crse_patch, old_band, 0, 0, nvars, 0);

  auto yf_arrs = crse_patch.arrays();
  amrex::GpuArray<amrex::MultiArray4<const CCTK_REAL>, RKSTAGES> kcs_arrs;
  for (int s = 0; s < RKSTAGES; ++s) {
    assert(ks_bands[s]);
    assert(ks_bands[s]->nComp() == nvars);
    assert(ks_bands[s]->boxArray() == crse_patch.boxArray());
    assert(ks_bands[s]->DistributionMap() == crse_patch.DistributionMap());
    kcs_arrs[s] = ks_bands[s]->const_arrays();
  }

  if constexpr (RKSTAGES == 3) {
    // AMReX RK3 (SSPRK3) dense-output coefficients, a degree-2 Taylor
    // expansion over S_old + k1, k2, k3 (no uttt term).
    const std::array<CCTK_REAL, 3> b = {
        xsi - (5. / 6.) * xsi2, // b1
        (1. / 6.) * xsi2,       // b2
        (2. / 3.) * xsi2        // b3
    };
    const std::array<CCTK_REAL, 3> bt = {
        1.0 - (5. / 3.) * xsi, // bt1
        (1. / 3.) * xsi,       // bt2
        (4. / 3.) * xsi        // bt3
    };
    constexpr std::array<CCTK_REAL, 3> btt = {
        -5. / 3., // btt1
        1. / 3.,  // btt2
        4. / 3.   // btt3
    };

    if (stage == 1) {
      amrex::ParallelFor(
          crse_patch, amrex::IntVect{0}, nvars,
          [=] AMREX_GPU_DEVICE(int b_, int i, int j, int k, int n) noexcept {
            const std::array<CCTK_REAL, 3> kk = {kcs_arrs[0][b_](i, j, k, n),
                                                 kcs_arrs[1][b_](i, j, k, n),
                                                 kcs_arrs[2][b_](i, j, k, n)};
            const CCTK_REAL uu = b[0] * kk[0] + b[1] * kk[1] + b[2] * kk[2];
            yf_arrs[b_](i, j, k, n) += dtc * uu;
          });
    } else if (stage == 2) {
      amrex::ParallelFor(
          crse_patch, amrex::IntVect{0}, nvars,
          [=] AMREX_GPU_DEVICE(int b_, int i, int j, int k, int n) noexcept {
            const std::array<CCTK_REAL, 3> kk = {kcs_arrs[0][b_](i, j, k, n),
                                                 kcs_arrs[1][b_](i, j, k, n),
                                                 kcs_arrs[2][b_](i, j, k, n)};
            const CCTK_REAL uu = b[0] * kk[0] + b[1] * kk[1] + b[2] * kk[2];
            const CCTK_REAL ut = bt[0] * kk[0] + bt[1] * kk[1] + bt[2] * kk[2];
            // note r*ut (not 0.5*r*ut as in the RK4 stage 2)
            yf_arrs[b_](i, j, k, n) += dtc * (uu + r * ut);
          });
    } else { // stage 3
      const CCTK_REAL r2 = r * r;
      amrex::ParallelFor(
          crse_patch, amrex::IntVect{0}, nvars,
          [=] AMREX_GPU_DEVICE(int b_, int i, int j, int k, int n) noexcept {
            const std::array<CCTK_REAL, 3> kk = {kcs_arrs[0][b_](i, j, k, n),
                                                 kcs_arrs[1][b_](i, j, k, n),
                                                 kcs_arrs[2][b_](i, j, k, n)};
            const CCTK_REAL uu = b[0] * kk[0] + b[1] * kk[1] + b[2] * kk[2];
            const CCTK_REAL ut = bt[0] * kk[0] + bt[1] * kk[1] + bt[2] * kk[2];
            const CCTK_REAL utt =
                btt[0] * kk[0] + btt[1] * kk[1] + btt[2] * kk[2];
            yf_arrs[b_](i, j, k, n) +=
                dtc * (uu + 0.5 * r * ut + 0.25 * r2 * utt);
          });
    }
  } else { // RKSTAGES == 4
    const CCTK_REAL xsi3 = xsi2 * xsi;

    // Coefficients for the dense output formulas (U, Ut, Utt, Uttt)
    const std::array<CCTK_REAL, 4> b = {
        xsi - 1.5 * xsi2 + (2. / 3.) * xsi3, // b1
        xsi2 - (2. / 3.) * xsi3,             // b2
        xsi2 - (2. / 3.) * xsi3,             // b3
        -0.5 * xsi2 + (2. / 3.) * xsi3       // b4
    };
    const std::array<CCTK_REAL, 4> bt = {
        1.0 - 3.0 * xsi + 2.0 * xsi2, // bt1
        2.0 * xsi - 2.0 * xsi2,       // bt2
        2.0 * xsi - 2.0 * xsi2,       // bt3
        -xsi + 2.0 * xsi2             // bt4
    };
    const std::array<CCTK_REAL, 4> btt = {
        -3.0 + 4.0 * xsi, // btt1
        2.0 - 4.0 * xsi,  // btt2
        2.0 - 4.0 * xsi,  // btt3
        -1.0 + 4.0 * xsi  // btt4
    };
    constexpr std::array<CCTK_REAL, 4> bttt = {
        4.0,  // bttt1
        -4.0, // bttt2
        -4.0, // bttt3
        4.0   // bttt4
    };

    if (stage == 1) {
      amrex::ParallelFor(
          crse_patch, amrex::IntVect{0}, nvars,
          [=] AMREX_GPU_DEVICE(int b_, int i, int j, int k, int n) noexcept {
            const std::array<CCTK_REAL, 4> kk = {
                kcs_arrs[0][b_](i, j, k, n), kcs_arrs[1][b_](i, j, k, n),
                kcs_arrs[2][b_](i, j, k, n), kcs_arrs[3][b_](i, j, k, n)};
            const CCTK_REAL uu =
                b[0] * kk[0] + b[1] * kk[1] + b[2] * kk[2] + b[3] * kk[3];
            yf_arrs[b_](i, j, k, n) += dtc * uu;
          });
    } else if (stage == 2) {
      amrex::ParallelFor(
          crse_patch, amrex::IntVect{0}, nvars,
          [=] AMREX_GPU_DEVICE(int b_, int i, int j, int k, int n) noexcept {
            const std::array<CCTK_REAL, 4> kk = {
                kcs_arrs[0][b_](i, j, k, n), kcs_arrs[1][b_](i, j, k, n),
                kcs_arrs[2][b_](i, j, k, n), kcs_arrs[3][b_](i, j, k, n)};
            const CCTK_REAL uu =
                b[0] * kk[0] + b[1] * kk[1] + b[2] * kk[2] + b[3] * kk[3];
            const CCTK_REAL ut =
                bt[0] * kk[0] + bt[1] * kk[1] + bt[2] * kk[2] + bt[3] * kk[3];
            yf_arrs[b_](i, j, k, n) += dtc * (uu + 0.5 * r * ut);
          });
    } else { // stage 3 or stage 4
      const CCTK_REAL r2 = r * r;
      const CCTK_REAL r3 = r2 * r;
      const CCTK_REAL at = (stage == 3) ? 0.5 * r : r;
      const CCTK_REAL att = (stage == 3) ? 0.25 * r2 : 0.5 * r2;
      const CCTK_REAL attt = (stage == 3) ? 0.0625 * r3 : 0.125 * r3;
      const CCTK_REAL ak = (stage == 3) ? -4.0 : 4.0;
      amrex::ParallelFor(
          crse_patch, amrex::IntVect{0}, nvars,
          [=] AMREX_GPU_DEVICE(int b_, int i, int j, int k, int n) noexcept {
            const std::array<CCTK_REAL, 4> kk = {
                kcs_arrs[0][b_](i, j, k, n), kcs_arrs[1][b_](i, j, k, n),
                kcs_arrs[2][b_](i, j, k, n), kcs_arrs[3][b_](i, j, k, n)};
            const CCTK_REAL uu =
                b[0] * kk[0] + b[1] * kk[1] + b[2] * kk[2] + b[3] * kk[3];
            const CCTK_REAL ut =
                bt[0] * kk[0] + bt[1] * kk[1] + bt[2] * kk[2] + bt[3] * kk[3];
            const CCTK_REAL utt = btt[0] * kk[0] + btt[1] * kk[1] +
                                  btt[2] * kk[2] + btt[3] * kk[3];
            const CCTK_REAL uttt = bttt[0] * kk[0] + bttt[1] * kk[1] +
                                   bttt[2] * kk[2] + bttt[3] * kk[3];
            yf_arrs[b_](i, j, k, n) +=
                dtc * (uu + at * ut + att * utt +
                       attt * (uttt + ak * (kk[2] - kk[1])));
          });
    }
  }

  // Wait for the device kernel before the result is consumed.
  amrex::Gpu::synchronize();
}

// crse_patch = u0 + dtc * P_stage(xsi; k_1..k_N), num_rk_stages in {3, 4}
void rk_dense_output(amrex::MultiFab &crse_patch,
                     const amrex::MultiFab &old_band,
                     const std::array<std::unique_ptr<amrex::MultiFab>,
                                      max_num_rk_stages> &ks_bands,
                     const int num_rk_stages, const int stage,
                     const CCTK_REAL xsi, const CCTK_REAL dtc) {
  switch (num_rk_stages) {
  case 3:
    rk_dense_output_impl<3>(crse_patch, old_band, ks_bands, stage, xsi, dtc);
    break;
  case 4:
    rk_dense_output_impl<4>(crse_patch, old_band, ks_bands, stage, xsi, dtc);
    break;
  default:
    CCTK_VERROR("Subcycling dense output supports 3 (SSPRK3) or 4 (RK4) RK "
                "stages, but ghext->num_rk_stages is %d",
                num_rk_stages);
  }
}

// Interior-only copy of `src` into a zero-ghost source band. Source-band boxes
// may lie outside the domain in periodic directions (they are coarsenings of
// the children's periodically grown cf-ghost footprint), so the copy must be
// allowed to wrap around, exactly as FillPatch_Prolongate's coarse-patch copy
// does.
void copy_interior_to_band(amrex::MultiFab &band, const amrex::MultiFab &src,
                           const amrex::Periodicity &period) {
  assert(band.ixType() == src.ixType());
  assert(band.nComp() == src.nComp());
  band.ParallelCopy(src, 0, 0, band.nComp(), amrex::IntVect{0},
                    amrex::IntVect{0}, period);
}

// The FPinfo of the RK boundary fill into `finemfab`: the same lookup as in
// FillPatch_Prolongate. The refinement ratio, the ghost width (that of
// `finemfab`) and the absent EB index space are fixed here, so that the
// allocation (EnsureRKBuffers) and the empty-footprint check (FillRKBoundary)
// cannot drift apart. Cached by AMReX; the returned reference lives as long as
// `finemfab`'s layout does.
const amrex::FabArrayBase::FPinfo &
rk_fill_fpinfo(const amrex::MultiFab &finemfab,
               amrex::Interpolater *const interpolator,
               const amrex::Geometry &fgeom, const amrex::Geometry &cgeom) {
  const amrex::IntVect ratio{2, 2, 2};
  const amrex::EB2::IndexSpace *const index_space = nullptr;
  const amrex::InterpolaterBoxCoarsener &coarsener =
      interpolator->BoxCoarsener(ratio);
  return amrex::FabArrayBase::TheFPinfo(finemfab, finemfab,
                                        finemfab.nGrowVect(), coarsener, fgeom,
                                        cgeom, index_space);
}

} // namespace

// The single allocation site of the RK buffers of a group: the source bands
// (old_source_band, ks_source_band[]: data with history, filled by the parent
// level) and the work buffers of the fill (rk_crse_patch, the dense-output
// destination, and rk_fine_patch, the interpolation destination). All four
// are owned by the refined level's GroupData and have the geometry of one
// FPinfo, that of this level's MultiFab of the group, which
// FillPatch_Prolongate would look up as well. The dense-output kernel walks
// the bands and rk_crse_patch box by box; they have one layout by
// construction, so there is nothing to compare across levels.
//
// Lazy and idempotent, without a dirty check: the layout is a function of this
// level's own layout alone, and a regrid that changes it destroys the buffers
// together with their LevelData. The work buffers are allocated exactly as the
// temporaries of FillPatch_Prolongate that they replace (the body of AMReX's
// make_mf_crse_patch / make_mf_fine_patch), minus the NaN prefill outside the
// domain: the dense output overwrites every point of rk_crse_patch.
void EnsureRKBuffers(const int patch, const int level, const int gi) {
  // The buffers only exist under subcycling, and only for the groups the time
  // integrator advances: it alone fills the bands (StoreRKOldState /
  // StoreRKStage) and publishes that set in rk_integrated_group. do_evolve is
  // no substitute, since it defaults to the checkpoint flag and is thus also
  // set for checkpointed groups that are never integrated. Recovery calls this
  // for every group and then expects a mid-cycle checkpoint to carry each band
  // allocated here, so this must match what the evolution allocates. It does
  // for checkpoints written with this per-group geometry, and for older ones
  // from runs whose integrated groups did not mix prolongation operators or
  // ghost widths. A mid-cycle checkpoint that an older driver wrote from a run
  // that did mix them carries bands on another group's geometry; that is not
  // handled.
  if (!ghext->use_subcycling)
    return;
  const std::vector<bool> &integrated = ghext->rk_integrated_group;
  if (gi < 0 || gi >= int(integrated.size()) || !integrated[gi])
    return;

  assert(level >= 1);
  const auto &patchdata = ghext->patchdata.at(patch);
  const auto &leveldata = patchdata.leveldata.at(level);
  const GroupData &groupdata = *leveldata.groupdata.at(gi);
  const int num_rk_stages = ghext->num_rk_stages;
  assert(num_rk_stages >= 0 && num_rk_stages <= max_num_rk_stages);

  // All four are allocated together, or none is
  [[maybe_unused]] const auto have_all_or_none = [&]() {
    const bool have = bool(groupdata.rk_crse_patch);
    bool good = bool(groupdata.rk_fine_patch) == have &&
                bool(groupdata.old_source_band) == have;
    for (int stage = 0; stage < num_rk_stages; ++stage)
      good = good && bool(groupdata.ks_source_band[stage]) == have;
    return good;
  };
  assert(have_all_or_none());
  if (groupdata.rk_crse_patch)
    return;

  assert(!groupdata.mfab.empty());
  const amrex::MultiFab &mfab = *groupdata.mfab.at(0);
  const amrex::FabArrayBase::FPinfo &fpc = rk_fill_fpinfo(
      mfab, groupdata.interpolator, patchdata.amrcore->Geom(level),
      patchdata.amrcore->Geom(level - 1));
  // Empty coarse-fine footprint: all buffers stay null
  if (fpc.ba_crse_patch.empty())
    return;

  const int ncomps = mfab.nComp();
  groupdata.rk_crse_patch = std::make_unique<amrex::MultiFab>(
      fpc.ba_crse_patch, fpc.dm_patch, ncomps, 0, amrex::MFInfo(),
      *fpc.fact_crse_patch);
  groupdata.rk_fine_patch = std::make_unique<amrex::MultiFab>(
      fpc.ba_fine_patch, fpc.dm_patch, ncomps, 0, amrex::MFInfo(),
      *fpc.fact_fine_patch);

  // Zero-ghost source bands on the coarse patch geometry
  const int numvars = groupdata.numvars;
  assert(numvars == ncomps);
  for (int stage = 0; stage < num_rk_stages; ++stage)
    groupdata.ks_source_band[stage] = std::make_unique<amrex::MultiFab>(
        fpc.ba_crse_patch, fpc.dm_patch, numvars, 0);
  groupdata.old_source_band = std::make_unique<amrex::MultiFab>(
      fpc.ba_crse_patch, fpc.dm_patch, numvars, 0);

  assert(have_all_or_none());
}

void StoreRKOldState(const int patch, const int level,
                     const std::vector<int> &var_groups, const int tl) {
  if (!ghext->use_subcycling)
    return;
  const auto &patchdata = ghext->patchdata.at(patch);
  const auto &leveldata = patchdata.leveldata.at(level);
  const bool have_child = level + 1 < int(patchdata.leveldata.size());
  // The bands sit in this level's index space: this level's periodicity
  const amrex::Periodicity &period =
      patchdata.amrcore->Geom(level).periodicity();

  for (const int gi : var_groups) {
    const GroupData &groupdata = *leveldata.groupdata.at(gi);
    // EnsureRKBuffers is a no-op for unpublished groups, which would leave the
    // children without a prolongation source and recovery without bands.
    if (gi >= int(ghext->rk_integrated_group.size()) ||
        !ghext->rk_integrated_group[gi])
      CCTK_VERROR("Group \"%s\" is integrated under subcycling but is not "
                  "listed in GHExt::rk_integrated_group. The time integrator "
                  "must publish its evolved groups at WRAGH.",
                  CCTK_FullGroupName(gi));
    // The finest level has no children to prolongate to.
    if (!have_child)
      continue;
    // The child owns the bands. Lazy, idempotent allocation: a child that was
    // remade since this level's last step has lost its buffers and gets new
    // ones here, before its first fill.
    EnsureRKBuffers(patch, level + 1, gi);
    const GroupData &childgroupdata =
        *patchdata.leveldata.at(level + 1).groupdata.at(gi);
    // Empty coarse-fine footprint
    if (!childgroupdata.old_source_band)
      continue;
    copy_interior_to_band(*childgroupdata.old_source_band,
                          *groupdata.mfab.at(tl), period);
  }
}

void StoreRKStage(const int patch, const int level,
                  const std::vector<int> &var_groups,
                  const std::vector<int> &rhs_groups, const int stage) {
  if (!ghext->use_subcycling)
    return;
  assert(stage >= 1 && stage <= ghext->num_rk_stages);
  assert(var_groups.size() == rhs_groups.size());
  const int s = stage - 1;
  const auto &patchdata = ghext->patchdata.at(patch);
  // The finest level has no children to prolongate to.
  if (level + 1 >= int(patchdata.leveldata.size()))
    return;
  const auto &leveldata = patchdata.leveldata.at(level);
  const auto &childleveldata = patchdata.leveldata.at(level + 1);
  const amrex::Periodicity &period =
      patchdata.amrcore->Geom(level).periodicity();

  // rhs_groups[i] and var_groups[i] are paired by sort order; the k-stage
  // bands live on the evolved group's GroupData on the child level.
  for (size_t i = 0; i < var_groups.size(); ++i) {
    const GroupData &childgroupdata =
        *childleveldata.groupdata.at(var_groups[i]);
    const GroupData &rhs_groupdata = *leveldata.groupdata.at(rhs_groups[i]);
    // Empty coarse-fine footprint. StoreRKOldState allocated the bands
    // earlier in this step; nothing is allocated here.
    if (!childgroupdata.ks_source_band[s])
      continue;
    copy_interior_to_band(*childgroupdata.ks_source_band[s],
                          *rhs_groupdata.mfab.at(0), period);
  }
}

void FillRKBoundary(const int patch, const int level,
                    const std::vector<int> &var_groups, const int tl,
                    const int stage, const CCTK_REAL xsi, const CCTK_REAL dtc) {
  if (level == 0)
    return;
  if (!ghext->use_subcycling)
    return;

  const auto &patchdata = ghext->patchdata.at(patch);
  auto &leveldata = patchdata.leveldata.at(level);
  const auto &coarseleveldata = patchdata.leveldata.at(level - 1);
  const auto &fgeom = patchdata.amrcore->Geom(level);
  const auto &cgeom = patchdata.amrcore->Geom(level - 1);
  const int num_rk_stages = ghext->num_rk_stages;

  // We need to loop over groups in a definite order so that AMReX's
  // communication pattern does not get confused (as in
  // SyncGroupsByDirIProlongateOnly_impl), hence the serial task managers.
  // tasks1 evaluates the dense output, tasks2 starts the prolongation, tasks3
  // finishes it.
  task_manager tasks1;
  task_manager tasks2;
  task_manager tasks3;

  for (const int gi : var_groups) {
    GroupData &groupdata = *leveldata.groupdata.at(gi);
    const GroupData &coarsegroupdata = *coarseleveldata.groupdata.at(gi);
    assert(!groupdata.mfab.empty());
    assert(!coarsegroupdata.mfab.empty());
    assert(coarsegroupdata.numvars == groupdata.numvars);
    assert(groupdata.do_evolve);

    amrex::MultiFab &mfab = *groupdata.mfab.at(tl);

    // This level owns the bands; the parent filled them (StoreRKOldState ran
    // on it before this level stepped, or recovery read them). Nothing is
    // allocated here. A null band means an empty coarse-fine footprint, i.e.
    // nothing to prolongate -- and not that nobody allocated the buffers.
    if (!groupdata.old_source_band) {
      assert(rk_fill_fpinfo(mfab, groupdata.interpolator, fgeom, cgeom)
                 .ba_crse_patch.empty());
      continue;
    }
    const amrex::MultiFab &old_band = *groupdata.old_source_band;

    // As in FillPatch_Prolongate: without ghosts there is nothing to fill
    if (mfab.nGrowVect().max() == 0)
      continue;

    // Persistent work buffers, allocated together with the bands
    assert(groupdata.rk_crse_patch && groupdata.rk_fine_patch);
    amrex::MultiFab &crse_patch = *groupdata.rk_crse_patch;
    amrex::MultiFab &fine_patch = *groupdata.rk_fine_patch;

    // Coarse state at the fine stage time, evaluated on the bands straight
    // into the coarse patch buffer: both have the geometry
    // fpc.ba_crse_patch / fpc.dm_patch. Every point of the buffer is
    // overwritten. Points outside the domain hold whatever the bands hold
    // there; the coarse boundary conditions overwrite them next.
    tasks1.submit_serially(
        [&groupdata, &crse_patch, &old_band, num_rk_stages, stage, xsi, dtc]() {
          rk_dense_output(crse_patch, old_band, groupdata.ks_source_band,
                          num_rk_stages, stage, xsi, dtc);
        });

    // Coarse BC on the combined state, spatial interpolation with the group's
    // operator, scatter into the fine ghost halo, fine BC: the same back half
    // every other coarse-to-fine fill takes.
    tasks2.submit_serially([&groupdata, &coarsegroupdata, &mfab, &crse_patch,
                            &fine_patch, &fgeom, &cgeom]() {
      Prolongate_Start(groupdata, coarsegroupdata, mfab, crse_patch, fine_patch,
                       fgeom, cgeom, groupdata.interpolator, groupdata.bcrecs);
    });
    tasks3.submit_serially(
        [&groupdata, &mfab]() { Prolongate_Finish(groupdata, mfab); });
  }

  tasks1.run_tasks_serially();
  synchronize();
  tasks2.run_tasks_serially();
  synchronize();
  tasks3.run_tasks_serially();
  synchronize();
}

} // namespace CarpetX
