#include "subcycling.hxx"

#include "driver.hxx"
#include "fillpatch.hxx"
#include "schedule.hxx"
#include "subcycling_tally.hxx"
#include "task_manager.hxx"
#include "timer.hxx"

#include <AMReX_FabArray.H>      // MultiArray4, MultiFab::arrays()
#include <AMReX_GpuContainers.H> // amrex::GpuArray
#include <AMReX_IntVect.H>
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
// one BoxArray and one DistributionMapping (see ensure_rk_fill_buffers).
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
  synchronize_device();
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

// Make sure the persistent buffers of the RK fill into `groupdata` exist:
// rk_crse_patch (the dense-output destination) and rk_fine_patch (the
// interpolation destination). Lazy and idempotent, in the style of
// LevelData::build_bands, but without a dirty check: both buffers are a
// function of the fine level's layout alone -- the FPinfo of `mfab`, which
// FillPatch_Prolongate would look up as well -- and they are owned by the fine
// level's GroupData, so a regrid that changes that layout destroys them
// together with their LevelData. They are allocated exactly as the temporaries
// they replace, but are not charged to the counter report as temporaries: like
// the source bands, they are reported as memory held (buffer_bytes).
//
// The dense-output kernel walks rk_crse_patch and the parent's bands box by
// box, so both must have one layout. They do by construction: build_bands
// takes the band geometry from the same FPinfo and rebuilds it whenever this
// level's BoxArray or DistributionMapping changes. This is checked on every
// call, before anything is launched.
void ensure_rk_fill_buffers(const GroupData &groupdata,
                            const amrex::MultiFab &mfab,
                            const amrex::MultiFab &parent_band,
                            const amrex::Geometry &fgeom,
                            const amrex::Geometry &cgeom) {
  assert(bool(groupdata.rk_crse_patch) == bool(groupdata.rk_fine_patch));
  if (!groupdata.rk_crse_patch) {
    const amrex::IntVect &nghosts = mfab.nGrowVect();
    const int ncomps = mfab.nComp();
    const amrex::IntVect ratio{2, 2, 2};
    const amrex::EB2::IndexSpace *const index_space = nullptr;
    const amrex::InterpolaterBoxCoarsener &coarsener =
        groupdata.interpolator->BoxCoarsener(ratio);
    // Cached by AMReX; the same lookup as in FillPatch_Prolongate and
    // build_bands
    const amrex::FabArrayBase::FPinfo &fpc = amrex::FabArrayBase::TheFPinfo(
        mfab, mfab, nghosts, coarsener, fgeom, cgeom, index_space);
    // The parent holds a band, hence the coarse-fine footprint is not empty
    assert(!fpc.ba_crse_patch.empty());
    groupdata.rk_crse_patch = std::make_unique<amrex::MultiFab>(
        fpc.ba_crse_patch, fpc.dm_patch, ncomps, 0, amrex::MFInfo(),
        *fpc.fact_crse_patch);
    groupdata.rk_fine_patch = std::make_unique<amrex::MultiFab>(
        fpc.ba_fine_patch, fpc.dm_patch, ncomps, 0, amrex::MFInfo(),
        *fpc.fact_fine_patch);
  }

  // Layout equality with the parent's bands
  assert(groupdata.rk_crse_patch->boxArray() == parent_band.boxArray());
  assert(groupdata.rk_crse_patch->DistributionMap() ==
         parent_band.DistributionMap());
  assert(groupdata.rk_crse_patch->nComp() == parent_band.nComp());
  assert(groupdata.rk_fine_patch->DistributionMap() ==
         parent_band.DistributionMap());
  assert(groupdata.rk_fine_patch->nComp() == mfab.nComp());
}

} // namespace

void CountLincombLaunches(const int n) { charge_launches(n); }

void StoreRKOldState(const int patch, const int level,
                     const std::vector<int> &var_groups, const int tl) {
  static Timer timer("StoreRKOldState");
  Interval interval(timer);

  if (ghext->use_subcycling) {
    const auto &patchdata = ghext->patchdata.at(patch);
    const auto &leveldata = patchdata.leveldata.at(level);
    const amrex::Periodicity &period =
        patchdata.amrcore->Geom(level).periodicity();

    for (const int gi : var_groups) {
      const GroupData &groupdata = *leveldata.groupdata.at(gi);
      // build_bands is a no-op for unpublished groups, which would leave the
      // children without a prolongation source and recovery without bands.
      if (gi >= int(ghext->rk_integrated_group.size()) ||
          !ghext->rk_integrated_group[gi])
        CCTK_VERROR("Group \"%s\" is integrated under subcycling but is not "
                    "listed in GHExt::rk_integrated_group. The time integrator "
                    "must publish its evolved groups at WRAGH.",
                    CCTK_FullGroupName(gi));
      // Lazy, idempotent allocation with a child-layout dirty check. The band
      // geometry reads the next-finer level, so all levels must already exist.
      leveldata.build_bands(groupdata);
      // The finest level has no source band (no children to prolongate to).
      if (!groupdata.old_source_band)
        continue;
      copy_interior_to_band(*groupdata.old_source_band, *groupdata.mfab.at(tl),
                            period);
    }
  }

  // Idle-on-return contract (see subcycling.hxx): the one wait of this
  // primitive. It is deliberately outside every branch and after the loop, so
  // that it is reached when no group has a band (finest level), when the group
  // list is empty, and for groups that `continue` above. The caller relies on
  // it to drain kernels it issued before the call (ODESolvers' deferred copy
  // of the old state), not only the band copies.
  synchronize();
}

void StoreRKStage(const int patch, const int level,
                  const std::vector<int> &var_groups,
                  const std::vector<int> &rhs_groups, const int stage) {
  static Timer timer("StoreRKStage");
  Interval interval(timer);

  if (ghext->use_subcycling) {
    assert(stage >= 1 && stage <= ghext->num_rk_stages);
    assert(var_groups.size() == rhs_groups.size());
    const int s = stage - 1;
    const auto &patchdata = ghext->patchdata.at(patch);
    const auto &leveldata = patchdata.leveldata.at(level);
    const amrex::Periodicity &period =
        patchdata.amrcore->Geom(level).periodicity();

    // rhs_groups[i] and var_groups[i] are paired by sort order; the k-stage
    // bands live on the evolved group's GroupData.
    for (size_t i = 0; i < var_groups.size(); ++i) {
      const GroupData &groupdata = *leveldata.groupdata.at(var_groups[i]);
      const GroupData &rhs_groupdata = *leveldata.groupdata.at(rhs_groups[i]);
      // The finest level has no source band (no children to prolongate to).
      if (!groupdata.ks_source_band[s])
        continue;
      copy_interior_to_band(*groupdata.ks_source_band[s],
                            *rhs_groupdata.mfab.at(0), period);
    }
  }

  // Idle-on-return contract (see subcycling.hxx): the one wait of this
  // primitive, reached on every path (see StoreRKOldState). On the finest
  // level, and on level 0 where there is no refinement-boundary fill, this is
  // the only wait that drains the stage's linear combinations, which
  // ODESolvers issues without a wait of their own, before the next RHS
  // evaluation reads the new state on other streams.
  synchronize();
}

void FillRKBoundary(const int patch, const int level,
                    const std::vector<int> &var_groups, const int tl,
                    const int stage, const CCTK_REAL xsi, const CCTK_REAL dtc) {
  if (level == 0)
    return;
  if (!ghext->use_subcycling)
    return;

  static Timer timer("FillRKBoundary");
  Interval interval(timer);

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

    // The parent must have built its band geometry (StoreRKOldState ran on it
    // before this level stepped); a null band then means an empty coarse-fine
    // footprint, i.e. nothing to prolongate.
    const std::array<int, dim> &indextype = coarsegroupdata.indextype;
    const int cs = (indextype[0] << 2) | (indextype[1] << 1) | indextype[2];
    assert(coarseleveldata.source_band_ba[cs]);
    if (!coarsegroupdata.old_source_band)
      continue;
    const amrex::MultiFab &old_band = *coarsegroupdata.old_source_band;

    amrex::MultiFab &mfab = *groupdata.mfab.at(tl);
    // As in FillPatch_Prolongate: without ghosts there is nothing to fill
    if (mfab.nGrowVect().max() == 0)
      continue;

    // Persistent work buffers, owned by this (fine) level; allocated by the
    // first fill after the level was made. There is no precondition on the
    // caller: the recovery fill allocates them just the same.
    ensure_rk_fill_buffers(groupdata, mfab, old_band, fgeom, cgeom);
    amrex::MultiFab &crse_patch = *groupdata.rk_crse_patch;
    amrex::MultiFab &fine_patch = *groupdata.rk_fine_patch;

    // Coarse state at the fine stage time, evaluated on the parent's bands
    // straight into the coarse patch buffer: both have the geometry
    // fpc.ba_crse_patch / fpc.dm_patch. Every point of the buffer is
    // overwritten. Points outside the domain hold whatever the bands hold
    // there; the coarse boundary conditions overwrite them next.
    tasks1.submit_serially([&coarsegroupdata, &crse_patch, &old_band,
                            num_rk_stages, stage, xsi, dtc]() {
      rk_dense_output(crse_patch, old_band, coarsegroupdata.ks_source_band,
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

  // The three waits are unconditional: they are also reached when every group
  // was skipped above, so the last one always upholds the idle-on-return
  // contract (see subcycling.hxx).
  tasks1.run_tasks_serially();
  synchronize();
  tasks2.run_tasks_serially();
  synchronize();
  tasks3.run_tasks_serially();
  synchronize();
}

} // namespace CarpetX
