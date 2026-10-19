#include "subcycling.hxx"

#include "driver.hxx"
#include "fillpatch.hxx"
#include "schedule.hxx"
#include "task_manager.hxx"

#include <AMReX_FabArray.H>
#include <AMReX_FabArrayBase.H>
#include <AMReX_Geometry.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_GpuDevice.H>
#include <AMReX_IntVect.H>
#include <AMReX_Interpolater.H>
#include <AMReX_MFParallelFor.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Periodicity.H>

#include <array>
#include <cassert>
#include <memory>
#include <vector>

namespace CarpetX {

namespace {

using GroupData = GHExt::PatchData::LevelData::GroupData;

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

  constexpr CCTK_REAL r = 0.5;

  const int nvars = crse_patch.nComp();
  assert(old_band.nComp() == nvars);
  assert(old_band.boxArray() == crse_patch.boxArray());
  assert(old_band.DistributionMap() == crse_patch.DistributionMap());

  const CCTK_REAL xsi2 = xsi * xsi;

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
    const std::array<CCTK_REAL, 3> b = {xsi - (5. / 6.) * xsi2,
                                        (1. / 6.) * xsi2, (2. / 3.) * xsi2};
    const std::array<CCTK_REAL, 3> bt = {1.0 - (5. / 3.) * xsi, (1. / 3.) * xsi,
                                         (4. / 3.) * xsi};
    constexpr std::array<CCTK_REAL, 3> btt = {-5. / 3., 1. / 3., 4. / 3.};

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
            yf_arrs[b_](i, j, k, n) += dtc * (uu + r * ut);
          });
    } else {
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
  } else {
    const CCTK_REAL xsi3 = xsi2 * xsi;

    const std::array<CCTK_REAL, 4> b = {
        xsi - 1.5 * xsi2 + (2. / 3.) * xsi3, xsi2 - (2. / 3.) * xsi3,
        xsi2 - (2. / 3.) * xsi3, -0.5 * xsi2 + (2. / 3.) * xsi3};
    const std::array<CCTK_REAL, 4> bt = {
        1.0 - 3.0 * xsi + 2.0 * xsi2, 2.0 * xsi - 2.0 * xsi2,
        2.0 * xsi - 2.0 * xsi2, -xsi + 2.0 * xsi2};
    const std::array<CCTK_REAL, 4> btt = {-3.0 + 4.0 * xsi, 2.0 - 4.0 * xsi,
                                          2.0 - 4.0 * xsi, -1.0 + 4.0 * xsi};
    constexpr std::array<CCTK_REAL, 4> bttt = {4.0, -4.0, -4.0, 4.0};

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
    } else {
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

  amrex::Gpu::synchronize();
}

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

void copy_interior_to_band(amrex::MultiFab &band, const amrex::MultiFab &src,
                           const amrex::Periodicity &period) {
  assert(band.ixType() == src.ixType());
  assert(band.nComp() == src.nComp());
  band.ParallelCopy(src, 0, 0, band.nComp(), amrex::IntVect{0},
                    amrex::IntVect{0}, period);
}

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

void EnsureRKBuffers(const int patch, const int level, const int gi) {
  if (!ghext->use_subcycling)
    return;
  if (!group_is_integrated(gi))
    return;

  assert(level >= 1);
  const auto &patchdata = ghext->patchdata.at(patch);
  const auto &leveldata = patchdata.leveldata.at(level);
  const GroupData &groupdata = *leveldata.groupdata.at(gi);
  const int num_rk_stages = ghext->num_rk_stages;
  assert(num_rk_stages >= 0 && num_rk_stages <= max_num_rk_stages);

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
  if (fpc.ba_crse_patch.empty())
    return;

  const int ncomps = mfab.nComp();
  groupdata.rk_crse_patch = std::make_unique<amrex::MultiFab>(
      fpc.ba_crse_patch, fpc.dm_patch, ncomps, 0, amrex::MFInfo(),
      *fpc.fact_crse_patch);
  groupdata.rk_fine_patch = std::make_unique<amrex::MultiFab>(
      fpc.ba_fine_patch, fpc.dm_patch, ncomps, 0, amrex::MFInfo(),
      *fpc.fact_fine_patch);

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
  const amrex::Periodicity &period =
      patchdata.amrcore->Geom(level).periodicity();

  for (const int gi : var_groups) {
    const GroupData &groupdata = *leveldata.groupdata.at(gi);
    if (!group_is_integrated(gi))
      CCTK_VERROR("Group \"%s\" is integrated under subcycling but is not "
                  "listed in GHExt::rk_integrated_group. The time integrator "
                  "must publish its evolved groups at WRAGH.",
                  CCTK_FullGroupName(gi));
    if (!have_child)
      continue;
    EnsureRKBuffers(patch, level + 1, gi);
    const GroupData &childgroupdata =
        *patchdata.leveldata.at(level + 1).groupdata.at(gi);
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
  if (level + 1 >= int(patchdata.leveldata.size()))
    return;
  const auto &leveldata = patchdata.leveldata.at(level);
  const auto &childleveldata = patchdata.leveldata.at(level + 1);
  const amrex::Periodicity &period =
      patchdata.amrcore->Geom(level).periodicity();

  for (size_t i = 0; i < var_groups.size(); ++i) {
    const GroupData &childgroupdata =
        *childleveldata.groupdata.at(var_groups[i]);
    const GroupData &rhs_groupdata = *leveldata.groupdata.at(rhs_groups[i]);
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

  task_manager tasks1;
  task_manager tasks2;
  task_manager tasks3;

  for (const int gi : var_groups) {
    GroupData &groupdata = *leveldata.groupdata.at(gi);
    const GroupData &coarsegroupdata = *coarseleveldata.groupdata.at(gi);
    assert(!groupdata.mfab.empty());
    assert(!coarsegroupdata.mfab.empty());
    assert(coarsegroupdata.numvars == groupdata.numvars);
    assert(group_is_integrated(gi));

    amrex::MultiFab &mfab = *groupdata.mfab.at(tl);

    if (mfab.nGrowVect().max() == 0)
      continue;

    if (!groupdata.old_source_band) {
      assert(rk_fill_fpinfo(mfab, groupdata.interpolator, fgeom, cgeom)
                 .ba_crse_patch.empty());
      continue;
    }
    const amrex::MultiFab &old_band = *groupdata.old_source_band;

    assert(groupdata.rk_crse_patch && groupdata.rk_fine_patch);
    amrex::MultiFab &crse_patch = *groupdata.rk_crse_patch;
    amrex::MultiFab &fine_patch = *groupdata.rk_fine_patch;

    tasks1.submit_serially(
        [&groupdata, &crse_patch, &old_band, num_rk_stages, stage, xsi, dtc]() {
          rk_dense_output(crse_patch, old_band, groupdata.ks_source_band,
                          num_rk_stages, stage, xsi, dtc);
        });

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
