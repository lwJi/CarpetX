#include "comm/rk_ghosts.hxx"

#include "config/indices.hxx"
#include "hierarchy/model.hxx"
#include "comm/buffers_inside.hxx"
#include "comm/ghost_prolongation.hxx"
#include "comm/timelevels.hxx"
#include "comm/fillpatch_inside.hxx"
#include "comm/integrator.hxx"
#include "comm/integrator_inside.hxx"
#include "comm/prolongate_3d_rf2_inside.hxx"
#include "comm/substeps.hxx"
#include "comm/substeps_inside.hxx"
#include "hierarchy/scopes.hxx"
#include "util/refuse.hxx"
#include "util/task_manager.hxx"
#include <carpetx.hxx>
#include <carpetx_rk.hxx>

#include <cctk.h>
#include <cctk_Parameters.h>

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
#include <cmath>
#include <limits>
#include <memory>
#include <optional>
#include <source_location>
#include <string>
#include <vector>

namespace CarpetX {

namespace {

using GroupData = GHExt::PatchData::LevelData::GroupData;

constexpr CCTK_REAL rk_substep_ratio = CCTK_REAL(1) / time_refinement;
static_assert(rk_substep_ratio * time_refinement == 1);

template <int RKSTAGES>
constexpr std::array<CCTK_REAL, RKSTAGES>
rk_dense_output_weights(const CCTK_REAL xsi) {
  static_assert(RKSTAGES == 3 || RKSTAGES == 4,
                "rk_dense_output_weights only supports RKSTAGES == 3 or 4");
  const CCTK_REAL xsi2 = xsi * xsi;
  if constexpr (RKSTAGES == 3) {
    return {xsi - (5. / 6.) * xsi2, (1. / 6.) * xsi2, (2. / 3.) * xsi2};
  } else {
    const CCTK_REAL xsi3 = xsi2 * xsi;
    return {xsi - 1.5 * xsi2 + (2. / 3.) * xsi3, xsi2 - (2. / 3.) * xsi3,
            xsi2 - (2. / 3.) * xsi3, -0.5 * xsi2 + (2. / 3.) * xsi3};
  }
}

template <rk_dense_output_t dense_output>
constexpr bool rk_dense_output_weights_end_at_method_b() {
  constexpr rk_dense_output_method_t method =
      rk_dense_output_method(dense_output);
  const std::array<CCTK_REAL, method.nstages> b =
      rk_dense_output_weights<method.nstages>(1);
  for (int s = 0; s < method.nstages; ++s) {
    const CCTK_REAL difference = b[s] - method.b[s];
    if (!(difference <= 4 * std::numeric_limits<CCTK_REAL>::epsilon() &&
          -difference <= 4 * std::numeric_limits<CCTK_REAL>::epsilon()))
      return false;
  }
  return true;
}

static_assert(
    rk_dense_output_weights_end_at_method_b<rk_dense_output_t::ssprk3>(),
    "rk_ghosts: the ssprk3 dense-output weights at xsi = 1 must be the b of "
    "CarpetX::rk_dense_output_method(rk_dense_output_t::ssprk3)");
static_assert(
    rk_dense_output_weights_end_at_method_b<rk_dense_output_t::rk4>(),
    "rk_ghosts: the rk4 dense-output weights at xsi = 1 must be the b of "
    "CarpetX::rk_dense_output_method(rk_dense_output_t::rk4)");

struct rk_fine_stage_t {
  CCTK_REAL f, df_f, d2f_f_f, df_df_f;
};

constexpr rk_fine_stage_t rk_fine_stage(const rk_dense_output_method_t &method,
                                        const int s) {
  rk_fine_stage_t taylor{method.c[s], 0, 0, 0};
  for (int j = 0; j < s; ++j) {
    taylor.df_f += method.a[s][j] * method.c[j];
    taylor.d2f_f_f += method.a[s][j] * method.c[j] * method.c[j] / 2;
    for (int k = 0; k < j; ++k)
      taylor.df_df_f += method.a[s][j] * method.a[j][k] * method.c[k];
  }
  return taylor;
}

template <rk_dense_output_t dense_output>
void rk_dense_output_impl(amrex::MultiFab &crse_patch,
                          const rk_sources_t::sources_t src,
                          const stage_t stage, const CCTK_REAL xsi,
                          const CCTK_REAL dtc) {
  constexpr rk_dense_output_method_t method =
      rk_dense_output_method(dense_output);
  constexpr int RKSTAGES = method.nstages;
  static_assert(RKSTAGES == 3 || RKSTAGES == 4,
                "rk_dense_output only supports RKSTAGES == 3 or 4");
  const stage_range stages = rk_stages(RKSTAGES);
  assert(stages.contains(stage));
  assert(int(src.ks.size()) == RKSTAGES);
  const amrex::MultiFab &old_band = src.old;

  constexpr CCTK_REAL r = rk_substep_ratio;
  const rk_fine_stage_t fine = rk_fine_stage(method, stages.offset(stage));
  const CCTK_REAL at = fine.f * r;
  const CCTK_REAL att = fine.df_f * r * r;

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
  for (const stage_t ks_stage : stages) {
    const amrex::MultiFab &ks_band = src.ks[stages.offset(ks_stage)];
    assert(ks_band.nComp() == nvars);
    assert(ks_band.boxArray() == crse_patch.boxArray());
    assert(ks_band.DistributionMap() == crse_patch.DistributionMap());
    kcs_arrs[stages.offset(ks_stage)] = ks_band.const_arrays();
  }

  if constexpr (RKSTAGES == 3) {
    const std::array<CCTK_REAL, 3> b = rk_dense_output_weights<3>(xsi);
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

    if (stage == stage_t{1}) {
      amrex::ParallelFor(
          crse_patch, amrex::IntVect{0}, nvars,
          [=] AMREX_GPU_DEVICE(int b_, int i, int j, int k, int n) noexcept {
            const std::array<CCTK_REAL, 3> kk = {kcs_arrs[0][b_](i, j, k, n),
                                                 kcs_arrs[1][b_](i, j, k, n),
                                                 kcs_arrs[2][b_](i, j, k, n)};
            const CCTK_REAL uu = b[0] * kk[0] + b[1] * kk[1] + b[2] * kk[2];
            yf_arrs[b_](i, j, k, n) += dtc * uu;
          });
    } else if (stage == stage_t{2}) {
      amrex::ParallelFor(
          crse_patch, amrex::IntVect{0}, nvars,
          [=] AMREX_GPU_DEVICE(int b_, int i, int j, int k, int n) noexcept {
            const std::array<CCTK_REAL, 3> kk = {kcs_arrs[0][b_](i, j, k, n),
                                                 kcs_arrs[1][b_](i, j, k, n),
                                                 kcs_arrs[2][b_](i, j, k, n)};
            const CCTK_REAL uu = b[0] * kk[0] + b[1] * kk[1] + b[2] * kk[2];
            const CCTK_REAL ut = bt[0] * kk[0] + bt[1] * kk[1] + bt[2] * kk[2];
            yf_arrs[b_](i, j, k, n) += dtc * (uu + at * ut);
          });
    } else { // stage 3
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
            yf_arrs[b_](i, j, k, n) += dtc * (uu + at * ut + att * utt);
          });
    }
  } else { // RKSTAGES == 4
    const std::array<CCTK_REAL, 4> b = rk_dense_output_weights<4>(xsi);
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

    if (stage == stage_t{1}) {
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
    } else if (stage == stage_t{2}) {
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
            yf_arrs[b_](i, j, k, n) += dtc * (uu + at * ut);
          });
    } else { // stage 3 or stage 4
      static_assert(method.c[1] == method.c[2],
                    "rk_ghosts: the rk4 kernel takes f'f'f from k3 - k2, "
                    "whose f'f term cancels only if stages 2 and 3 share a "
                    "stage time");
      constexpr CCTK_REAL k3_minus_k2 =
          rk_fine_stage(method, 2).df_f - rk_fine_stage(method, 1).df_f;
      const CCTK_REAL attt = fine.d2f_f_f * r * r * r;
      const CCTK_REAL ak =
          (fine.df_df_f - fine.d2f_f_f) / (fine.d2f_f_f * k3_minus_k2);
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

void rk_dense_output(amrex::MultiFab &crse_patch,
                     const rk_sources_t::sources_t src,
                     const rk_dense_output_t dense_output, const stage_t stage,
                     const CCTK_REAL xsi, const CCTK_REAL dtc) {
  switch (dense_output) {
  case rk_dense_output_t::ssprk3:
    rk_dense_output_impl<rk_dense_output_t::ssprk3>(crse_patch, src, stage, xsi,
                                                    dtc);
    break;
  case rk_dense_output_t::rk4:
    rk_dense_output_impl<rk_dense_output_t::rk4>(crse_patch, src, stage, xsi,
                                                 dtc);
    break;
  default:
    CCTK_ERROR("Subcycling dense output: the registered integrator has no "
               "dense output (CarpetX::register_integrator refuses that under "
               "CarpetX::use_subcycling)");
  }
}

[[maybe_unused]] bool
bands_match_footprint(const std::optional<rk_buffers_t> &rk,
                      const amrex::FabArrayBase::FPinfo &fp) {
  if (!rk)
    return fp.ba_crse_patch.empty();
  return rk->fill_patches->footprint == fp.ba_fine_patch;
}

} // namespace

void StoreRKOldState(const level_t level,
                     const std::span<const gi_t> var_groups, const tl_t tl) {
  static Timer timer("StoreRKOldState");
  Interval interval(timer);

  const GHExt &ext = driver();
  const auto &patchdata = ext.patch(level.patch());
  const auto &leveldata = patchdata.level(level);
  const std::optional<level_t> child = level.child();
  const amrex::Periodicity &period =
      patchdata.amrcore().Geom(level.index()).periodicity();
  const rk_source_writer_t storing;
  std::vector<rk_sources_t::store_t> stores;

  for (const gi_t gi : var_groups) {
    const GroupData &groupdata = leveldata.group(gi);
    if (!group_is_integrated(gi))
      CCTK_VERROR("Group \"%s\" is integrated under subcycling but is not "
                  "among the integrated groups the time integrator registered "
                  "(CarpetX::register_integrator, AT WRAGH)",
                  CCTK_FullGroupName(gi.index()));
    if (!child)
      continue;
    const std::optional<rk_buffers_t> &rk =
        buffers_of(patchdata.level(*child), gi).rk;
    if (!rk)
      continue;
    stores.push_back(
        {storing.open(rk->sources), groupdata.timelevel(tl).mfab()});
  }

  rk_sources_t::store_old(stores, period, leveldata.iteration());
}

void StoreRKStage(const level_t level, const std::span<const gi_t> var_groups,
                  const std::span<const gi_t> rhs_groups, const stage_t stage) {
  static Timer timer("StoreRKStage");
  Interval interval(timer);

  const GHExt &ext = driver();
  assert(registered_stages().contains(stage));
  assert(var_groups.size() == rhs_groups.size());
  const auto &patchdata = ext.patch(level.patch());
  const std::optional<level_t> child = level.child();
  if (!child)
    return;
  const auto &leveldata = patchdata.level(level);
  const auto &childleveldata = patchdata.level(*child);
  const amrex::Periodicity &period =
      patchdata.amrcore().Geom(level.index()).periodicity();
  const rk_source_writer_t storing;
  std::vector<rk_sources_t::store_t> stores;

  for (size_t i = 0; i < var_groups.size(); ++i) {
    const std::optional<rk_buffers_t> &rk =
        buffers_of(childleveldata, var_groups[i]).rk;
    const GroupData &rhs_groupdata = leveldata.group(rhs_groups[i]);
    if (!rk)
      continue;
    stores.push_back({storing.open(rk->sources),
                      rhs_groupdata.timelevel(tl_t{0}).mfab()});
  }

  rk_sources_t::store_stage(stores, stage, period,
                            std::source_location::current());
}

void FillRKBoundary(const level_t child, const std::span<const gi_t> states,
                    const stage_t completed, const std::string &event) {
  static Timer timer("FillRKBoundary");
  Interval interval(timer);

  const GHExt &ext = driver();
  assert(ext.use_subcycling);
  assert(child.index() > 0);

  auto &patchdata = ext.patch(child.patch());
  auto &leveldata = patchdata.level(child);
  const level_t parent = child.parent();
  const auto &coarseleveldata = patchdata.level(parent);
  const auto &fgeom = patchdata.amrcore().Geom(child.index());
  const auto &cgeom = patchdata.amrcore().Geom(parent.index());
  const rk_dense_output_t dense_output = registered_dense_output();
  const stage_range stages = registered_stages();
  assert(stages.contains(completed));
  const substep_t substep = substep_of_parent_step(child);
  const bool substep_done = completed == stages.last();
  const stage_t stage = substep_done ? stages.first() : next_index(completed);
  const CCTK_REAL xsi = CCTK_REAL(substep_done ? substep.end : substep.begin);
  const CCTK_REAL dtc =
      ext.run_clock().delta_time * CCTK_REAL(coarseleveldata.delta_iteration());
  assert(std::isfinite(dtc) && dtc > 0);

  const window_t window = window_t::one_level(child);
  write_scopes_t scopes;
  for (const gi_t gi : states) {
    scopes.open(window, {.gi = gi,
                         .tl = rk_tl,
                         .fills = make_valid_coarse_fine(),
                         .keeps = ~make_valid_coarse_fine(),
                         .event = event});
  }

  task_manager tasks1;
  task_manager tasks2;
  const rk_fill_patch_writer_t filling;

  for (const gi_t gi : states) {
    const GroupData &groupdata = leveldata.group(gi);
    const GroupData &coarsegroupdata = coarseleveldata.group(gi);
    assert(coarsegroupdata.record.numvars == groupdata.record.numvars);
    // Only integrated groups have bands, and only for them does the
    // subcycling SYNC leave the coarse-fine ghosts to this function
    assert(group_is_integrated(gi));

    amrex::MultiFab &mfab = scopes.mfab(child, gi, rk_tl);

    // As in FillPatch_Prolongate: without ghosts there is nothing to fill
    if (mfab.nGrowVect().max() == 0)
      continue;

    const std::optional<rk_buffers_t> &rk = buffers_of(leveldata, gi).rk;
    assert(bands_match_footprint(
        rk,
        rk_fill_fpinfo(mfab, interpolator_of(groupdata.record), fgeom, cgeom)));
    if (!rk)
      continue;
    const rk_sources_t::sources_t src =
        rk->sources.get().sources(coarseleveldata.iteration(), {child, gi});
    rk_fill_patches_t &patches = filling.open(rk->fill_patches);
    amrex::MultiFab &crse_patch = patches.crse;
    amrex::MultiFab &fine_patch = patches.fine;

    tasks1.submit_serially([&crse_patch, src, dense_output, stage, xsi, dtc]() {
      rk_dense_output(crse_patch, src, dense_output, stage, xsi, dtc);
    });

    tasks2.submit_serially([&groupdata, &coarsegroupdata, &mfab, &crse_patch,
                            &fine_patch, &fgeom, &cgeom]() {
      Prolongate_WithinRank(groupdata, coarsegroupdata, mfab, crse_patch,
                            fine_patch, fgeom, cgeom);
    });
  }

  tasks1.run_tasks_serially();
  synchronize();
  tasks2.run_tasks_serially();
  synchronize();
}

void RecoverRKBoundary(const window_t window) {
  DECLARE_CCTK_PARAMETERS;
  const GHExt &ext = driver();
  if (!ext.use_subcycling)
    return;
  const std::optional<GHExt::integrator_t> &integrator = ext.integrator();
  if (!integrator)
    return;
  if (integrator->rhs.empty())
    return;
  const std::vector<gi_t> states = integrated_states(*integrator);
  const operation_scope_t op("CarpetX::RecoverRKBoundary");

  if (verbose)
    CCTK_VINFO("RecoverRKBoundary on %s: refilling the refinement-boundary "
               "ghosts of the integrated groups (spatial prolongation on "
               "time-aligned levels of a restricted window, dense output from "
               "the level's restored source bands otherwise)",
               describe(window).c_str());

  static Timer timer("RecoverRKBoundary");
  Interval interval(timer);

  const std::vector<rk_level_t> levels = integrated_levels(window, *integrator);
  for (const rk_level_t &entry : levels)
    for (const gi_t gi : entry.groups.states)
      require(window_t::one_level(entry.level),
              {.gi = gi,
               .tl = rk_tl,
               .where = make_valid_int(),
               .event = "CarpetX::RecoverRKBoundary requires the tl=0 "
                        "interior of the integrated groups to be populated by "
                        "checkpoint recovery"});

  const bool restricted = window_is_restricted(window);
  if (restricted)
    ProlongateAligned(window, states, timelevels_t::one(rk_tl));

  for (const rk_level_t &entry : levels) {
    if (entry.level.index() == 0)
      continue;
    const bool aligned = aligned_with_parent(entry.level);
    if (aligned != (entry.level.index() > window.level_bounds().min))
      CCTK_VERROR("CarpetX::RecoverRKBoundary on %s: refinement level %d is "
                  "%s its parent, but recovery traverses each clock group "
                  "separately, so only the group's coarsest level lags its "
                  "parent",
                  describe(window).c_str(), entry.level.index(),
                  aligned ? "aligned with" : "lagging");
    if (restricted && aligned)
      continue;
    FillRKBoundary(entry.level, entry.groups.states,
                   integrator->stages().last(),
                   "CarpetX::RecoverRKBoundary after recovery");
  }
  synchronize();
}

} // namespace CarpetX
