#include "comm/subcycling.hxx"

#include "hierarchy/driver.hxx"
#include "hierarchy/group_tags.hxx"
#include "comm/buffers.hxx"
#include "comm/comm.hxx"
#include "comm/fillpatch.hxx"
#include "comm/prolongate_3d_rf2.hxx"
#include "hierarchy/loops.hxx"
#include "hierarchy/valid.hxx"
#include "util/refuse.hxx"
#include "util/task_manager.hxx"
#include <carpetx.hxx>

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
#include <AMReX_FluxRegister.H>
#include <AMReX_Orientation.H>

#include <array>
#include <cassert>
#include <cmath>
#include <iomanip>
#include <memory>
#include <optional>
#include <source_location>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace CarpetX {

namespace {

using GroupData = GHExt::PatchData::LevelData::GroupData;

template <int RKSTAGES>
void rk_dense_output_impl(amrex::MultiFab &crse_patch,
                          const rk_bands_t::sources_t src, const stage_t stage,
                          const CCTK_REAL xsi, const CCTK_REAL dtc) {
  static_assert(RKSTAGES == 3 || RKSTAGES == 4,
                "rk_dense_output only supports RKSTAGES == 3 or 4");
  const stage_range stages = rk_stages(RKSTAGES);
  assert(stages.contains(stage));
  assert(int(src.ks.size()) == RKSTAGES);
  const amrex::MultiFab &old_band = src.old;

  constexpr CCTK_REAL r = rk_substep_ratio;

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
            yf_arrs[b_](i, j, k, n) += dtc * (uu + 0.5 * r * ut);
          });
    } else { // stage 3 or stage 4
      const CCTK_REAL r2 = r * r;
      const CCTK_REAL r3 = r2 * r;
      const bool third = stage == stage_t{3};
      const CCTK_REAL at = third ? 0.5 * r : r;
      const CCTK_REAL att = third ? 0.25 * r2 : 0.5 * r2;
      const CCTK_REAL attt = third ? 0.0625 * r3 : 0.125 * r3;
      const CCTK_REAL ak = third ? -4.0 : 4.0;
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
                     const rk_bands_t::sources_t src,
                     const rk_dense_output_t dense_output, const stage_t stage,
                     const CCTK_REAL xsi, const CCTK_REAL dtc) {
  switch (dense_output) {
  case rk_dense_output_t::ssprk3:
    rk_dense_output_impl<3>(crse_patch, src, stage, xsi, dtc);
    break;
  case rk_dense_output_t::rk4:
    rk_dense_output_impl<4>(crse_patch, src, stage, xsi, dtc);
    break;
  default:
    CCTK_ERROR("Subcycling dense output: the registered integrator has no "
               "dense output (CarpetX::register_integrator refuses that under "
               "CarpetX::use_subcycling)");
  }
}

[[maybe_unused]] bool
bands_match_footprint(std::optional<rk_bands_t> &bands,
                      const amrex::FabArrayBase::FPinfo &fp) {
  if (!bands)
    return fp.ba_crse_patch.empty();
  return bands->crse_patch().boxArray() == fp.ba_crse_patch &&
         bands->fine_patch().boxArray() == fp.ba_fine_patch;
}

std::string ks_source_band_tag(const stage_t stage) {
  std::ostringstream buf;
  buf << "kss_s" << std::setw(2) << std::setfill('0')
      << all_rk_stages.offset(stage);
  return buf.str();
}

std::string old_source_band_tag() { return "olds"; }

std::string flux_register_band_tag(const amrex::Orientation face) {
  assert(face.coordDir() >= 0 && face.coordDir() < dim);
  std::ostringstream buf;
  buf << "freg_" << "xyz"[face.coordDir()] << (face.isHigh() ? "hi" : "lo");
  return buf.str();
}

std::vector<checkpoint_band_t>
list_bands(group_buffers_t &buffers,
           const std::optional<rk_bands_t::slots_t> rk_slots) {
  std::vector<checkpoint_band_t> bands;
  const auto add_band = [&](amrex::MultiFab *const band, std::string tag) {
    if (band && !band->empty())
      bands.push_back({std::move(tag), band});
  };
  if (rk_slots) {
    const stage_range stages = rk_stages(int(rk_slots->ks.size()));
    for (const stage_t stage : stages)
      add_band(&rk_slots->ks[stages.offset(stage)], ks_source_band_tag(stage));
    add_band(&rk_slots->old, old_source_band_tag());
  }
  if (buffers.freg)
    for (int dir = 0; dir < dim; ++dir)
      for (const auto side :
           {amrex::Orientation::low, amrex::Orientation::high}) {
        const amrex::Orientation face(dir, side);
        add_band(&(*buffers.freg)[face].multiFab(),
                 flux_register_band_tag(face));
      }
  return bands;
}

} // namespace

void StoreRKOldState(const level_t level,
                     const std::span<const gi_t> var_groups, const tl_t tl) {
  const GHExt &ext = driver();
  const auto &patchdata = ext.patch(level.patch());
  const auto &leveldata = patchdata.level(level);
  const std::optional<level_t> child = level.child();
  const amrex::Periodicity &period =
      patchdata.amrcore->Geom(level.index()).periodicity();

  for (const gi_t gi : var_groups) {
    const GroupData &groupdata = leveldata.group(gi);
    if (!group_is_integrated(gi))
      CCTK_VERROR("Group \"%s\" is integrated under subcycling but is not "
                  "among the integrated groups the time integrator registered "
                  "(CarpetX::register_integrator, AT WRAGH)",
                  CCTK_FullGroupName(gi.index()));
    if (!child)
      continue;
    std::optional<rk_bands_t> &bands =
        buffers_of(patchdata.level(*child), gi).rk_bands;
    if (!bands)
      continue;
    bands->store_old(*groupdata.timelevel(tl).mfab, period,
                     leveldata.iteration);
  }
}

void StoreRKStage(const level_t level, const std::span<const gi_t> var_groups,
                  const std::span<const gi_t> rhs_groups, const stage_t stage) {
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
      patchdata.amrcore->Geom(level.index()).periodicity();

  for (size_t i = 0; i < var_groups.size(); ++i) {
    std::optional<rk_bands_t> &bands =
        buffers_of(childleveldata, var_groups[i]).rk_bands;
    const GroupData &rhs_groupdata = leveldata.group(rhs_groups[i]);
    if (!bands)
      continue;
    bands->store_stage(stage, *rhs_groupdata.timelevel(tl_t{0}).mfab, period,
                       std::source_location::current());
  }
}

void FillRKBoundary(const level_t child, const std::span<const gi_t> states,
                    const stage_t completed, const std::string &event) {
  const GHExt &ext = driver();
  assert(ext.use_subcycling);
  assert(child.index() > 0);

  auto &patchdata = ext.patch(child.patch());
  auto &leveldata = patchdata.level(child);
  const level_t parent = child.parent();
  const auto &coarseleveldata = patchdata.level(parent);
  const auto &fgeom = patchdata.amrcore->Geom(child.index());
  const auto &cgeom = patchdata.amrcore->Geom(parent.index());
  const rk_dense_output_t dense_output = registered_dense_output();
  const stage_range stages = registered_stages();
  assert(stages.contains(completed));
  const substep_t substep = substep_of_parent_step(child);
  const bool substep_done = completed == stages.last();
  const stage_t stage = substep_done ? stages.first() : next_index(completed);
  const CCTK_REAL xsi = CCTK_REAL(substep_done ? substep.end : substep.begin);
  const CCTK_REAL dtc = ext.get_global_cctkGH()->cctk_delta_time *
                        CCTK_REAL(coarseleveldata.delta_iteration);
  assert(std::isfinite(dtc) && dtc > 0);

  const window_t window = window_t::one_level(child);
  std::vector<write_scope_t> scopes;
  for (const gi_t gi : states) {
    scopes.push_back(open_write(window, {.gi = gi,
                                         .tl = rk_tl,
                                         .fills = make_valid_coarse_fine(),
                                         .keeps = ~make_valid_coarse_fine(),
                                         .event = event}));
  }

  task_manager tasks1;
  task_manager tasks2;
  task_manager tasks3;

  for (const gi_t gi : states) {
    const GroupData &groupdata = leveldata.group(gi);
    const GroupData &coarsegroupdata = coarseleveldata.group(gi);
    assert(coarsegroupdata.record.numvars == groupdata.record.numvars);
    // Only integrated groups have bands, and only for them does the
    // subcycling SYNC leave the coarse-fine ghosts to this function
    assert(group_is_integrated(gi));

    amrex::MultiFab &mfab = *groupdata.timelevel(rk_tl).mfab;

    // As in FillPatch_Prolongate: without ghosts there is nothing to fill
    if (mfab.nGrowVect().max() == 0)
      continue;

    std::optional<rk_bands_t> &bands = buffers_of(leveldata, gi).rk_bands;
    assert(bands_match_footprint(
        bands,
        rk_fill_fpinfo(mfab, interpolator_of(groupdata.record), fgeom, cgeom)));
    if (!bands)
      continue;
    const rk_bands_t::sources_t src =
        bands->sources(coarseleveldata.iteration, {child, gi});
    amrex::MultiFab &crse_patch = bands->crse_patch();
    amrex::MultiFab &fine_patch = bands->fine_patch();

    tasks1.submit_serially([&crse_patch, src, dense_output, stage, xsi, dtc]() {
      rk_dense_output(crse_patch, src, dense_output, stage, xsi, dtc);
    });

    tasks2.submit_serially([&groupdata, &coarsegroupdata, &mfab, &crse_patch,
                            &fine_patch, &fgeom, &cgeom]() {
      Prolongate_Start(groupdata, coarsegroupdata, mfab, crse_patch, fine_patch,
                       fgeom, cgeom);
    });
    tasks3.submit_serially([&mfab]() { Prolongate_Finish(mfab); });
  }

  tasks1.run_tasks_serially();
  synchronize();
  tasks2.run_tasks_serially();
  synchronize();
  tasks3.run_tasks_serially();
  synchronize();
}

std::vector<checkpoint_band_t>
checkpoint_bands(const level_t level, const gi_t gi, const rat64 parent_step) {
  const std::optional<level_t> child = level.child();
  if (!child)
    return {};
  group_buffers_t &childbuffers = buffers_of(driver().level(*child), gi);
  std::optional<rk_bands_t> &rk_bands = childbuffers.rk_bands;
  return list_bands(childbuffers,
                    rk_bands ? rk_bands->stored(parent_step) : std::nullopt);
}

std::vector<checkpoint_band_t> restore_bands(const level_t level, const gi_t gi,
                                             const rat64 parent_step) {
  const std::optional<level_t> child = level.child();
  if (!child)
    return {};
  group_buffers_t &childbuffers = buffers_of(driver().level(*child), gi);
  std::optional<rk_bands_t> &rk_bands = childbuffers.rk_bands;
  return list_bands(childbuffers, rk_bands ? std::optional<rk_bands_t::slots_t>(
                                                 rk_bands->restore(parent_step))
                                           : std::nullopt);
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
