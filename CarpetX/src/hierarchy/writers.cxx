#include "hierarchy/writers.hxx"
#include "config/stages.hxx"
#include "hierarchy/binding.hxx"
#include "hierarchy/clock.hxx"
#include "hierarchy/loops.hxx"
#include "hierarchy/model.hxx"
#include "hierarchy/modes.hxx"
#include "hierarchy/scopes.hxx"

#include <carpetx.hxx>

#include <cctk.h>

#include <AMReX_AmrMesh.H>
#include <AMReX_BoxArray.H>
#include <AMReX_BoxList.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParallelDescriptor.H>

#include <cassert>
#include <cmath>
#include <cstddef>
#include <functional>
#include <memory>
#include <optional>
#include <source_location>
#include <string>
#include <utility>

namespace CarpetX {

std::unique_ptr<hierarchy_writers_t>
create_hierarchy(const GHExt::config_t config) {
  return std::unique_ptr<hierarchy_writers_t>(new hierarchy_writers_t(config));
}

hierarchy_writers_t::hierarchy_writers_t(const GHExt::config_t config)
    : hierarchy(config), clock(hierarchy), stage(hierarchy),
      structure(hierarchy), registry(hierarchy), scratch(hierarchy) {}

hierarchy_writers_t::~hierarchy_writers_t() {
  cctkGH_binding::unbind_arrays(hierarchy.flesh_cctkGH());
}

const GHExt::PatchData &
structure_writer_t::add_patch(const patch_t patch,
                              const faces_t<symmetry_t> &symmetries,
                              const faces_t<boundary_t> &default_boundaries) {
  return ext.sealed_patches().open(*this).add_patch(
      ext, levels, patch, symmetries, default_boundaries);
}

GHExt::PatchData::levels_t &level_writer_t::changing(const patch_t patch) {
  ++ext.sealed_level_set().open(*this).epoch;
  return ext.patch(patch).sealed_levels().open(*this);
}

void level_writer_t::ensure_level_cctkGH(const level_t level) {
  auto &level_set = ext.sealed_level_set().open(*this);
  if (int(level_set.level_cctkGHs.size()) > level.index())
    return;
  cGH *const cctkGH = ext.get_global_cctkGH();
  enter_level_mode(cctkGH, level);
  assert(int(level_set.level_cctkGHs.size()) == level.index());
  level_set.level_cctkGHs.push_back(GHExt::cctkGHptr::copy_of(cctkGH));
  leave_level_mode(cctkGH, level);
}

void level_writer_t::reset_modified(const patch_t patch) {
  const GHExt::PatchData &patchdata = ext.patch(patch);
  assert(patchdata.num_levels() == patchdata.amrcore().finestLevel() + 1);
  patchdata.sealed_levels().open(*this).reset_level_modified();
}

const GHExt::PatchData::LevelData &
level_writer_t::add_level(const level_t level, const amrex::BoxArray &ba,
                          const amrex::DistributionMapping &dm,
                          const std::function<std::string()> &why) {
  if (ext.steered().verbose)
#pragma omp critical
    CCTK_VINFO("SetupLevel patch %d level %d", level.patch().index(),
               level.index());

  ensure_level_cctkGH(level);
  const GHExt::PatchData::LevelData &leveldata =
      changing(level.patch()).add_level(level, ba, dm, why);

  if (ext.steered().verbose)
#pragma omp critical
    CCTK_VINFO("SetupLevel patch %d level %d done.", level.patch().index(),
               level.index());

  return leveldata;
}

const GHExt::PatchData::LevelData &
level_writer_t::make_level(const level_t level, const amrex::BoxArray &ba,
                           const amrex::DistributionMapping &dm,
                           const std::string &why) {
  const GHExt::PatchData::LevelData &leveldata =
      add_level(level, ba, dm, [why]() { return why; });

  const window_t window = window_t::one_level(level);
  for (const auto &groupdata : leveldata.groups()) {
    for (const tl_t tl : groupdata.record.timelevels())
      invalidate(window, {.gi = groupdata.record.groupindex,
                          .tl = tl,
                          .where = make_valid_all(),
                          .event = why});
  }
  return leveldata;
}

GHExt::PatchData::LevelData
level_writer_t::remake_level(const level_t level, const amrex::BoxArray &ba,
                             const amrex::DistributionMapping &dm,
                             const std::function<std::string()> &why) {
  GHExt::PatchData::LevelData fresh(level, ba, dm, why);
  return changing(level.patch()).replace_level(level, std::move(fresh));
}

void level_writer_t::erase_levels_from(const level_t level) {
  changing(level.patch()).erase_levels_from(level);
}

void structure_writer_t::set_max_grid_size(
    const patch_t patch, const amrex::Vector<amrex::IntVect> &max_grid_sizes) {
  ext.patch(patch).sealed_core().open(*this).amrcore_->SetMaxGridSize(
      max_grid_sizes);
}

void structure_writer_t::make_new_grids(const patch_t patch,
                                        const CCTK_REAL time) {
  ext.patch(patch).sealed_core().open(*this).amrcore_->MakeNewGrids(time);
}

void structure_writer_t::mark_initialized() {
  ext.sealed_patches().open(*this).mark_initialized();
}

void structure_writer_t::regrid(const level_t base_level,
                                const CCTK_REAL time) {
  levels.reset_modified(base_level.patch());
  ext.patch(base_level.patch())
      .sealed_core()
      .open(*this)
      .amrcore_->regrid(base_level.index(), time);
}

void structure_writer_t::set_finest_level(const level_t finest_level) {
  ext.patch(finest_level.patch())
      .sealed_core()
      .open(*this)
      .amrcore_->SetFinestLevel(finest_level.index());
}

static amrex::BoxArray rechop_level(const amrex::AmrMesh &mesh,
                                    const level_t recovered_level,
                                    amrex::BoxArray ba) {
  if (recovered_level.index() == 0)
    return mesh.MakeBaseGrids();

  amrex::BoxList bl = ba.boxList();
  while (bl.simplify())
    ;
  ba = amrex::BoxArray(std::move(bl));
  ba.maxSize(mesh.maxGridSize(recovered_level.index()));
  mesh.ChopGrids(recovered_level.index(), ba,
                 amrex::ParallelDescriptor::NProcs());
  return ba;
}

void structure_writer_t::recover_level(const level_t level, amrex::BoxArray ba,
                                       const bool rechop) {
  amrex::AmrCore &amrcore =
      *ext.patch(level.patch()).sealed_core().open(*this).amrcore_;
  const int lev = level.index();
  if (lev > 0) {
    amrex::Geometry geom = amrcore.Geom(lev - 1);
    geom.refine({2, 2, 2});
    amrcore.SetGeometry(lev, geom);
  }
  if (rechop)
    ba = rechop_level(amrcore, level, ba);
  amrcore.SetBoxArray(lev, ba);
  const amrex::DistributionMapping dm(ba);
  amrcore.SetDistributionMap(lev, dm);
  levels.make_level(level, ba, dm, "Recovering");
}

void registry_writer_t::record_storage(const gi_t gi, const int timelevels) {
  ext.sealed_registry().open(*this).record_storage(gi, timelevels);
}

void registry_writer_t::freeze_storage(
    std::vector<group_declaration_t> declarations) {
  ext.sealed_registry().open(*this).freeze_storage(std::move(declarations));
}

void registry_writer_t::record_integrator(GHExt::integrator_t integrator,
                                          const std::source_location site) {
  ext.sealed_registry().open(*this).record_integrator(std::move(integrator),
                                                      site);
}

void registry_writer_t::freeze_integrator(const std::source_location site) {
  ext.sealed_registry().open(*this).freeze_integrator(site);
}

void registry_writer_t::adopt_flesh_cctkGH(cGH *const cctkGH) {
  auto &registry = ext.sealed_registry().open(*this);
  assert(cctkGH);
  assert(!registry.flesh_cctkGH());
  assert(!registry.global_cctkGH());
  registry.adopt_flesh_cctkGH(cctkGH);
  cctkGH_binding::bind_arrays(cctkGH);
  registry.adopt_global_cctkGH(GHExt::cctkGHptr::copy_of(cctkGH));
}

std::size_t scratch_writer_t::acquire(const group_t group,
                                      const std::source_location site) {
  const auto &groupdata = ext.group(group, site);
  auto &pool = groupdata.sealed_scratch_pool().open(*this).slots;
  std::size_t slot = 0;
  while (slot < pool.size() && pool.at(slot).in_use)
    ++slot;
  if (slot == pool.size()) {
    const amrex::MultiFab &mfab0 = groupdata.timelevel(tl_t{0}).mfab();
    pool.push_back({std::make_unique<amrex::MultiFab>(
                        mfab0.boxArray(), mfab0.DistributionMap(),
                        mfab0.nComp(), mfab0.nGrowVect()),
                    false});
  }
  pool.at(slot).in_use = true;
  return slot;
}

void scratch_writer_t::release(const group_t group, const std::size_t slot) {
  auto &pool = ext.group(group).sealed_scratch_pool().open(*this).slots;
  assert(slot < pool.size() && pool.at(slot).in_use);
  pool.at(slot).in_use = false;
}

amrex::MultiFab &scratch_writer_t::mfab(const group_t group,
                                        const std::size_t slot,
                                        const std::source_location site) {
  auto &pool = ext.group(group, site).sealed_scratch_pool().open(*this).slots;
  assert(slot < pool.size() && pool.at(slot).in_use);
  return *pool.at(slot).mfab;
}

namespace {
GHExt::run_clock_t showing_level(GHExt::run_clock_t clock, const GHExt &ext,
                                 const level_t level) {
  if (ext.fixed.use_subcycling) {
    clock.timefac = substeps_per_coarsest_step(level.index());
    clock.time = clock.initial_time +
                 clock.delta_time * double(ext.level(level).iteration());
  } else {
    clock.timefac = 1;
  }
  return clock;
}
} // namespace

void clock_writer_t::set(const GHExt::run_clock_t next) {
  cctkGH_clock::set_run(*this, ext, next);
}

void clock_writer_t::start_run(const CCTK_REAL initial_time) {
  set({.iteration = 0,
       .time = initial_time,
       .delta_time = NAN,
       .timefac = 1,
       .initial_time = initial_time});
}

void clock_writer_t::resume_run(const int iteration, const CCTK_REAL time) {
  GHExt::run_clock_t next = ext.run_clock();
  next.iteration = iteration;
  next.time = time;
  next.recovered_iteration = iteration;
  set(next);
}

void clock_writer_t::set_delta_time(const CCTK_REAL delta_time) {
  using std::isfinite;
  assert(isfinite(delta_time));
  GHExt::run_clock_t next = ext.run_clock();
  next.delta_time = delta_time;
  set(next);
}

void clock_writer_t::begin_iteration() {
  GHExt::run_clock_t next = ext.run_clock();
  ++next.iteration;
  set(next);
}

void clock_writer_t::begin_step(const window_t window) {
  loop_serially(window, [this](const GHExt::PatchData::LevelData &leveldata) {
    auto &clock = leveldata.sealed_clock().open(*this);
    clock.iteration += clock.delta_iteration;
  });
  GHExt::run_clock_t next = ext.run_clock();
  if (!ext.fixed.use_subcycling)
    next.time += next.delta_time;
  set(showing_level(next, ext, window.coarsest()));
}

void clock_writer_t::show_level(const level_t level) {
  set(showing_level(ext.run_clock(), ext, level));
}

void clock_writer_t::replay_last_step(const level_t level) {
  GHExt::run_clock_t next = ext.run_clock();
  next.iteration = ext.level(level).last_step_iteration();
  set(showing_level(next, ext, level));
}

void stage_writer_t::run_at(const CCTK_REAL time, const CCTK_REAL delta_time,
                            const std::function<void()> &body) {
  const std::optional<GHExt::stage_clock_t> outer = ext.shown_stage();
  cctkGH_clock::show_stage(*this, ext, GHExt::stage_clock_t{time, delta_time});
  body();
  cctkGH_clock::show_stage(*this, ext, outer);
}

void clock_writer_t::stamp_last_step(const window_t window) {
  const int iteration = ext.run_clock().iteration;
  loop_serially(window, [&](const GHExt::PatchData::LevelData &leveldata) {
    leveldata.sealed_clock().open(*this).last_step_iteration = iteration;
  });
}

void clock_writer_t::stamp_last_finalized(const window_t window) {
  const int iteration = ext.run_clock().iteration;
  loop_serially(window, [&](const GHExt::PatchData::LevelData &leveldata) {
    leveldata.sealed_clock().open(*this).last_finalized_iteration = iteration;
  });
}

void clock_writer_t::restore_level_clock(
    const level_t level, const rat64 iteration,
    const std::optional<int> last_step_iteration) {
  auto &clock = ext.level(level).sealed_clock().open(*this);
  clock.iteration = iteration;
  if (last_step_iteration)
    clock.last_step_iteration = *last_step_iteration;
}

void clock_writer_t::rotate_timelevels(const window_t window, const gi_t gi) {
  cctkGH_binding::rotate(ext, window, gi);
}

void clock_writer_t::rotate_array_timelevels(const gi_t gi) {
  cctkGH_binding::rotate(ext, gi);
}

} // namespace CarpetX
