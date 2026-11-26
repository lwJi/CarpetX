#include "comm/rk_bands_inside.hxx"

#include "comm/rk_band_placement_inside.hxx"
#include "config/stages.hxx"
#include "util/refuse.hxx"
#include "util/task_manager.hxx"
#include <carpetx.hxx>

#include <cctk.h>

#include <AMReX_FabArrayBase.H>
#include <AMReX_Geometry.H>
#include <AMReX_IntVect.H>
#include <AMReX_Interpolater.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Periodicity.H>

#include <cassert>
#include <optional>
#include <source_location>
#include <span>
#include <string>

namespace CarpetX {

namespace {
void copy_interior_to_band(amrex::MultiFab &band, const amrex::MultiFab &src,
                           const amrex::Periodicity &period) {
  assert(band.ixType() == src.ixType());
  assert(band.nComp() == src.nComp());
  band.ParallelCopy(src, 0, 0, band.nComp(), amrex::IntVect{0},
                    amrex::IntVect{0}, period);
}

void post_interior_to_band(amrex::MultiFab &band, const amrex::MultiFab &src,
                           const amrex::Periodicity &period) {
  assert(band.ixType() == src.ixType());
  assert(band.nComp() == src.nComp());
  band.ParallelCopy_nowait(src, 0, 0, band.nComp(), amrex::IntVect{0},
                           amrex::IntVect{0}, period);
}

std::string describe_parent_step(const rat64 parent_step) {
  if (parent_step.den == 1)
    return std::to_string(parent_step.num);
  return std::to_string(parent_step.num) + "/" +
         std::to_string(parent_step.den);
}

std::string describe_held(const bool store_pending,
                          const std::optional<rat64> &parent_step,
                          const int stages_stored, const int num_stages) {
  if (store_pending)
    return "nothing while a store is pending";
  if (!parent_step)
    return "no parent step";
  return std::to_string(stages_stored) + " of " + std::to_string(num_stages) +
         " stages of parent step " + describe_parent_step(*parent_step);
}
} // namespace

const amrex::FabArrayBase::FPinfo &
rk_fill_fpinfo(const amrex::FabArrayBase &fine,
               amrex::Interpolater *const interpolator,
               const amrex::Geometry &fgeom, const amrex::Geometry &cgeom) {
  const amrex::IntVect ratio{2, 2, 2};
  const amrex::EB2::IndexSpace *const index_space = nullptr;
  const amrex::InterpolaterBoxCoarsener &coarsener =
      interpolator->BoxCoarsener(ratio);
  return amrex::FabArrayBase::TheFPinfo(fine, fine, fine.nGrowVect(), coarsener,
                                        fgeom, cgeom, index_space);
}

rk_sources_t::rk_sources_t(const rk_band_layout_t &layout, const int numvars,
                           const stage_range stages)
    : old_source_(layout.crse, layout.owner, numvars, 0) {
  assert(!layout.crse.empty());
  ks_source_.reserve(stages.size());
  for ([[maybe_unused]] const stage_t stage : stages)
    ks_source_.emplace_back(layout.crse, layout.owner, numvars, 0);
}

void rk_sources_t::store_old(const amrex::MultiFab &parent_state,
                             const amrex::Periodicity &period,
                             const rat64 parent_step) {
  const int num_stages = int(ks_source_.size());
  if (pending_)
    refuse(std::source_location::current(),
           "the RK bands hold %s; the old state of parent step %s cannot be "
           "stored before that store has finished",
           describe_held(pending_, parent_step_, stages_stored_, num_stages)
               .c_str(),
           describe_parent_step(parent_step).c_str());
  copy_interior_to_band(old_source_, parent_state, period);
  parent_step_ = parent_step;
  stages_stored_ = 0;
}

void rk_sources_t::store_stage(task_manager &finishing, const stage_t stage,
                               const amrex::MultiFab &parent_rhs,
                               const amrex::Periodicity &period,
                               const std::source_location site) {
  const int num_stages = int(ks_source_.size());
  const stage_range stages = rk_stages(num_stages);
  const bool is_next_stage =
      !pending_ && parent_step_ && stages_stored_ < num_stages &&
      stages.contains(stage) && stages.offset(stage) == stages_stored_;
  if (!is_next_stage)
    refuse(site,
           "the RK bands hold %s; RK stage %d is not the next stage to store",
           describe_held(pending_, parent_step_, stages_stored_, num_stages)
               .c_str(),
           stage.index());
  amrex::MultiFab &band = ks_source_.at(stages.offset(stage));
  post_interior_to_band(band, parent_rhs, period);
  pending_ = true;
  finishing.submit_serially([this, &band]() {
    band.ParallelCopy_finish();
    ++stages_stored_;
    pending_ = false;
  });
}

bool rk_sources_t::holds(const rat64 parent_step) const {
  return !pending_ && parent_step_ && *parent_step_ == parent_step &&
         stages_stored_ == int(ks_source_.size());
}

std::optional<rk_sources_t::sources_t>
rk_sources_t::stored(const rat64 parent_step) const {
  if (!holds(parent_step))
    return std::nullopt;
  return sources_t{old_source_, std::span<const amrex::MultiFab>(ks_source_)};
}

rk_sources_t::slots_t rk_sources_t::restore(const rat64 parent_step) {
  if (pending_)
    refuse(std::source_location::current(),
           "the RK bands hold %s; parent step %s cannot be restored before "
           "that store has finished",
           describe_held(pending_, parent_step_, stages_stored_,
                         int(ks_source_.size()))
               .c_str(),
           describe_parent_step(parent_step).c_str());
  parent_step_ = parent_step;
  stages_stored_ = int(ks_source_.size());
  return {old_source_, std::span<amrex::MultiFab>(ks_source_)};
}

rk_sources_t::sources_t rk_sources_t::sources(const rat64 parent_step,
                                              const fill_site_t &where) const {
  const int num_stages = int(ks_source_.size());
  if (!holds(parent_step))
    refuse(std::source_location::current(),
           "the RK bands of group %s on patch %d, refinement level %d hold "
           "%s; the fill needs all %d stages of parent step %s",
           CCTK_FullGroupName(where.gi.index()), where.level.patch().index(),
           where.level.index(),
           describe_held(pending_, parent_step_, stages_stored_, num_stages)
               .c_str(),
           num_stages, describe_parent_step(parent_step).c_str());
  return {old_source_, std::span<const amrex::MultiFab>(ks_source_)};
}

rk_fill_patches_t::rk_fill_patches_t(const rk_band_layout_t &layout,
                                     const int numvars)
    : crse(layout.crse, layout.owner, numvars, 0, amrex::MFInfo(),
           layout.crse_factory),
      fine(layout.fine, layout.owner, numvars, 0, amrex::MFInfo(),
           layout.fine_factory),
      footprint(layout.footprint) {
  assert(!layout.crse.empty());
}

} // namespace CarpetX
