#include "comm/rk_bands_inside.hxx"

#include "comm/prolongate_3d_rf2_inside.hxx"
#include "comm/rk_band_placement_inside.hxx"
#include "config/stages.hxx"
#include "util/refuse.hxx"
#include <carpetx.hxx>

#include <cctk.h>

#include <AMReX_FabArrayBase.H>
#include <AMReX_Geometry.H>
#include <AMReX_IntVect.H>
#include <AMReX_Interpolater.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Periodicity.H>
#include <AMReX_Vector.H>

#include <cassert>
#include <cstddef>
#include <optional>
#include <source_location>
#include <span>
#include <string>
#include <vector>

namespace CarpetX {

namespace {
struct band_copy_t {
  amrex::MultiFab &band;
  const amrex::MultiFab &src;
};

[[maybe_unused]] bool
each_band_once(const std::span<const band_copy_t> copies) {
  for (std::size_t i = 0; i < copies.size(); ++i)
    for (std::size_t j = 0; j < i; ++j)
      if (&copies[i].band == &copies[j].band)
        return false;
  return true;
}

void copy_interiors_to_bands(const std::span<const band_copy_t> copies,
                             const amrex::Periodicity &period) {
  assert(each_band_once(copies));
  for (const band_copy_t &copy : copies) {
    assert(copy.band.ixType() == copy.src.ixType());
    assert(copy.band.nComp() == copy.src.nComp());
    copy.band.ParallelCopy_nowait(copy.src, 0, 0, copy.band.nComp(),
                                  amrex::IntVect{0}, amrex::IntVect{0}, period);
  }
  for (const band_copy_t &copy : copies)
    copy.band.ParallelCopy_finish();
}

std::string describe_parent_step(const rat64 parent_step) {
  if (parent_step.den == 1)
    return std::to_string(parent_step.num);
  return std::to_string(parent_step.num) + "/" +
         std::to_string(parent_step.den);
}

std::string describe_held(const std::optional<rat64> &parent_step,
                          const int stages_stored, const int num_stages) {
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

void rk_sources_t::store_old(const std::span<const store_t> stores,
                             const amrex::Periodicity &period,
                             const rat64 parent_step) {
  std::vector<band_copy_t> copies;
  copies.reserve(stores.size());
  for (const store_t &store : stores)
    copies.push_back({store.bands.old_source_, store.src});
  copy_interiors_to_bands(copies, period);
  for (const store_t &store : stores) {
    store.bands.parent_step_ = parent_step;
    store.bands.stages_stored_ = 0;
  }
}

void rk_sources_t::store_stage(const std::span<const store_t> stores,
                               const stage_t stage,
                               const amrex::Periodicity &period,
                               const std::source_location site) {
  std::vector<band_copy_t> copies;
  copies.reserve(stores.size());
  for (const store_t &store : stores)
    copies.push_back({store.bands.next_stage_band(stage, site), store.src});
  copy_interiors_to_bands(copies, period);
  for (const store_t &store : stores)
    ++store.bands.stages_stored_;
}

amrex::MultiFab &
rk_sources_t::next_stage_band(const stage_t stage,
                              const std::source_location site) {
  const int num_stages = int(ks_source_.size());
  const stage_range stages = rk_stages(num_stages);
  const bool is_next_stage = parent_step_ && stages_stored_ < num_stages &&
                             stages.contains(stage) &&
                             stages.offset(stage) == stages_stored_;
  if (!is_next_stage)
    refuse(site,
           "the RK bands hold %s; RK stage %d is not the next stage to store",
           describe_held(parent_step_, stages_stored_, num_stages).c_str(),
           stage.index());
  return ks_source_.at(stages.offset(stage));
}

bool rk_sources_t::holds(const rat64 parent_step) const {
  return parent_step_ && *parent_step_ == parent_step &&
         stages_stored_ == int(ks_source_.size());
}

std::optional<rk_sources_t::sources_t>
rk_sources_t::stored(const rat64 parent_step) const {
  if (!holds(parent_step))
    return std::nullopt;
  return sources_t{old_source_, std::span<const amrex::MultiFab>(ks_source_)};
}

rk_sources_t::slots_t rk_sources_t::restore(const rat64 parent_step) {
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
           describe_held(parent_step_, stages_stored_, num_stages).c_str(),
           num_stages, describe_parent_step(parent_step).c_str());
  return {old_source_, std::span<const amrex::MultiFab>(ks_source_)};
}

rk_fill_patches_t::rk_fill_patches_t(const rk_band_layout_t &layout,
                                     const int numvars)
    : crse(layout.crse, layout.owner, numvars, 0, amrex::MFInfo(),
           layout.crse_factory),
      overlaps(amrex::Vector<ghost_overlap_t>(layout.overlaps.begin(),
                                              layout.overlaps.end())),
      footprint(layout.footprint) {
  assert(!layout.crse.empty());
}

} // namespace CarpetX
