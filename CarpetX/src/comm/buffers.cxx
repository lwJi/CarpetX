#include "comm/buffers.hxx"

#include "comm/prolongate_3d_rf2.hxx"
#include "hierarchy/driver.hxx"
#include "hierarchy/integrator.hxx"
#include "util/refuse.hxx"
#include <carpetx.hxx>

#include <cctk.h>

#include <AMReX_FabArrayBase.H>
#include <AMReX_FluxRegister.H>
#include <AMReX_Geometry.H>
#include <AMReX_IntVect.H>
#include <AMReX_Interpolater.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Periodicity.H>

#include <array>
#include <cassert>
#include <memory>
#include <optional>
#include <source_location>
#include <span>
#include <string>

namespace CarpetX {

namespace {
std::unique_ptr<amrex::FluxRegister>
flux_register_of(const GHExt::GroupRecord &record, const level_t level,
                 const amrex::BoxArray &ba,
                 const amrex::DistributionMapping &dm) {
  if (level.index() == 0 || !group_has_flux_register(record.groupindex))
    return nullptr;
  const amrex::IntVect ratio =
      driver().patch(level.patch()).amrcore->refRatio(level.parent().index());
  auto freg = std::make_unique<amrex::FluxRegister>(
      ba, dm, ratio, level.index(), record.numvars);
  freg->setVal(0);
  return freg;
}

std::optional<rk_bands_t> rk_bands_of(const GHExt::GroupRecord &record,
                                      const level_t level,
                                      const amrex::BoxArray &ba,
                                      const amrex::DistributionMapping &dm) {
  const GHExt &ext = driver();
  const gi_t gi = record.groupindex;
  if (!ext.use_subcycling || level.index() == 0 || !group_is_integrated(gi))
    return std::nullopt;
  const amrex::IntVect nghost(record.grid_function().nghostzones);
  if (nghost.max() == 0)
    return std::nullopt;
  const std::array<int, dim> &indextype = record.grid_function().indextype;
  const amrex::BoxArray gba = convert(
      ba, amrex::IndexType(
              indextype[0] ? amrex::IndexType::CELL : amrex::IndexType::NODE,
              indextype[1] ? amrex::IndexType::CELL : amrex::IndexType::NODE,
              indextype[2] ? amrex::IndexType::CELL : amrex::IndexType::NODE));
  const amrex::MultiFab probe(gba, dm, record.numvars, nghost,
                              amrex::MFInfo().SetAlloc(false));
  const auto &amrcore = ext.patch(level.patch()).amrcore;
  const amrex::FabArrayBase::FPinfo &fp = rk_fill_fpinfo(
      probe, interpolator_of(record), amrcore->Geom(level.index()),
      amrcore->Geom(level.parent().index()));
  if (fp.ba_crse_patch.empty())
    return std::nullopt;
  return std::optional<rk_bands_t>(std::in_place, fp, record.numvars,
                                   registered_stages());
}

void copy_interior_to_band(amrex::MultiFab &band, const amrex::MultiFab &src,
                           const amrex::Periodicity &period) {
  assert(band.ixType() == src.ixType());
  assert(band.nComp() == src.nComp());
  band.ParallelCopy(src, 0, 0, band.nComp(), amrex::IntVect{0},
                    amrex::IntVect{0}, period);
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

rk_bands_t::rk_bands_t(const amrex::FabArrayBase::FPinfo &fp, const int numvars,
                       const stage_range stages)
    : old_source_(fp.ba_crse_patch, fp.dm_patch, numvars, 0),
      crse_patch_(fp.ba_crse_patch, fp.dm_patch, numvars, 0, amrex::MFInfo(),
                  *fp.fact_crse_patch),
      fine_patch_(fp.ba_fine_patch, fp.dm_patch, numvars, 0, amrex::MFInfo(),
                  *fp.fact_fine_patch) {
  assert(!fp.ba_crse_patch.empty());
  ks_source_.reserve(stages.size());
  for ([[maybe_unused]] const stage_t stage : stages)
    ks_source_.emplace_back(fp.ba_crse_patch, fp.dm_patch, numvars, 0);
}

void rk_bands_t::store_old(const amrex::MultiFab &parent_state,
                           const amrex::Periodicity &period,
                           const rat64 parent_step) {
  copy_interior_to_band(old_source_, parent_state, period);
  parent_step_ = parent_step;
  stages_stored_ = 0;
}

void rk_bands_t::store_stage(const stage_t stage,
                             const amrex::MultiFab &parent_rhs,
                             const amrex::Periodicity &period,
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
  copy_interior_to_band(ks_source_.at(stages.offset(stage)), parent_rhs,
                        period);
  ++stages_stored_;
}

bool rk_bands_t::holds(const rat64 parent_step) const {
  return parent_step_ && *parent_step_ == parent_step &&
         stages_stored_ == int(ks_source_.size());
}

std::optional<rk_bands_t::slots_t> rk_bands_t::stored(const rat64 parent_step) {
  if (!holds(parent_step))
    return std::nullopt;
  return slots_t{old_source_, std::span<amrex::MultiFab>(ks_source_)};
}

rk_bands_t::slots_t rk_bands_t::restore(const rat64 parent_step) {
  parent_step_ = parent_step;
  stages_stored_ = int(ks_source_.size());
  return {old_source_, std::span<amrex::MultiFab>(ks_source_)};
}

rk_bands_t::sources_t rk_bands_t::sources(const rat64 parent_step,
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

amrex::MultiFab &rk_bands_t::crse_patch() { return crse_patch_; }
amrex::MultiFab &rk_bands_t::fine_patch() { return fine_patch_; }

level_buffers_ptr make_level_buffers(const level_t level,
                                     const amrex::BoxArray &ba,
                                     const amrex::DistributionMapping &dm) {
  level_buffers_ptr buffers(new level_buffers_t,
                            [](level_buffers_t *const p) { delete p; });
  const int numgroups = CCTK_NumGroups();
  buffers->groups.resize(numgroups);
  for (const gi_t gi : index_range<gi_t>(gi_t{0}, gi_t{numgroups})) {
    const GHExt::GroupRecord &record = driver().record(gi);
    if (record.kind != group_kind::grid_function)
      continue;
    if (!interpolator_of(record))
      CCTK_VERROR("Group %s has no prolongation operator",
                  record.groupname.c_str());
    group_buffers_t &group = buffers->groups.at(gi.index());
    group.freg = flux_register_of(record, level, ba, dm);
    group.rk_bands = rk_bands_of(record, level, ba, dm);
  }
  return buffers;
}

level_buffers_t &buffers_of(const GHExt::PatchData::LevelData &leveldata) {
  assert(leveldata.buffers);
  return *leveldata.buffers;
}

group_buffers_t &buffers_of(const GHExt::PatchData::LevelData &leveldata,
                            const gi_t gi, const std::source_location site) {
  leveldata.group(gi, site);
  return buffers_of(leveldata).groups.at(gi.index());
}

} // namespace CarpetX
