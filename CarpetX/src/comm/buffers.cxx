#include "comm/buffers.hxx"

#include "comm/prolongate_3d_rf2.hxx"
#include "hierarchy/driver.hxx"
#include "hierarchy/group_tags.hxx"
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
  const std::array<int, dim> nghostzones = get_group_nghostzones(gi);
  const amrex::IntVect nghost(nghostzones);
  if (nghost.max() == 0)
    return std::nullopt;
  const std::array<int, dim> indextype = get_group_indextype(gi);
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
                           const amrex::Periodicity &period) {
  copy_interior_to_band(old_source_, parent_state, period);
}

void rk_bands_t::store_stage(const stage_t stage,
                             const amrex::MultiFab &parent_rhs,
                             const amrex::Periodicity &period) {
  const stage_range stages = rk_stages(int(ks_source_.size()));
  copy_interior_to_band(ks_source_.at(stages.offset(stage)), parent_rhs,
                        period);
}

rk_bands_t::sources_t rk_bands_t::sources() const {
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
