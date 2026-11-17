#include "comm/buffers.hxx"

#include "comm/integrator.hxx"
#include "comm/prolongate_3d_rf2.hxx"
#include "comm/reflux.hxx"
#include "comm/rk_bands.hxx"
#include "hierarchy/model.hxx"
#include <carpetx.hxx>

#include <cctk.h>

#include <AMReX_FabArrayBase.H>
#include <AMReX_FluxRegister.H>
#include <AMReX_Geometry.H>
#include <AMReX_IntVect.H>
#include <AMReX_MultiFab.H>

#include <array>
#include <cassert>
#include <memory>
#include <optional>
#include <source_location>
#include <utility>

namespace CarpetX {

namespace {
std::unique_ptr<amrex::FluxRegister>
flux_register_of(const GHExt::GroupRecord &record, const level_t level,
                 const amrex::BoxArray &ba,
                 const amrex::DistributionMapping &dm) {
  if (level.index() == 0 || !group_has_flux_register(record.groupindex))
    return nullptr;
  const amrex::IntVect ratio =
      driver().patch(level.patch()).amrcore().refRatio(level.parent().index());
  auto freg = std::make_unique<amrex::FluxRegister>(
      ba, dm, ratio, level.index(), record.numvars);
  freg->setVal(0);
  return freg;
}

void make_rk_buffers(group_buffers_t &group, const GHExt::GroupRecord &record,
                     const level_t level, const amrex::BoxArray &ba,
                     const amrex::DistributionMapping &dm) {
  const GHExt &ext = driver();
  const gi_t gi = record.groupindex;
  if (!ext.use_subcycling || level.index() == 0 || !group_is_integrated(gi))
    return;
  const amrex::IntVect nghost(record.grid_function().nghostzones);
  if (nghost.max() == 0)
    return;
  const std::array<int, dim> &indextype = record.grid_function().indextype;
  const amrex::BoxArray gba = convert(
      ba, amrex::IndexType(
              indextype[0] ? amrex::IndexType::CELL : amrex::IndexType::NODE,
              indextype[1] ? amrex::IndexType::CELL : amrex::IndexType::NODE,
              indextype[2] ? amrex::IndexType::CELL : amrex::IndexType::NODE));
  const amrex::MultiFab probe(gba, dm, record.numvars, nghost,
                              amrex::MFInfo().SetAlloc(false));
  const amrex::AmrCore &amrcore = ext.patch(level.patch()).amrcore();
  const amrex::FabArrayBase::FPinfo &fp = rk_fill_fpinfo(
      probe, interpolator_of(record), amrcore.Geom(level.index()),
      amrcore.Geom(level.parent().index()));
  if (fp.ba_crse_patch.empty())
    return;
  group.rk_sources.emplace(fp, record.numvars, registered_stages());
  group.rk_fill_patches.emplace(std::in_place, fp, record.numvars);
}

std::optional<flux_geometry_t>
flux_geometry_of(const level_t level, const amrex::BoxArray &ba,
                 const amrex::DistributionMapping &dm) {
  const GHExt &ext = driver();
  if (!ext.do_reflux)
    return std::nullopt;
  bool any_flux_register = false;
  for (const gi_t gi : index_range<gi_t>(gi_t{0}, gi_t{CCTK_NumGroups()}))
    any_flux_register = any_flux_register || group_has_flux_register(gi);
  if (!any_flux_register)
    return std::nullopt;
  assert(ba.ixType().cellCentered());
  const amrex::Geometry &geom =
      ext.patch(level.patch()).amrcore().Geom(level.index());
  std::optional<flux_geometry_t> geometry(std::in_place);
  for (int d = 0; d < dim; ++d)
    geom.GetFaceArea(geometry->face_area[d], ba, dm, d, 0);
  geom.GetVolume(geometry->cell_volume, ba, dm, 0);
  return geometry;
}
} // namespace

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
    group.registers = sealed_flux_registers_t(
        flux_registers_t{flux_register_of(record, level, ba, dm)});
    make_rk_buffers(group, record, level, ba, dm);
  }
  buffers->geometry = flux_geometry_of(level, ba, dm);
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
