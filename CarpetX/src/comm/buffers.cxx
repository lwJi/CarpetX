#include "comm/buffers.hxx"
#include "comm/buffers_inside.hxx"

#include "comm/integrator_inside.hxx"
#include "comm/prolongate_3d_rf2_inside.hxx"
#include "comm/reflux_inside.hxx"
#include "comm/rk_band_placement_inside.hxx"
#include "comm/rk_bands_inside.hxx"
#include "hierarchy/model.hxx"
#include <carpetx.hxx>

#include <cctk.h>
#include <cctk_Parameters.h>

#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_FabArrayBase.H>
#include <AMReX_FluxRegister.H>
#include <AMReX_Geometry.H>
#include <AMReX_IndexType.H>
#include <AMReX_IntVect.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Periodicity.H>

#include <array>
#include <cassert>
#include <memory>
#include <optional>
#include <source_location>
#include <utility>

namespace CarpetX {

flux_buffers_t::flux_buffers_t(const amrex::BoxArray &ba,
                               const amrex::DistributionMapping &dm,
                               const amrex::IntVect &ratio,
                               const int fine_level, const int numvars)
    : reg(ba, dm, ratio, fine_level, numvars),
      scratch(ba, dm, ratio, fine_level, numvars) {
  reg.setVal(0);
  scratch.setVal(0);
}

namespace {

bool rk_band_locality_is_reported() {
  DECLARE_CCTK_PARAMETERS;
  return report_rk_band_locality;
}

std::optional<rk_buffers_t>
rk_buffers_of(const GHExt::GroupRecord &record, const level_t level,
              const amrex::BoxArray &ba, const amrex::DistributionMapping &dm,
              const bool count_locality, rk_band_locality_t &level_locality) {
  const GHExt &ext = driver();
  const gi_t gi = record.groupindex;
  if (!ext.use_subcycling || level.index() == 0 || !group_is_integrated(gi))
    return std::nullopt;
  const amrex::IntVect nghost(record.grid_function().nghostzones);
  if (nghost.max() == 0)
    return std::nullopt;
  const std::array<int, dim> &indextype = record.grid_function().indextype;
  const amrex::IndexType group_indextype(
      indextype[0] ? amrex::IndexType::CELL : amrex::IndexType::NODE,
      indextype[1] ? amrex::IndexType::CELL : amrex::IndexType::NODE,
      indextype[2] ? amrex::IndexType::CELL : amrex::IndexType::NODE);
  const amrex::BoxArray gba = convert(ba, group_indextype);
  const amrex::MultiFab probe(gba, dm, record.numvars, nghost,
                              amrex::MFInfo().SetAlloc(false));
  const amrex::AmrCore &amrcore = ext.patch(level.patch()).amrcore();
  const int parent = level.parent().index();
  const amrex::FabArrayBase::FPinfo &fp =
      rk_fill_fpinfo(probe, interpolator_of(record),
                     amrcore.Geom(level.index()), amrcore.Geom(parent));
  if (fp.ba_crse_patch.empty())
    return std::nullopt;
  const rk_band_layout_t layout = rk_band_layout_of(fp);
  if (count_locality)
    level_locality += rk_band_locality_of(
        layout, record.numvars, gba, dm, nghost,
        convert(amrcore.boxArray(parent), group_indextype),
        amrcore.DistributionMap(parent), amrcore.Geom(parent).periodicity());
  return std::optional<rk_buffers_t>(std::in_place, layout, record.numvars,
                                     registered_stages());
}

std::optional<amrex::MultiFab>
cell_volume_of(const level_t level, const amrex::BoxArray &ba,
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
  std::optional<amrex::MultiFab> cell_volume(std::in_place);
  geom.GetVolume(*cell_volume, ba, dm, 0);
  return cell_volume;
}
} // namespace

level_buffers_ptr make_level_buffers(const level_t level,
                                     const amrex::BoxArray &ba,
                                     const amrex::DistributionMapping &dm) {
  std::unique_ptr<level_buffers_t> buffers =
      std::make_unique<level_buffers_t>();
  const int numgroups = CCTK_NumGroups();
  buffers->groups.resize(numgroups);
  const bool count_locality = rk_band_locality_is_reported();
  rk_band_locality_t level_locality;
  bool level_has_bands = false;
  for (const gi_t gi : index_range<gi_t>(gi_t{0}, gi_t{numgroups})) {
    const GHExt::GroupRecord &record = driver().record(gi);
    if (record.kind != group_kind::grid_function)
      continue;
    if (!interpolator_of(record))
      CCTK_VERROR("Group %s has no prolongation operator",
                  record.groupname.c_str());
    group_buffers_t &group = buffers->groups.at(gi.index());
    if (level.index() > 0 && group_has_flux_register(gi))
      group.flux.emplace(std::in_place, ba, dm,
                         driver()
                             .patch(level.patch())
                             .amrcore()
                             .refRatio(level.parent().index()),
                         level.index(), record.numvars);
    group.rk =
        rk_buffers_of(record, level, ba, dm, count_locality, level_locality);
    level_has_bands = level_has_bands || group.rk;
  }
  if (count_locality && level_has_bands)
    report_rk_band_locality(level, current_iteration(), level_locality);
  buffers->cell_volume = cell_volume_of(level, ba, dm);
  return level_buffers_ptr(buffers.release(),
                           [](const level_buffers_t *const p) { delete p; });
}

const level_buffers_t &
buffers_of(const GHExt::PatchData::LevelData &leveldata) {
  assert(leveldata.buffers);
  return *leveldata.buffers;
}

const group_buffers_t &buffers_of(const GHExt::PatchData::LevelData &leveldata,
                                  const gi_t gi,
                                  const std::source_location site) {
  leveldata.group(gi, site);
  return buffers_of(leveldata).groups.at(gi.index());
}

} // namespace CarpetX
