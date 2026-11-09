#include "comm/buffers.hxx"

#include "comm/prolongate_3d_rf2.hxx"
#include "hierarchy/driver.hxx"
#include "hierarchy/group_tags.hxx"
#include <carpetx.hxx>

#include <cctk.h>

#include <AMReX_BCRec.H>
#include <AMReX_FluxRegister.H>
#include <AMReX_Interpolater.H>

#include <array>
#include <cassert>
#include <memory>
#include <optional>
#include <source_location>

namespace CarpetX {

namespace {
amrex::IndexType index_type_of(const std::array<int, dim> &indextype) {
  return amrex::IndexType(
      indextype[0] ? amrex::IndexType::CELL : amrex::IndexType::NODE,
      indextype[1] ? amrex::IndexType::CELL : amrex::IndexType::NODE,
      indextype[2] ? amrex::IndexType::CELL : amrex::IndexType::NODE);
}

amrex::Vector<amrex::BCRec> bcrecs_of(const GHExt::GroupRecord &record,
                                      const level_t level) {
  const GHExt::PatchData &patchdata = driver().patch(level.patch());
  amrex::BCRec bcrec;
  for (int f = 0; f < 2; ++f) {
    for (int d = 0; d < dim; ++d) {
      const auto bc = patchdata.symmetries[f][d] == symmetry_t::periodic
                          ? amrex::BCType::int_dir
                          : amrex::BCType::ext_dir;
      if (f == 0)
        bcrec.setLo(d, bc);
      else
        bcrec.setHi(d, bc);
    }
  }
  return amrex::Vector<amrex::BCRec>(record.numvars, bcrec);
}

std::unique_ptr<amrex::FluxRegister>
flux_register_of(const GHExt::GroupRecord &record, const level_t level,
                 const std::array<int, dim> &indextype,
                 const amrex::BoxArray &ba,
                 const amrex::DistributionMapping &dm) {
  const gi_t gi = record.groupindex;
  if (!get_group_fluxes(gi))
    return nullptr;
  if (level.index() == 0 || !group_has_flux_register(gi))
    return nullptr;
  assert((indextype == std::array<int, dim>{1, 1, 1}));
  const amrex::IntVect ratio =
      driver().patch(level.patch()).amrcore->refRatio(level.parent().index());
  auto freg = std::make_unique<amrex::FluxRegister>(
      convert(ba, index_type_of(indextype)), dm, ratio, level.index(),
      record.numvars);
  freg->setVal(0);
  return freg;
}
} // namespace

group_buffers_t::group_buffers_t(const GHExt::GroupRecord &record,
                                 const level_t level, const amrex::BoxArray &ba,
                                 const amrex::DistributionMapping &dm)
    : interpolator(
          get_interpolator(get_group_prolongation_type(record.groupindex),
                           get_group_prolongation_order(record.groupindex),
                           get_group_indextype(record.groupindex))),
      bcrecs(bcrecs_of(record, level)),
      freg(flux_register_of(record, level,
                            get_group_indextype(record.groupindex), ba, dm)) {
  assert(record.kind == group_kind::grid_function);
}

level_buffers_ptr make_level_buffers(const level_t level,
                                     const amrex::BoxArray &ba,
                                     const amrex::DistributionMapping &dm) {
  level_buffers_ptr buffers(new level_buffers_t,
                            [](level_buffers_t *const p) { delete p; });
  const int numgroups = CCTK_NumGroups();
  buffers->groups.reserve(numgroups);
  for (const gi_t gi : index_range<gi_t>(gi_t{0}, gi_t{numgroups})) {
    const GHExt::GroupRecord &record = driver().record(gi);
    if (record.kind != group_kind::grid_function)
      buffers->groups.emplace_back(nullptr);
    else
      buffers->groups.push_back(
          std::make_unique<group_buffers_t>(record, level, ba, dm));
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
  const std::unique_ptr<group_buffers_t> &buffers =
      buffers_of(leveldata).groups.at(gi.index());
  assert(buffers);
  return *buffers;
}

} // namespace CarpetX
