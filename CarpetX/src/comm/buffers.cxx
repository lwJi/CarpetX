#include "comm/buffers.hxx"

#include "comm/prolongate_3d_rf2.hxx"
#include "hierarchy/driver.hxx"
#include "hierarchy/group_tags.hxx"
#include <carpetx.hxx>

#include <cctk.h>

#include <AMReX_FluxRegister.H>

#include <cassert>
#include <memory>
#include <source_location>

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
    buffers->groups.at(gi.index()).freg =
        flux_register_of(record, level, ba, dm);
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
