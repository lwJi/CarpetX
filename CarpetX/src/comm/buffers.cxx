#include "comm/buffers.hxx"

#include "hierarchy/driver.hxx"
#include <carpetx.hxx>

#include <cctk.h>

#include <cassert>
#include <memory>

namespace CarpetX {

level_buffers_ptr make_level_buffers() {
  level_buffers_ptr buffers(new level_buffers_t,
                            [](level_buffers_t *const p) { delete p; });
  buffers->groups.resize(CCTK_NumGroups());
  return buffers;
}

level_buffers_t &buffers_of(const GHExt::PatchData::LevelData &leveldata) {
  assert(leveldata.buffers);
  return *leveldata.buffers;
}

group_buffers_t &buffers_of(const GHExt::PatchData::LevelData &leveldata,
                            const gi_t gi) {
  return buffers_of(leveldata).groups.at(gi.index());
}

} // namespace CarpetX
