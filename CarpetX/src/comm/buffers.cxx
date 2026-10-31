#include "comm/buffers.hxx"

#include "hierarchy/driver.hxx"
#include <carpetx.hxx>

#include <cctk.h>

#include <cassert>
#include <memory>

namespace CarpetX {

void level_buffers_deleter_t::operator()(level_buffers_t *const buffers) const {
  delete buffers;
}

std::unique_ptr<level_buffers_t, level_buffers_deleter_t> make_level_buffers() {
  std::unique_ptr<level_buffers_t, level_buffers_deleter_t> buffers(
      new level_buffers_t);
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
