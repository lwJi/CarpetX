#ifndef CARPETX_CARPETX_API_HXX
#define CARPETX_CARPETX_API_HXX

#ifndef CARPETX_INTERNAL
#error                                                                         \
    "CarpetX internal header; include <carpetx.hxx> instead (see agent_docs/carpetx-public-api.md)"
#endif

#include <carpetx.hxx>

#include "driver.hxx"
#include "schedule.hxx"

#include <vector>

namespace CarpetX {

GHExt &driver();
GHExt::PatchData &resolve_patch(int patch);
GHExt::PatchData::LevelData &resolve(level_t level);
GHExt::PatchData::LevelData::GroupData &resolve(const char *what,
                                                group_t group);
void require_no_open_write(
    const char *what, const GHExt::PatchData::LevelData::GroupData &groupdata);

active_levels_t require_operable(const char *what, window_t window,
                                 const std::vector<int> &groups);

} // namespace CarpetX

#endif
