#ifndef CARPETX_CARPETX_API_HXX
#define CARPETX_CARPETX_API_HXX

#include "internal.hxx"

#include <carpetx.hxx>

#include "driver.hxx"
#include "schedule.hxx"

#include <span>

namespace CarpetX {

void require_no_open_write(
    const GHExt::PatchData::LevelData::GroupData &groupdata);

active_levels_t require_operable(window_t window, std::span<const gi_t> groups);
active_levels_t one_level(level_t level);

} // namespace CarpetX

#endif
