#ifndef CARPETX_CARPETX_HIERARCHY_LOOPS_HXX
#define CARPETX_CARPETX_HIERARCHY_LOOPS_HXX

#include "internal.hxx"

#include "hierarchy/model.hxx"

#include <carpetx.hxx>

#include <cctk.h>

#include <functional>

namespace CarpetX {

void loop_coarse_to_fine(
    window_t window,
    const std::function<void(const GHExt::PatchData::LevelData &leveldata)>
        &kernel);

void loop_fine_to_coarse(
    window_t window,
    const std::function<void(const GHExt::PatchData::LevelData &leveldata)>
        &kernel);

inline void loop_serially(
    const window_t window,
    const std::function<void(const GHExt::PatchData::LevelData &leveldata)>
        &kernel) {
  loop_coarse_to_fine(window, kernel);
}

void loop_parallel(
    window_t window,
    const std::function<void(const GHExt::PatchData::LevelData &leveldata,
                             tile_t tile, const cGH *cctkGH)> &kernel);

void loop_serially(
    window_t window,
    const std::function<void(const GHExt::PatchData::LevelData &leveldata,
                             tile_t tile, const cGH *cctkGH)> &kernel);

} // namespace CarpetX

#endif
