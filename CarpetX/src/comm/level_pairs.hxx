#ifndef CARPETX_CARPETX_COMM_LEVEL_PAIRS_HXX
#define CARPETX_CARPETX_COMM_LEVEL_PAIRS_HXX

#include "internal.hxx"

#include "hierarchy/driver.hxx"
#include "hierarchy/loops.hxx"
#include <carpetx.hxx>

#include <functional>

namespace CarpetX {

inline void
loop_pairs_fine_to_coarse(const window_t window,
                          const std::function<void(level_t level)> &kernel) {
  loop_fine_to_coarse(
      window, [&](const GHExt::PatchData::LevelData &leveldata) {
        if (leveldata.level.index() + 1 < window.level_bounds().max)
          kernel(leveldata.level);
      });
}

} // namespace CarpetX

#endif
