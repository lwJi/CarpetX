#ifndef CARPETX_CARPETX_COMM_LEVEL_RUNS_INSIDE_HXX
#define CARPETX_CARPETX_COMM_LEVEL_RUNS_INSIDE_HXX

#include "internal.hxx"

#include "hierarchy/model.hxx"
#include <carpetx.hxx>

#include <functional>
#include <vector>

namespace CarpetX {

struct level_run_t {
  window_t levels;
  bool value;
};

std::vector<level_run_t> split_levels(
    window_t window,
    const std::function<bool(const GHExt::PatchData::LevelData &)> &predicate);

} // namespace CarpetX

#endif
