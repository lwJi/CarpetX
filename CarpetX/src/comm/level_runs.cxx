#include "comm/level_runs_inside.hxx"

#include "hierarchy/model.hxx"
#include "hierarchy/loops.hxx"
#include <carpetx.hxx>

#include <cassert>
#include <functional>
#include <optional>
#include <vector>

namespace CarpetX {

std::vector<level_run_t> split_levels(
    const window_t window,
    const std::function<bool(const GHExt::PatchData::LevelData &)> &predicate) {
  const auto levels = window.level_bounds();
  std::vector<std::optional<bool> > values(levels.max - levels.min);
  loop_serially(window, [&](auto &restrict leveldata) {
    std::optional<bool> &value =
        values.at(leveldata.level.index() - levels.min);
    const bool level_value = predicate(leveldata);
    assert(!value || *value == level_value);
    value = level_value;
  });
  std::vector<level_run_t> runs;
  for (int lo = levels.min; lo < levels.max;) {
    const std::optional<bool> value = values.at(lo - levels.min);
    int hi = lo + 1;
    while (hi < levels.max && values.at(hi - levels.min) == value)
      ++hi;
    if (value)
      runs.push_back({window.narrow(lo, hi), *value});
    lo = hi;
  }
  return runs;
}

} // namespace CarpetX
