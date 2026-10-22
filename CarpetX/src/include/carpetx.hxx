#ifndef CARPETX_CARPETX_INCLUDE_CARPETX_HXX
#define CARPETX_CARPETX_INCLUDE_CARPETX_HXX

#include <loop.hxx>
#include <rational.hxx>

#include <cctk.h>

#include <AMReX_Geometry.H>

#include <array>
#include <cstdint>
#include <functional>
#include <ostream>
#include <vector>

namespace CarpetX {

using rat64 = Arith::rational<std::int64_t>;

enum class symmetry_t {
  none,
  interpatch,
  periodic,
  reflection,
};
std::ostream &operator<<(std::ostream &os, symmetry_t symmetry);

struct level_t {
  int patch, level;
};

struct group_t {
  level_t level;
  int gi;
};

int num_patches();
int num_levels(int patch);

level_t parent(level_t level);
rat64 iteration(level_t level);
rat64 delta_iteration(level_t level);
const amrex::Geometry &geometry(level_t level);
std::array<std::array<symmetry_t, Loop::dim>, 2> symmetries(level_t level);

struct window_t {
  int min_level, max_level;
  int min_patch, max_patch;

  void loop_serially(const std::function<void(level_t level)> &kernel) const;
  void
  loop_coarse_to_fine(const std::function<void(level_t level)> &kernel) const;
  void loop_parallel(
      const std::function<void(level_t level, int index, int component,
                               const cGH *cctkGH)> &kernel) const;
};

window_t active_window();
window_t all_levels();

void restrict_no_poison(window_t window, const std::vector<int> &groups);
void synchronize();

} // namespace CarpetX

#endif
