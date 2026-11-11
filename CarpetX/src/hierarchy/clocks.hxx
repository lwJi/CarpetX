#ifndef CARPETX_CARPETX_HIERARCHY_CLOCKS_HXX
#define CARPETX_CARPETX_HIERARCHY_CLOCKS_HXX

#include "internal.hxx"

#include <carpetx.hxx>

namespace CarpetX {

inline constexpr int time_refinement = 2;

constexpr int substeps_per_coarsest_step(const int levels_below) {
  int substeps = 1;
  for (int n = 0; n < levels_below; ++n)
    substeps *= time_refinement;
  return substeps;
}

struct substep_t {
  rat64 begin, end;
  constexpr bool aligned() const { return end == 1; }
};

substep_t substep_of_parent_step(level_t child);

inline bool aligned_with_parent(const level_t child) {
  return substep_of_parent_step(child).aligned();
}

} // namespace CarpetX

#endif
