#ifndef CARPETX_CARPETX_COMM_SUBSTEPS_HXX
#define CARPETX_CARPETX_COMM_SUBSTEPS_HXX

#include "internal.hxx"

#include "hierarchy/clocks.hxx"
#include <carpetx.hxx>

namespace CarpetX {

substep_t substep_of_parent_step(level_t child);

inline bool aligned_with_parent(const level_t child) {
  return substep_of_parent_step(child).aligned();
}

bool window_is_restricted(window_t window);

} // namespace CarpetX

#endif
