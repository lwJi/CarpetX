#ifndef CARPETX_CARPETX_COMM_SUBSTEPS_HXX
#define CARPETX_CARPETX_COMM_SUBSTEPS_HXX

#include "internal.hxx"

#include <carpetx.hxx>

namespace CarpetX {

bool aligned_with_parent(level_t child);

bool window_is_restricted(window_t window);

} // namespace CarpetX

#endif
