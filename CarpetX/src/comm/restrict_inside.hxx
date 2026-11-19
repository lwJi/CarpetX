#ifndef CARPETX_CARPETX_COMM_RESTRICT_INSIDE_HXX
#define CARPETX_CARPETX_COMM_RESTRICT_INSIDE_HXX

#include "internal.hxx"

#include <carpetx.hxx>

#include <span>

namespace CarpetX {

void restrict_level(level_t level, std::span<const gi_t> groups,
                    valid_t coarse_keeps);

} // namespace CarpetX

#endif
