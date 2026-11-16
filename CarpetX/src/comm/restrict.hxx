#ifndef CARPETX_CARPETX_COMM_RESTRICT_HXX
#define CARPETX_CARPETX_COMM_RESTRICT_HXX

#include "internal.hxx"

#include <carpetx.hxx>

#include <span>
#include <vector>

namespace CarpetX {

std::vector<gi_t> collect_restrictable_groups();

void Restrict(window_t window, std::span<const gi_t> groups);

void RestrictNoPoison(window_t window, std::span<const gi_t> groups);

void restrict_level(level_t level, std::span<const gi_t> groups,
                    valid_t coarse_keeps);

} // namespace CarpetX

#endif
