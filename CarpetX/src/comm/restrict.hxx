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

} // namespace CarpetX

#endif
