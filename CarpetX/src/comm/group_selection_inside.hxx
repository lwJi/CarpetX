#ifndef CARPETX_CARPETX_COMM_GROUP_SELECTION_INSIDE_HXX
#define CARPETX_CARPETX_COMM_GROUP_SELECTION_INSIDE_HXX

#include "internal.hxx"

#include <carpetx.hxx>

#include <span>
#include <vector>

namespace CarpetX {

void sync_log_groups(const char *label, std::span<const gi_t> groups0);

std::vector<gi_t> sync_filter_groups(std::span<const gi_t> groups0);

} // namespace CarpetX

#endif
