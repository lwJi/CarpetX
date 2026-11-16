#include "comm/group_selection.hxx"

#include "hierarchy/model.hxx"
#include <carpetx.hxx>

#include <cctk.h>
#include <cctk_Parameters.h>

#include <cstddef>
#include <span>
#include <sstream>
#include <vector>

namespace CarpetX {

void sync_log_groups(const char *label, const std::span<const gi_t> groups0) {
  DECLARE_CCTK_PARAMETERS;
  if (verbose) {
    std::ostringstream buf;
    for (std::size_t n = 0; n < groups0.size(); ++n) {
      if (n != 0)
        buf << ", ";
      buf << CCTK_FullGroupName(groups0[n].index());
    }
#pragma omp critical
    CCTK_VINFO("%s %s", label, buf.str().c_str());
  }
}

std::vector<gi_t> sync_filter_groups(const std::span<const gi_t> groups0) {
  std::vector<gi_t> groups;
  for (const gi_t gi : groups0) {
    const auto &group = driver().record(gi);
    if (!group.has_storage() || group.kind != group_kind::grid_function)
      continue;
    groups.push_back(gi);
  }
  return groups;
}

} // namespace CarpetX
