#ifndef CARPETX_CARPETX_SYNC_RESTRICT_INTERNAL_HXX
#define CARPETX_CARPETX_SYNC_RESTRICT_INTERNAL_HXX

#include "internal.hxx"

#include <carpetx.hxx>

#include <cctk.h>

#include <span>
#include <vector>

namespace CarpetX {

// Intra-CarpetX symbols defined in sync_restrict.cxx and called from
// schedule.cxx. Not part of the public schedule.hxx surface.

std::vector<gi_t> collect_restrictable_groups();

struct active_levels_t;

void Restrict(const cGH *cctkGH, int min_level, int max_level,
              std::span<const gi_t> groups);
void ProlongateAfterRestriction(const active_levels_t &active_levels,
                                const cGH *cctkGH);

} // namespace CarpetX

#endif // #ifndef CARPETX_CARPETX_SYNC_RESTRICT_INTERNAL_HXX
