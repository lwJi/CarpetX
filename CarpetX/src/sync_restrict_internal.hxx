#ifndef CARPETX_CARPETX_SYNC_RESTRICT_INTERNAL_HXX
#define CARPETX_CARPETX_SYNC_RESTRICT_INTERNAL_HXX

#ifndef CARPETX_INTERNAL
#error                                                                         \
    "CarpetX internal header; include <carpetx.hxx> instead (see agent_docs/carpetx-public-api.md)"
#endif

#include <cctk.h>

#include <vector>

namespace CarpetX {

// Intra-CarpetX symbols defined in sync_restrict.cxx and called from
// schedule.cxx. Not part of the public schedule.hxx surface.

// The groups the end-of-step restriction restricts (the restrict tag); one of
// the group selectors in sync_restrict.cxx
std::vector<int> collect_restrictable_groups();

struct active_levels_t;

void Restrict(const cGH *cctkGH, int min_level, int max_level,
              const std::vector<int> &groups);
void ProlongateAfterRestriction(const active_levels_t &active_levels,
                                const cGH *cctkGH);

} // namespace CarpetX

#endif // #ifndef CARPETX_CARPETX_SYNC_RESTRICT_INTERNAL_HXX
