#ifndef CARPETX_CARPETX_SYNC_RESTRICT_INTERNAL_HXX
#define CARPETX_CARPETX_SYNC_RESTRICT_INTERNAL_HXX

#include <cctk.h>

#include <vector>

namespace CarpetX {

// Intra-CarpetX symbols defined in sync_restrict.cxx and called from
// schedule.cxx. Not part of the public schedule.hxx surface.

// The groups the end-of-step restriction restricts (the restrict tag, but
// never an end-of-step result); one of the group selectors in
// sync_restrict.cxx
std::vector<int> collect_restrictable_groups();

// Restrict exactly `groups`, every level pair (level, level + 1) with both
// levels in [min_level, max_level), finest pair first, and mark the coarse
// ghosts and outer boundaries invalid. Its counterpart without validity
// tracking, RestrictNoPoison, is exported in schedule.hxx. The reflux,
// Reflux(cctkGH, min_level, max_level), is declared in subcycling.hxx, which
// exports it to ODESolvers as well.
void Restrict(const cGH *cctkGH, int min_level, int max_level,
              const std::vector<int> &groups);
void ProlongateAfterRestriction(const cGH *cctkGH);

} // namespace CarpetX

#endif // #ifndef CARPETX_CARPETX_SYNC_RESTRICT_INTERNAL_HXX
