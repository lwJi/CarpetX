#ifndef CARPETX_CARPETX_SYNC_RESTRICT_INTERNAL_HXX
#define CARPETX_CARPETX_SYNC_RESTRICT_INTERNAL_HXX

#include <cctk.h>

namespace CarpetX {

// Intra-CarpetX symbols defined in sync_restrict.cxx and called from
// schedule.cxx. Not part of the public schedule.hxx surface.

// Apply the flux register of the pair (level, level + 1) to the coarse
// state's time level 0. Postcondition: the refluxed groups' same-level
// ghosts (including periodic images) and outer boundary hold the corrected
// values, and every validity flag is what it was before the call. The
// finest-first cascade over a level range, Reflux(cctkGH, min_level,
// max_level), is exported through subcycling.hxx for ODESolvers.
void Reflux(const cGH *cctkGH, int level);
void Restrict(const cGH *cctkGH, int level);
void ProlongateRestrictedGFs(const cGH *cctkGH);

} // namespace CarpetX

#endif // #ifndef CARPETX_CARPETX_SYNC_RESTRICT_INTERNAL_HXX
