#ifndef CARPETX_CARPETX_SYNC_RESTRICT_INTERNAL_HXX
#define CARPETX_CARPETX_SYNC_RESTRICT_INTERNAL_HXX

#include <cctk.h>

namespace CarpetX {

// Intra-CarpetX symbols defined in sync_restrict.cxx and called from
// schedule.cxx. Not part of the public schedule.hxx surface.

// The reflux, Reflux(cctkGH, min_level, max_level), is declared in
// subcycling.hxx, which exports it to ODESolvers as well.
void Restrict(const cGH *cctkGH, int level);
void ProlongateRestrictedGFs(const cGH *cctkGH);

} // namespace CarpetX

#endif // #ifndef CARPETX_CARPETX_SYNC_RESTRICT_INTERNAL_HXX
