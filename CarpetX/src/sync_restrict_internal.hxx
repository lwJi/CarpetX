#ifndef CARPETX_CARPETX_SYNC_RESTRICT_INTERNAL_HXX
#define CARPETX_CARPETX_SYNC_RESTRICT_INTERNAL_HXX

#include <cctk.h>

#include <vector>

namespace CarpetX {

std::vector<int> collect_restrictable_groups();

void Restrict(const cGH *cctkGH, int min_level, int max_level,
              const std::vector<int> &groups);
void ProlongateAfterRestriction(const cGH *cctkGH);

} // namespace CarpetX

#endif
