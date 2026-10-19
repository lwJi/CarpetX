#ifndef CARPETX_CARPETX_SUBCYCLING_HXX
#define CARPETX_CARPETX_SUBCYCLING_HXX

#include <cctk.h>

#include <vector>

namespace CarpetX {

void EnsureRKBuffers(int patch, int level, int gi);

void StoreRKOldState(int patch, int level, const std::vector<int> &var_groups,
                     int tl);

void StoreRKStage(int patch, int level, const std::vector<int> &var_groups,
                  const std::vector<int> &rhs_groups, int stage);

void FillRKBoundary(int patch, int level, const std::vector<int> &var_groups,
                    int tl, int stage, CCTK_REAL xsi, CCTK_REAL dtc);

bool group_is_integrated(int gi);

bool group_has_flux_register(int gi);

bool window_is_restricted(int min_level, int max_level);

void EnsureFluxGeometry(int patch, int level);

void AccumulateFluxes(int patch, int level, int stage, CCTK_REAL weight);

void Reflux(const cGH *cctkGH, int min_level, int max_level);

} // namespace CarpetX

#endif
