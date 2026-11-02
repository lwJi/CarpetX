#ifndef CARPETX_CARPETX_SCHEDULE_CLOCKS_HXX
#define CARPETX_CARPETX_SCHEDULE_CLOCKS_HXX

#include "internal.hxx"

#include <carpetx.hxx>

#include <cctk.h>

#include <vector>

namespace CarpetX {

level_t WidenMinLevel(level_t level, rat64 target_iteration);
window_t FinestClockGroup();
std::vector<window_t> StepGroups();

void SetDeltaTime(cGH *restrict cctkGH);
void SetStepTime(cGH *restrict cctkGH, level_t level, rat64 clock);

void RecoverFromCheckpoint(cGH *restrict cctkGH);

void TraverseTailBin(cGH *restrict cctkGH, const char *bin);
void FinalizeWindow(window_t window, cGH *restrict cctkGH, bool reflux);

bool EvolutionIsDone(cGH *restrict cctkGH);
void CycleTimelevels(window_t window, cGH *restrict cctkGH);

} // namespace CarpetX

#endif
