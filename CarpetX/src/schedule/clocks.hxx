#ifndef CARPETX_CARPETX_SCHEDULE_CLOCKS_HXX
#define CARPETX_CARPETX_SCHEDULE_CLOCKS_HXX

#include "internal.hxx"

#include "writers/writers.hxx"

#include <carpetx.hxx>

#include <cctk.h>

#include <vector>

namespace CarpetX {

level_t WidenMinLevel(level_t level);
window_t FinestClockGroup();
std::vector<window_t> StepGroups();

void SetDeltaTime(cGH *restrict cctkGH);
void SetStepTime(cGH *restrict cctkGH, level_t level, rat64 clock);

void RecoverFromCheckpoint(cGH *restrict cctkGH, clock_writer_t &clock,
                           structure_writer_t &structure);

void TraverseTailBin(cGH *restrict cctkGH, const char *bin);
void FinalizeWindow(window_t window, cGH *restrict cctkGH, bool reflux);

bool EvolutionIsDone(cGH *restrict cctkGH);
void CycleTimelevels(window_t window, cGH *restrict cctkGH,
                     clock_writer_t &clock);

} // namespace CarpetX

#endif
