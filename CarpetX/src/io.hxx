#ifndef CARPETX_CARPETX_IO_HXX
#define CARPETX_CARPETX_IO_HXX

#include <cctk.h>

#include <limits>

namespace CarpetX {

void RecoverGridStructure(cGH *cctkGH);
void RecoverGH(const cGH *cctkGH);
void InputGH(const cGH *cctkGH);

int OutputGH(const cGH *cctkGH);

// The refinement levels [min_level, max_level) that a writer writes. A writer
// skips every other level: its data, and its per-level attributes or metadata.
struct level_range_t {
  int min_level, max_level;
  constexpr bool contains(const int level) const {
    return level >= min_level && level < max_level;
  }
};

// Every level, however many there are. Checkpoints write every level.
inline constexpr level_range_t every_level{0, std::numeric_limits<int>::max()};

// The levels that OutputGH writes: the active levels when it is called in a
// window, else every level. The driver calls it in the window whose levels
// share the finest level's clock: in Evolve the finest batch's widened window,
// in the Initialise tail [0, num_levels), and in recovery the finest clock
// group. Under subcycling, each level is thus written at the iterations at
// which the finest level has caught up with it; without subcycling the window
// is every level at every iteration.
level_range_t output_levels();

// Whether `levels` holds every level of the grid hierarchy, so that a
// reduction over all levels (the norms) sees one time
bool contains_every_level(const level_range_t &levels);

} // namespace CarpetX

#endif // #ifndef CARPETX_CARPETX_IO_HXX
