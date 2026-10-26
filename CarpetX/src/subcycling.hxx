#ifndef CARPETX_CARPETX_SUBCYCLING_HXX
#define CARPETX_CARPETX_SUBCYCLING_HXX

#include "internal.hxx"

#include <carpetx.hxx>

#include <cctk.h>

#include <span>

namespace CarpetX {

inline constexpr CCTK_REAL rk_substep_ratio = 0.5;

void EnsureRKBuffers(int patch, int level, gi_t gi);

void StoreRKOldState(int patch, int level, std::span<const gi_t> var_groups,
                     int tl);

void StoreRKStage(int patch, int level, std::span<const gi_t> var_groups,
                  std::span<const gi_t> rhs_groups, int stage);

void FillRKBoundary(int patch, int level, std::span<const gi_t> var_groups,
                    int tl, int stage, CCTK_REAL xsi, CCTK_REAL dtc);

bool group_is_integrated(gi_t gi);

bool group_has_flux_register(gi_t gi);

struct active_levels_t;

void RecoverRKBoundary(const active_levels_t &active_levels, const cGH *cctkGH);

// True iff the driver restricts the window [min_level, max_level) of
// time-aligned levels at its end of step: the window holds the finest level
// and at least one other. The driver then restricts the window and refills
// the coarse-fine ghosts of every window level aligned with its parent by
// spatial prolongation from the restricted parent
// (ProlongateAfterRestriction, for every restricted group and, under
// subcycling, every integrated one, whether restricted or not), after the
// integrator's end-of-step fill. Any
// other window keeps the integrator's fill until a later window that holds it
// is restricted. This is the one definition of "the window is restricted";
// FinalizeWindow, for Initialise and Evolve, calls it. During recovery the
// window is a clock group, the window in which its levels last stepped
// (ClockGroups), so it tells which of the two fills last wrote the
// integrated groups' ghosts. Defined in schedule.cxx.
bool window_is_restricted(int min_level, int max_level);

// Allocate (lazily, idempotently) the cached flux geometry of (patch,
// level): LevelData::face_area[0..dim) and LevelData::cell_volume, filled
// with the level's constant face areas and cell volume by
// Geom(level).GetFaceArea and GetVolume, on the level's own layout (see
// driver.hxx). Called by AccumulateFluxes and Reflux on a level with a
// register on either side; defined in sync_restrict.cxx.
void EnsureFluxGeometry(int patch, int level);

void AccumulateFluxes(int patch, int level, int stage, CCTK_REAL weight);

void Reflux(const cGH *cctkGH, int min_level, int max_level);

} // namespace CarpetX

#endif // #ifndef CARPETX_CARPETX_SUBCYCLING_HXX
