#ifndef CARPETX_CARPETX_SUBCYCLING_HXX
#define CARPETX_CARPETX_SUBCYCLING_HXX

#include "internal.hxx"

#include <cctk.h>

#include <vector>

namespace CarpetX {

inline constexpr CCTK_REAL rk_substep_ratio = 0.5;

// Allocate (lazily, idempotently) all RK buffers of group gi on the refined
// level (patch, level >= 1): old_source_band, ks_source_band[0..num_rk_stages),
// rk_crse_patch, rk_fine_patch, from one FPinfo. No-op without subcycling, for
// groups not in rk_integrated_group, and for an empty coarse-fine footprint.
// Warms AMReX's FPinfo cache: must run single-threaded.
void EnsureRKBuffers(int patch, int level, int gi);

// var(tl) interior on (patch, level) -> old_source_band of the same group on
// (patch, level + 1), which owns the bands; allocates that level's RK buffers
// lazily (EnsureRKBuffers). No-op on levels without children (there is nothing
// to prolongate to). Must run single-threaded.
void StoreRKOldState(int patch, int level, const std::vector<int> &var_groups,
                     int tl);

// rhs interior on (patch, level) -> ks_source_band[stage-1] of the paired
// evolved group on (patch, level + 1). var_groups[i] pairs with rhs_groups[i].
// Never allocates. No-op on levels without children.
void StoreRKStage(int patch, int level, const std::vector<int> &var_groups,
                  const std::vector<int> &rhs_groups, int stage);

// This level's old_source_band + ks_source_band[] (filled by the parent) ->
// dense output at (stage, xsi) on the band geometry -> coarse boundary
// conditions -> spatial prolongation (the group's interpolator) into the
// refinement-boundary ghosts of var(tl) on (patch, level). No-op at level 0.
// `dtc` is the parent's time step; `xsi` is the fine substep's start within
// the parent step (0 or 1/2), possibly plus 1/2 for the virtual
// end-of-substep evaluation. Ghost validity is left to the caller.
//
// Called only for group_is_integrated groups (asserted). These are exactly the
// groups whose refined-level coarse-fine ghosts the subcycling SYNC
// (SyncGroupsByDirISubcycling) skips during evolution: this function owns
// them. Every other group gets them prolongated by the SYNC.
//
// Never allocates. Precondition: StoreRKOldState on the parent level, or the
// RecoverGH pre-pass, ran since this level was made; they allocate the bands
// and the work buffers. A null band then means an empty coarse-fine footprint.
void FillRKBoundary(int patch, int level, const std::vector<int> &var_groups,
                    int tl, int stage, CCTK_REAL xsi, CCTK_REAL dtc);

bool group_is_integrated(int gi);

bool group_has_flux_register(int gi);

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
