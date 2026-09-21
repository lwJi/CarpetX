#ifndef CARPETX_CARPETX_SUBCYCLING_HXX
#define CARPETX_CARPETX_SUBCYCLING_HXX

// Driver primitives for subcycling-in-time. The RK dense-output fill of the
// refinement-boundary ghosts follows AMReX's FillPatcher::fillRK order of
// operations: the coarse level's start-of-step state and stage derivatives are
// kept on zero-ghost "source bands" (the coarse cells underneath the refined
// level's coarse-fine ghost footprint), the time polynomial is evaluated on
// the coarse side to produce a single coarse *state* at the fine stage time,
// and only that state is prolongated in space into the fine ghost halo.
//
// The refined level owns everything the fill needs, per evolved group: the
// source bands, which the coarse level fills during its own step
// (StoreRKOldState, StoreRKStage), and the fill's two work buffers
// (GroupData::rk_crse_patch and rk_fine_patch). All of them have the geometry
// of one FPinfo, are allocated together (EnsureRKBuffers) and stay allocated
// from regrid to regrid, so that a fill in steady state allocates nothing.
//
// All entry points are C++-only and operate on one (patch, level) like the
// driver's other internals. StoreRKOldState, StoreRKStage and FillRKBoundary
// are called by ODESolvers, which owns the RK tableau and the choice of
// (stage, xsi) evaluation points; EnsureRKBuffers is called by StoreRKOldState
// and by the recovery path (RecoverGH).

#include <cctk.h>

#include <vector>

namespace CarpetX {

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
// Never allocates. Precondition: StoreRKOldState on the parent level, or the
// RecoverGH pre-pass, ran since this level was made; they allocate the bands
// and the work buffers. A null band then means an empty coarse-fine footprint.
void FillRKBoundary(int patch, int level, const std::vector<int> &var_groups,
                    int tl, int stage, CCTK_REAL xsi, CCTK_REAL dtc);

} // namespace CarpetX

#endif // #ifndef CARPETX_CARPETX_SUBCYCLING_HXX
