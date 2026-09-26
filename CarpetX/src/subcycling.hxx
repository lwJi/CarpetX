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

// Flux-register (reflux) accumulation for one RK stage on (patch, level).
// Called by both ODESolvers solvers once per stage, after ODESolvers_RHS
// evaluated the fluxes and before the state update consumes them (and marks
// them invalid as dependents of the state). `weight` is b_stage * dt of
// this level's step, the stage's effective weight in the update, so that
// after a full step a register holds exactly the flux combination the state
// received.
//
// For every ODESolvers-integrated GF group with a fluxes= tag (only these
// own a register; a flux-tagged group outside rk_integrated_group is never
// touched here; defined in sync_restrict.cxx):
//  - as the coarse side of the pair (level, level + 1): stage 1 zeroes the
//    child's register (the coarse step is the reset), then every stage adds
//    -weight * area_d * flux_d;
//  - as the fine side of (level - 1, level): every stage adds
//    +weight * area_d * flux_d into this level's own register.
// Fluxes are per unit area, following d/dt state + div(flux) = 0; area_d is
// the level's own face area, so that the fine faces under a coarse face sum
// to the coarse face and FluxRegister::Reflux can divide by the coarse cell
// volume. Reads only interior faces of the flux groups (time level 0, which
// must be valid there) and updates no valid flag. No-op with do_reflux = no
// and for groups without a register.
void AccumulateFluxes(int patch, int level, int stage, CCTK_REAL weight);

// Flux-register (reflux) correction of every level pair (level, level + 1)
// with both levels in [min_level, max_level), finest pair first: the coarse
// state's time level 0 receives register / volume on the cells next to the
// coarse-fine boundary, then its same-level ghosts and outer boundary are
// re-established (validity flags unchanged). Defined in sync_restrict.cxx.
//
// Two callers, one per solver, each at the point where it knows a pair's
// step is complete: the driver's evolve loop under subcycling, with the
// widened time-aligned window, once per coarse step in the restrict block
// before the fine state is restricted; ODESolvers_Solve without subcycling,
// with [0, num_levels), in its final stage after the state update. Every
// call must be followed by an ODESolvers_PostStep on the corrected levels
// before the state is read again: the correction leaves the state's
// dependents= groups (and, with restrict_during_sync, the in-sync
// restriction) to that traversal. No-op with do_reflux = no and for pairs
// without a register.
void Reflux(const cGH *cctkGH, int min_level, int max_level);

} // namespace CarpetX

#endif // #ifndef CARPETX_CARPETX_SUBCYCLING_HXX
