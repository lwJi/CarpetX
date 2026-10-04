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

// True iff group gi is advanced by the time integrator, i.e. listed in
// ghext->rk_integrated_group, which ODESolvers publishes at WRAGH (the grid
// functions with an rhs= tag). An empty vector, as when ODESolvers is not
// active, means nothing is integrated. This is the one definition of "gi is
// integrated"; SyncGroupsByDirISubcycling (which skips the refined-level
// coarse-fine ghosts of exactly these groups during evolution),
// FillRKBoundary, EnsureRKBuffers, StoreRKOldState, group_has_flux_register,
// warn_inert_flux_tags, ODESolvers_CheckCheckpointed and
// ODESolvers_CheckEvolved call it. Uses only ghext and no grid structure, so
// it is valid from PARAMCHECK on, before any level exists. Defined in
// driver.cxx.
bool group_is_integrated(int gi);

// True iff group gi gets a flux register on every level > 0: a grid function
// with a non-empty fluxes= tag, integrated by ODESolvers
// (group_is_integrated), with CarpetX::do_reflux.
// This is the one definition of "gi is refluxed"; the GroupData constructor
// (register allocation), ODESolvers_CheckMethod and the startup warning
// about inert fluxes= tags (warn_inert_flux_tags, driver.hxx) call it. Uses
// only tag tables, parameters and ghext, and no grid structure, so it is valid
// from PARAMCHECK on, before any level exists. Defined in driver.cxx.
bool group_has_flux_register(int gi);

// Allocate (lazily, idempotently) the cached flux geometry of (patch,
// level): LevelData::face_area[0..dim) and LevelData::cell_volume, filled
// with the level's constant face areas and cell volume by
// Geom(level).GetFaceArea and GetVolume, on the level's own layout (see
// driver.hxx). Called by AccumulateFluxes and Reflux on a level with a
// register on either side; defined in sync_restrict.cxx.
void EnsureFluxGeometry(int patch, int level);

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
// the level's own face area (the constant of LevelData::face_area[d], see
// EnsureFluxGeometry), so that the fine faces under a coarse face sum to
// the coarse face and FluxRegister::Reflux can divide by the coarse cell
// volume. Reads only interior faces of the flux groups (time level 0, which
// must be valid there) and updates no valid flag. No-op with do_reflux = no
// and for groups without a register. Allocates nothing in steady state: the
// coarse side stages through the child's GroupData::freg_scratch.
void AccumulateFluxes(int patch, int level, int stage, CCTK_REAL weight);

// Flux-register (reflux) correction of every level pair (level, level + 1)
// with both levels in [min_level, max_level), finest pair first: the coarse
// state's time level 0 receives register / volume on the interior cells next
// to the coarse-fine boundary. Only the registers are applied: no ghost,
// outer boundary point or validity flag is touched, so the corrected cells'
// copies in same-level ghosts (including periodic images), in the outer
// boundary and in inter-patch ghosts keep their old values, and so do the
// state's dependents= groups. Defined in sync_restrict.cxx.
//
// Two callers, one per solver, each at the point where it knows a pair's
// step is complete, and each followed by an ODESolvers_PostStep on the
// corrected levels before the state is read again. That traversal's SYNC
// carries the correction into the ghosts, the outer boundary and the
// inter-patch ghosts, and it recomputes the dependents:
//  - ODESolvers_Solve without subcycling, with [0, num_levels), in its
//    final stage after the state update, right before that stage's
//    PostStep (which, with restrict_during_sync, also restricts); the state
//    is valid on the interior only there;
//  - the driver's evolve loop under subcycling, with the widened
//    time-aligned window, once per coarse step in the restrict block, after
//    CarpetX_PreRestrict and immediately before restriction. The PostStep
//    is the one at POSTRESTRICT. Until then the state's ghost flags stay set
//    over stale copies, exactly as they do after the restriction itself
//    (RestrictNoPoison), and nothing in between reads them: restriction
//    reads fine interior cells only, and ProlongateRestrictedGFs copies no
//    coarse ghost. Clearing the flags instead would not work: the
//    subcycling SYNC never re-marks an evolved group's ghosts valid at
//    iteration > 0 (the solver does, after each refinement-boundary fill).
// No-op with do_reflux = no and for pairs without a register.
void Reflux(const cGH *cctkGH, int min_level, int max_level);

} // namespace CarpetX

#endif // #ifndef CARPETX_CARPETX_SUBCYCLING_HXX
