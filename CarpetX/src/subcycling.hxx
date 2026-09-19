#ifndef CARPETX_CARPETX_SUBCYCLING_HXX
#define CARPETX_CARPETX_SUBCYCLING_HXX

// Driver primitives for subcycling-in-time. The RK dense-output fill of the
// refinement-boundary ghosts follows AMReX's FillPatcher::fillRK order of
// operations: the coarse level keeps its start-of-step state and stage
// derivatives on zero-ghost "source bands" (the children's coarse-fine ghost
// footprint, see LevelData::build_bands), the time polynomial is evaluated on
// the coarse side to produce a single coarse *state* at the fine stage time,
// and only that state is prolongated in space into the fine ghost halo.
// No data is cached on the fine level between stages; the fine level only
// keeps the fill's two work buffers per evolved group (GroupData::rk_crse_patch
// and rk_fine_patch) allocated from regrid to regrid, so that a fill in steady
// state allocates nothing.
//
// Who calls what
// --------------
// All three entry points are C++-only, operate on one (patch, level) like the
// driver's other internals, and are called by the time integrator (ODESolvers).
// The integrator owns the RK tableau, the stage order and the choice of
// (stage, xsi) evaluation points, and publishes GHExt::num_rk_stages and
// GHExt::rk_integrated_group at WRAGH. The driver owns the buffers, the GPU
// streams, every device wait on the subcycling path, and the counters.
//
// Per solver call on a level the integrator calls StoreRKOldState once, before
// the stages overwrite the state; and per stage StoreRKStage, after it has
// issued the stage's linear combinations, followed by FillRKBoundary for the
// next evaluation point. After a recovery it calls FillRKBoundary alone, on
// the levels that are behind their parent, from the bands the checkpoint
// restored. Nothing else calls these functions.
//
// They must be called from serial host code: not from inside an OpenMP
// parallel region, a local-mode routine or an MFIter loop. The stores and the
// fill communicate, so every process must make the same calls in the same
// order; and only outside of an MFIter loop is AMReX's current stream the
// default stream, which the rule below builds on.
//
// Contract: idle on return
// ------------------------
// On return from any of these, the device is idle. The driver owns every
// device wait on the subcycling path; the time integrator issues none (a
// `synchronize()` or `synchronize_device()` it adds under its solver scope is
// charged to the counter report and changes the checked-in counts). Each
// primitive closes with exactly one wait on all streams, so that its own
// result, and every kernel the caller issued before the call (ODESolvers
// launches its linear combinations with `drain_t::deferred`, i.e. without
// waiting for them), may be consumed by kernels on any stream or by host code.
// StoreRKOldState and StoreRKStage wait on every path: also on levels without
// children, where there is no band to fill, and for an empty group list. The
// one exception is a FillRKBoundary call that has nothing to fill (level 0, or
// subcycling off): it launches nothing and does not wait, so the device is as
// idle as it was on entry. A caller that needs its own kernels drained must
// therefore issue them before one of the two stores, not before the fill.
//
// The default-stream rule
// -----------------------
// Inside a primitive there is no wait of ours before the closing one. What
// orders the kernels instead is that every kernel CarpetX itself issues on
// this path goes on AMReX's default stream (stream 0), where issue order is
// execution order, and that every AMReX operation in between either waits for
// itself or runs on stream 0 as well. The AMReX behaviours this leans on, and
// the version they were read in, are written down above FillRKBoundary
// (subcycling.cxx); the same rule carries the subcycling SYNC that only
// exchanges same-level ghosts, see SyncGroupsByDirISubcycling
// (sync_restrict.cxx). Only these two paths ask for
// `bc_streams_t::default_stream`; a SYNC that prolongates, the regrid fills
// and every sync without subcycling keep their boundary kernels spread over
// the streams, together with their waits.
//
// What the rule asks of the caller:
// - The kernels a caller leaves in flight are its own to order until a store
//   has returned; nothing waits for them before that. ODESolvers orders them
//   by the same rule: a fused linear combination issued outside of an MFIter
//   loop goes on the default stream. Until the store has returned, their
//   result may therefore only be consumed by further default-stream kernels
//   (the next linear combination; a validity check, whose kernels
//   `loop_parallel` launches outside of its MFIter loop), not by kernels on
//   other streams and not by host code.
// - They must not write what the store reads (the interior of var(tl), or of
//   the rhs groups) from another stream: the store's copies are issued on the
//   default stream, without a wait ahead of them.
// - FillRKBoundary is entered with the device idle. ODESolvers guarantees
//   that by calling it right after a store, and in its recovery routine right
//   after a prolongation that ends in an all-stream wait.

#include "subcycling_tally.hxx"

#include <cctk.h>

#include <vector>

namespace CarpetX {

// var(tl) interior -> old_source_band on (patch, level); builds the bands
// lazily. Copies nothing on levels without children (there is nothing to
// prolongate to), but still closes with its wait. Must run single-threaded
// (build_bands opens its own MFIter region), once all levels exist (the band
// geometry reads the next finer level).
void StoreRKOldState(int patch, int level, const std::vector<int> &var_groups,
                     int tl);

// rhs interior -> ks_source_band[stage-1] of the paired evolved group on
// (patch, level). var_groups[i] pairs with rhs_groups[i]. Copies nothing on
// levels without children, but still closes with its wait.
void StoreRKStage(int patch, int level, const std::vector<int> &var_groups,
                  const std::vector<int> &rhs_groups, int stage);

// Parent's old_source_band + ks_source_band[] -> dense output at (stage, xsi)
// on the parent's band geometry -> coarse boundary conditions -> spatial
// prolongation (the group's interpolator) into the refinement-boundary ghosts
// of var(tl) on (patch, level). No-op at level 0, without a wait. Allocates
// the level's persistent fill buffers on first use, so there is no "bands must
// have been built by this process first" precondition beyond the parent's
// bands existing (recovery reads them from the checkpoint). `dtc` is the
// parent's time step; `xsi` is the fine substep's start within the parent step
// (0 or 1/2), possibly plus 1/2 for the virtual end-of-substep evaluation.
// Ghost validity is left to the caller.
void FillRKBoundary(int patch, int level, const std::vector<int> &var_groups,
                    int tl, int stage, CCTK_REAL xsi, CCTK_REAL dtc);

// Counter report (`CarpetX::out_subcycling_counts`). The time integrator takes
// part through these two entry points only; both are no-ops unless the report
// is on.

// RAII: while alive, the waits, temporaries and kernel launches of the
// subcycling path are charged to one solver call on `level`.
struct SubcyclingSolverScope : TallyScope {
  explicit SubcyclingSolverScope(const int level)
      : TallyScope(scope_kind_t::solver, level) {}
};

// Report the number of kernel launches of one RK linear combination, as a GPU
// build issues them (one per group); CPU builds report the same number. Called
// by `statecomp_t::lincomb`, inside or outside of a solver scope; outside, the
// number is dropped.
void CountLincombLaunches(int n);

} // namespace CarpetX

#endif // #ifndef CARPETX_CARPETX_SUBCYCLING_HXX
