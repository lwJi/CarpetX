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
// All three entry points are C++-only, operate on one (patch, level) like the
// driver's other internals, and are called by ODESolvers, which owns the RK
// tableau and the choice of (stage, xsi) evaluation points.
//
// Contract: on return from any of these, the device is idle. The driver owns
// every device wait on the subcycling path; the time integrator issues none.
// Each primitive closes with a wait on all streams, so that its own result,
// and every kernel the caller issued before the call (ODESolvers launches its
// linear combinations without waiting for them), may be consumed by kernels
// on any stream or by host code. StoreRKOldState and StoreRKStage wait on
// every path: also on levels without children, where there is no band to
// fill, and for an empty group list. The one exception is a FillRKBoundary
// call that has nothing to fill (level 0): it launches nothing and does not
// wait, so the device is as idle as it was on entry. A caller that needs its
// own kernels drained must therefore issue them before one of the two stores,
// not before the fill.

#include "subcycling_tally.hxx"

#include <cctk.h>

#include <vector>

namespace CarpetX {

// var(tl) interior -> old_source_band on (patch, level); builds the bands
// lazily. No-op on levels without children (there is nothing to prolongate
// to). Must run single-threaded (build_bands opens its own MFIter region).
void StoreRKOldState(int patch, int level, const std::vector<int> &var_groups,
                     int tl);

// rhs interior -> ks_source_band[stage-1] of the paired evolved group on
// (patch, level). var_groups[i] pairs with rhs_groups[i]. No-op on levels
// without children.
void StoreRKStage(int patch, int level, const std::vector<int> &var_groups,
                  const std::vector<int> &rhs_groups, int stage);

// Parent's old_source_band + ks_source_band[] -> dense output at (stage, xsi)
// on the parent's band geometry -> coarse boundary conditions -> spatial
// prolongation (the group's interpolator) into the refinement-boundary ghosts
// of var(tl) on (patch, level). No-op at level 0. Allocates the level's
// persistent fill buffers on first use, so there is no "bands must have been
// built by this process first" precondition beyond the parent's bands existing
// (recovery reads them from the checkpoint). `dtc` is the parent's time
// step; `xsi` is the fine substep's start within the parent step (0 or 1/2),
// possibly plus 1/2 for the virtual end-of-substep evaluation. Ghost validity
// is left to the caller.
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

// Report the number of kernel launches of one RK linear combination.
void CountLincombLaunches(int n);

} // namespace CarpetX

#endif // #ifndef CARPETX_CARPETX_SUBCYCLING_HXX
