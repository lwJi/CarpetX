#ifndef CARPETX_CARPETX_SUBCYCLING_HXX
#define CARPETX_CARPETX_SUBCYCLING_HXX

// Driver primitives for subcycling-in-time. The RK dense-output fill of the
// refinement-boundary ghosts follows AMReX's FillPatcher::fillRK order of
// operations: the coarse level keeps its start-of-step state and stage
// derivatives on zero-ghost "source bands" (the children's coarse-fine ghost
// footprint, see LevelData::build_bands), the time polynomial is evaluated on
// the coarse side to produce a single coarse *state* at the fine stage time,
// and only that state is prolongated in space into the fine ghost halo.
// Nothing is cached on the fine level between stages.
//
// All three entry points are C++-only, operate on one (patch, level) like the
// driver's other internals, and are called by ODESolvers, which owns the RK
// tableau and the choice of (stage, xsi) evaluation points.

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
// of var(tl) on (patch, level). No-op at level 0. `dtc` is the parent's time
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
