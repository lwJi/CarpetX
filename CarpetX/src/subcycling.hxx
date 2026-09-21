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

#include <cctk.h>

#include <AMReX_FabArrayBase.H>
#include <AMReX_Geometry.H>
#include <AMReX_Interpolater.H>
#include <AMReX_MultiFab.H>

#include <vector>

namespace CarpetX {

// The FPinfo of the RK boundary fill into `finemfab`: the one lookup behind
// both the parent's source bands (LevelData::build_bands) and the child's
// rk_crse_patch/rk_fine_patch (FillRKBoundary), which the dense-output kernel
// walks with one local box index. Cached by AMReX; the returned reference
// lives as long as `finemfab`'s layout does.
const amrex::FabArrayBase::FPinfo &
rk_fill_fpinfo(const amrex::MultiFab &finemfab,
               amrex::Interpolater *interpolator, const amrex::Geometry &fgeom,
               const amrex::Geometry &cgeom);

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

} // namespace CarpetX

#endif // #ifndef CARPETX_CARPETX_SUBCYCLING_HXX
