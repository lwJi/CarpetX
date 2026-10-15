#ifndef CARPETX_CARPETX_END_OF_STEP_HXX
#define CARPETX_CARPETX_END_OF_STEP_HXX

// The end of step (CCTK_POSTSTEP, CCTK_ANALYSIS) under subcycling.
//
// The driver starts the end-of-step traversals itself: in Evolve, after each
// batch's restriction block, and in Initialise, after the initial data (where
// CCTK_CPINITIAL, which runs between the two, stays outside). Under subcycling
// it traverses them through TraverseClassified, which records the bin being
// traversed in GHExt::phase and GHExt::bin. The flesh passes the driver
// neither the bin nor the schedule group a routine sits in (cFunctionData's
// `where` is the immediate schedule.ccl parent), so this is the only way the
// driver knows that a routine runs at end of step. Evolve traverses the
// evolution part of each batch (CCTK_PRESTEP, CCTK_EVOL, CarpetX_PreRestrict,
// CCTK_POSTRESTRICT) through TraverseClassified as well, with phase
// `evolution`, so that check_walked covers every routine those bins reach,
// including through CallScheduleGroup.
//
// The no-sync rule. Under subcycling nothing scheduled in CCTK_POSTSTEP or
// CCTK_ANALYSIS, directly or through a schedule group placed there, may cause
// a ghost synchronization. End-of-step routines see the ghosts as they stand
// and never cause them to be filled. Every way a sync could happen there is an
// error, caught where the sync would run, before it runs:
//
//  - a routine's SYNC clause: CallFunction, before the routine runs (the flesh
//    would sync at item exit, since CallFunction reports "not synchronised",
//    and by then the routine could no longer be named);
//  - a READS clause that needs ghosts that are invalid under
//    presync_mode = "presync-only": CallFunction, before its presync;
//  - an explicit CCTK_SyncGroup* call in a routine's body, and a schedule
//    group's SYNC clause: the subcycling overload
//    (SyncGroupsByDirISubcycling), which names the routine running, if any
//    (CCTK_ScheduleQueryCurrentFunction).
//
// Only syncs that would happen are errors: a SYNCing routine that an `if`
// disables is fine, and so is a READS clause whose ghosts are already valid.
// The evolved state's ghosts are valid at end of step (the solver leaves them
// valid after each substep, and the ODESolvers_PostStep SYNC at POSTRESTRICT
// repairs them after the reflux and the restriction), so a pointwise
// diagnostic reads the state everywhere and writes everywhere, and a stencil
// diagnostic writes its interior only, leaving its ghosts invalid. There is no
// override.
//
// End-of-step results. A grid-function group is an end-of-step result if a
// routine in CCTK_POSTSTEP or CCTK_ANALYSIS WRITES it and nothing in the
// evolution part of a step does: CCTK_PRESTEP, CCTK_EVOL (including the
// groups ODESolvers runs only through CallScheduleGroup, ODESolvers_RHS and
// ODESolvers_PostStep), CarpetX_PreRestrict and CCTK_POSTRESTRICT. The groups
// the time integrator advances (GHExt::rk_integrated_group, which ODESolvers
// writes without a WRITES clause naming them) count as written in the
// evolution part. Grid scalars and arrays, which belong to no level, never
// count, and neither do writes in any other bin (initial data, regrid,
// recovery). Examples: diagnostics (error, energy, constraints),
// CarpetXRegrid::regrid_error; not the evolved state, even where a thorn
// adjusts it at POSTSTEP, and not ODESolvers_RHS outputs.
//
// ClassifyEndOfStepResults derives the set once, under subcycling, before
// initial data or recovery, so it is the same in a fresh and in a restarted
// run. It does not observe calls; it walks the sorted schedule tree of those
// bins with the flesh's structural traversal (CCTKi_DoScheduleTraverse),
// which ignores `if`, `while` and triggers and visits every group body once.
// That is how it sees the groups ODESolvers runs only through
// CallScheduleGroup: ODESolvers/schedule.ccl places them, for the schedule
// printout, in ODESolvers_CalledBySolve AT EVOL WHILE (do_substeps), whose
// body never runs. The walk is conservative (a routine an `if` disables still
// counts as a writer), and it reads WRITES clauses only (an undeclared write
// is invisible to it, as it is to validity tracking).
//
// The walk reaches each routine's public cFunctionData through a mirrored
// prefix of the flesh's private t_attribute (end_of_step.cxx). The mirror is
// verified on every call rather than trusted: the flesh hands CallFunction a
// pointer into the same t_attribute, so under subcycling every routine
// CarpetX calls from a classified traversal must arrive at a pointer the walk
// recorded (check_walked). The same check rejects a group that
// CallScheduleGroup reaches during evolution or at end of step but the walk
// cannot see because no bin holds it; such a group must be placed under a
// bin like ODESolvers_CalledBySolve.
//
// Without subcycling none of this applies: TraverseClassified is a plain
// CCTK_Traverse, the phase stays `other`, SYNCs and presyncs in these bins
// run as before, and the schedule is neither walked nor checked.

#include <cctk.h>
#include <cctk_Schedule.h>

#include <string>

namespace CarpetX {

// What the driver is traversing, as far as the end of step is concerned. Set
// only by TraverseClassified, and only under subcycling.
enum class traversal_phase_t { other, evolution, end_of_step };

// CCTK_Traverse(cctkGH, bin) with GHExt::phase = phase and GHExt::bin = bin
// for the duration of the traversal, restored to (other, nullptr) on exit.
// Classified traversals do not nest (asserted). Without subcycling this is a
// plain CCTK_Traverse and sets nothing.
void TraverseClassified(cGH *cctkGH, const char *bin, traversal_phase_t phase);

// Under subcycling, walk the schedule of the evolution and end-of-step bins,
// record every walked routine (GHExt::walked_functions), and derive the
// end-of-step results (GHExt::end_of_step_result). Prints one INFO line
// listing them. Called once by Initialise, after WRAGH (it needs
// rk_integrated_group) and before initial data or recovery. Without
// subcycling it does nothing.
void ClassifyEndOfStepResults();

// Whether group gi is an end-of-step result. Always false without
// subcycling.
bool is_end_of_step_result(int gi);

// Abort unless the walk recorded `attribute`: either the routine sits in a
// group that is traversed only through CallScheduleGroup and placed under no
// walked bin, or the flesh's t_attribute layout no longer matches the mirror.
// Called by CallFunction under subcycling, inside a classified traversal.
void check_walked(const cFunctionData *attribute);

// Abort with the no-sync rule: "under subcycling, nothing scheduled in <bin>
// may SYNC", followed by `what` (the subject: routine and clause or call) and
// the full names of the n groups. Called only inside an end-of-step traversal
// (asserted). Never returns.
[[noreturn]] void error_if_end_of_step_sync(const char *what, int n,
                                            const int *groups);

// The subject of an error_if_end_of_step_sync about the scheduled function
// `attribute`: "Routine <thorn>::<routine> (in group <where>)", with the group
// omitted when the routine is scheduled directly in the bin.
std::string end_of_step_subject(const cFunctionData *attribute);

} // namespace CarpetX

#endif // #ifndef CARPETX_CARPETX_END_OF_STEP_HXX
