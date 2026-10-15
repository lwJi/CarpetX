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
// driver knows that a routine runs at end of step.
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
// Without subcycling none of this applies: TraverseClassified is a plain
// CCTK_Traverse, the phase stays `other`, and SYNCs and presyncs in these bins
// run as before.

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
