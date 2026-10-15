#include "end_of_step.hxx"
#include "driver.hxx"
#include "schedule.hxx"
#include "subcycling.hxx"
#include "valid.hxx"

#include <cctk.h>
#include <cctk_Schedule.h>
#include <cctki_Schedule.h>

#include <cassert>
#include <cstring>
#include <initializer_list>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

namespace CarpetX {

namespace {

// The leading members of the flesh's private t_attribute
// (flesh/src/main/ScheduleInterface.c), which CCTKi_DoScheduleTraverse hands
// its function callback. They have not changed since flesh commit 0018aad4.
// `type` is an iSchedType there, an enum, hence an int. The mirror is not
// trusted: check_walked verifies it on every call (end_of_step.hxx).
struct flesh_attribute_prefix {
  char *description;
  char *implementation;
  int type;
  cFunctionData FunctionData;
};

// The state of one walk: the writer set of the bins being walked, and every
// routine seen
struct walk_t {
  std::vector<bool> &written; // [group index]
  std::unordered_set<const cFunctionData *> &walked;
};

// CCTKi_DoScheduleTraverse function callback: record the routine and the
// groups its WRITES clauses name. The flesh decodes the clauses into RDWR
// whenever presync_mode is not "off", and CarpetX requires "mixed-error" or
// "presync-only" (Initialise).
int record_function(void * /*function*/, void *const attribute,
                    void *const data) {
  assert(attribute);
  assert(data);
  const cFunctionData *const fdata =
      &static_cast<const flesh_attribute_prefix *>(attribute)->FunctionData;
  walk_t &walk = *static_cast<walk_t *>(data);
  walk.walked.insert(fdata);
  for (int n = 0; n < fdata->n_RDWR; ++n) {
    const RDWR_entry &entry = fdata->RDWR[n];
    if (entry.where_wr == 0)
      continue;
    const int gi = CCTK_GroupIndexFromVarI(entry.varindex);
    assert(gi >= 0 && gi < int(walk.written.size()));
    walk.written.at(gi) = true;
  }
  return 1;
}

// Walk one bin as CCTK_ScheduleTraverse would traverse it ($ENTRY, the bin,
// $EXIT), but structurally: no if, while or trigger is evaluated, and every
// group body is visited once. The $ENTRY and $EXIT groups exist only if
// something is scheduled there.
void walk_bin(const char *const bin, walk_t &walk) {
  for (const char *const suffix : {"$ENTRY", "", "$EXIT"}) {
    const std::string group = std::string(bin) + suffix;
    const int ierr = CCTKi_DoScheduleTraverse(group.c_str(), nullptr, nullptr,
                                              nullptr, nullptr,
                                              record_function, &walk);
    if (*suffix == '\0' && ierr < 0)
      CCTK_VERROR("End-of-step classification: schedule bin %s not found",
                  bin);
  }
}

// The bins of the evolution part of a step, and of the end of step, as Evolve
// traverses them (TraverseClassified)
constexpr const char *evolution_bins[] = {
    "CCTK_PRESTEP", "CCTK_EVOL", "CarpetX_PreRestrict", "CCTK_POSTRESTRICT"};
constexpr const char *end_of_step_bins[] = {"CCTK_POSTSTEP", "CCTK_ANALYSIS"};

// Marks a classified traversal (GHExt::phase, GHExt::bin), like
// recovery_traversal_scope (schedule.cxx)
struct traversal_phase_scope {
  traversal_phase_scope(const char *bin, const traversal_phase_t phase) {
    assert(phase != traversal_phase_t::other);
    assert(ghext->phase == traversal_phase_t::other);
    assert(!ghext->bin);
    ghext->phase = phase;
    ghext->bin = bin;
  }
  ~traversal_phase_scope() {
    ghext->phase = traversal_phase_t::other;
    ghext->bin = nullptr;
  }
  traversal_phase_scope(const traversal_phase_scope &) = delete;
  traversal_phase_scope &operator=(const traversal_phase_scope &) = delete;
};
} // namespace

void TraverseClassified(cGH *const cctkGH, const char *const bin,
                        const traversal_phase_t phase) {
  assert(bin);
  if (!ghext->use_subcycling) {
    CCTK_Traverse(cctkGH, bin);
    return;
  }
  traversal_phase_scope scope(bin, phase);
  CCTK_Traverse(cctkGH, bin);
}

void ClassifyEndOfStepResults() {
  if (!ghext->use_subcycling)
    return;
  assert(ghext->walked_functions.empty());
  assert(ghext->end_of_step_result.empty());

  const int numgroups = CCTK_NumGroups();
  std::vector<bool> written_in_evolution(numgroups, false);
  std::vector<bool> written_at_end_of_step(numgroups, false);

  walk_t evolution_walk{written_in_evolution, ghext->walked_functions};
  for (const char *const bin : evolution_bins)
    walk_bin(bin, evolution_walk);
  walk_t end_of_step_walk{written_at_end_of_step, ghext->walked_functions};
  for (const char *const bin : end_of_step_bins)
    walk_bin(bin, end_of_step_walk);

  // The time integrator writes the groups it advances without a WRITES
  // clause naming them
  for (int gi = 0; gi < numgroups; ++gi)
    if (group_is_integrated(gi))
      written_in_evolution.at(gi) = true;

  std::vector<bool> &result = ghext->end_of_step_result;
  result.assign(numgroups, false);
  for (int gi = 0; gi < numgroups; ++gi)
    result.at(gi) = CCTK_GroupTypeI(gi) == CCTK_GF &&
                    written_at_end_of_step.at(gi) &&
                    !written_in_evolution.at(gi);

  // Every process walks the same schedule; only the root reports it
  if (CCTK_MyProc(nullptr) == 0) {
    std::ostringstream buf;
    for (int gi = 0; gi < numgroups; ++gi) {
      if (!result.at(gi))
        continue;
      if (buf.tellp() > 0)
        buf << ", ";
      buf << CCTK_FullGroupName(gi);
    }
    CCTK_VINFO("Subcycling: end-of-step results (grid functions written in "
               "CCTK_POSTSTEP or CCTK_ANALYSIS, but not in the evolution part "
               "of a step): %s",
               buf.tellp() > 0 ? buf.str().c_str() : "none");
  }
}

bool is_end_of_step_result(const int gi) {
  assert(gi >= 0 && gi < CCTK_NumGroups());
  const std::vector<bool> &result = ghext->end_of_step_result;
  return gi < int(result.size()) && result.at(gi);
}

void InvalidateEndOfStepResults(const int min_level, const int max_level) {
  if (!ghext->use_subcycling)
    return;
  assert(min_level >= 0 && min_level < max_level &&
         max_level <= ghext->num_levels());
  // One clock: a batch or a clock group (asserted by the loops)
  const active_levels_t levels(min_level, max_level);
  const int numgroups = CCTK_NumGroups();
  for (int gi = 0; gi < numgroups; ++gi) {
    if (!is_end_of_step_result(gi))
      continue;
    // Storage is the same on every level
    const auto &groupdata0 =
        *ghext->patchdata.at(0).leveldata.at(min_level).groupdata.at(gi);
    if (groupdata0.mfab.empty())
      continue;
    const int ntls = groupdata0.mfab.size();
    const int numvars = groupdata0.numvars;
    // Like an INVALIDATES clause (CallFunction): the reason becomes the
    // "why" of the usual invalid-input error, so that a routine that reads
    // a mid-step result learns the rule from the message
    levels.loop_serially([&](auto &restrict leveldata) {
      auto &restrict groupdata = *leveldata.groupdata.at(gi);
      assert(int(groupdata.mfab.size()) == ntls);
      for (int tl = 0; tl < ntls; ++tl)
        for (int vi = 0; vi < numvars; ++vi)
          groupdata.valid.at(tl).at(vi).set_invalid(
              make_valid_all(), [level = leveldata.level] {
                return "Level " + std::to_string(level) +
                       " is mid-step under subcycling: its end-of-step "
                       "results (written in CCTK_POSTSTEP or CCTK_ANALYSIS) "
                       "are invalid until its next end of step (see \"End "
                       "of step under subcycling\" in the CarpetX "
                       "documentation)";
              });
    });
    // A no-op unless CarpetX::poison_undefined_values is set
    for (int tl = 0; tl < ntls; ++tl)
      for (int vi = 0; vi < numvars; ++vi)
        poison_invalid_gf(levels, gi, vi, tl);
  }
}

void check_walked(const cFunctionData *const attribute) {
  assert(attribute);
  assert(ghext->use_subcycling);
  assert(ghext->phase != traversal_phase_t::other);
  if (ghext->walked_functions.count(attribute))
    return;
  CCTK_VERROR("%s::%s (scheduled in %s), called while traversing %s, was not "
              "seen by the end-of-step classification walk. Either it sits in "
              "a schedule group that is traversed only through "
              "CallScheduleGroup and is placed under none of the bins "
              "CCTK_PRESTEP, CCTK_EVOL, CarpetX_PreRestrict, "
              "CCTK_POSTRESTRICT, CCTK_POSTSTEP or CCTK_ANALYSIS (place such "
              "a group like ODESolvers_CalledBySolve in "
              "ODESolvers/schedule.ccl), or the layout of the flesh's "
              "t_attribute no longer matches CarpetX's mirror "
              "(end_of_step.cxx).",
              attribute->thorn, attribute->routine, attribute->where,
              ghext->bin);
}

std::string end_of_step_subject(const cFunctionData *const attribute) {
  assert(attribute);
  std::ostringstream buf;
  buf << "Routine " << attribute->thorn << "::" << attribute->routine;
  // `where` is the immediate schedule.ccl parent: a schedule group, or the
  // bin itself for a routine scheduled AT it
  if (!(ghext->bin && std::strcmp(attribute->where, ghext->bin) == 0))
    buf << " (in group " << attribute->where << ")";
  return buf.str();
}

void error_if_end_of_step_sync(const char *const what, const int n,
                               const int *const groups) {
  assert(ghext->phase == traversal_phase_t::end_of_step);
  assert(ghext->bin);
  assert(what);
  assert(n >= 0);
  std::ostringstream buf;
  for (int i = 0; i < n; ++i) {
    if (i != 0)
      buf << ", ";
    const char *const groupname = CCTK_FullGroupName(groups[i]);
    buf << (groupname ? groupname : "<unknown group>");
  }
  CCTK_VERROR("under subcycling, nothing scheduled in %s may SYNC.\n"
              "  %s: %s\n"
              "  End-of-step routines see the ghosts as they stand. Compute "
              "pointwise diagnostics everywhere (READS the state everywhere, "
              "WRITES everywhere), and leave stencil diagnostics "
              "interior-only, without a SYNC (see \"End of step under "
              "subcycling\" in the CarpetX documentation).",
              ghext->bin, what, buf.str().c_str());
}

} // namespace CarpetX
