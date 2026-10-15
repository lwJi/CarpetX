#include "end_of_step.hxx"
#include "driver.hxx"

#include <cctk.h>
#include <cctk_Schedule.h>

#include <cassert>
#include <cstring>
#include <sstream>
#include <string>

namespace CarpetX {

namespace {
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
