#include "schedule/schedule.hxx"
#include "hierarchy/driver.hxx"
#include "hierarchy/operations.hxx"
#include <carpetx.hxx>

#include <cctk.h>
#include <cctk_Parameters.h>

#include <algorithm>
#include <cassert>
#include <climits>

namespace CarpetX {

namespace {
int GroupStorageCrease(const cGH *cctkGH, int n_groups, const int *groups,
                       const int *requested_tls, int *status, const bool inc) {
  DECLARE_CCTK_PARAMETERS;

  assert(cctkGH);
  assert(n_groups >= 0);
  assert(groups);
  assert(requested_tls);
  const GHExt &ext = driver();

  for (int n = 0; n < n_groups; ++n) {
    if (groups[n] < 0 or groups[n] >= CCTK_NumGroups()) {
      CCTK_VWARN(CCTK_WARN_ALERT, "Group index %d is illegal", groups[n]);
      return -1;
    }
    assert(requested_tls[n] >= 0 or requested_tls[n] == -1);
  }

  int min_num_timelevels = INT_MAX;
  for (int n = 0; n < n_groups; ++n) {
    const gi_t gi{groups[n]};
    const int gid = gi.index();
    const int declared_tls = CCTK_DeclaredTimeLevelsGI(gid);
    int ntls = requested_tls[n];
    if (ntls == -1)
      ntls = declared_tls;
    if (inc and declared_tls < 2 and ntls > declared_tls) {
      CCTK_VWARN(CCTK_WARN_ALERT,
                 "Attempting to activate %d timelevels for group '%s' which "
                 "only has a single timelevel declared in interface.ccl. "
                 "Please declare at least 2 timelevels in interface.ccl to "
                 "allow more timelevels to be created at runtime.",
                 ntls, CCTK_FullGroupName(gid));
      ntls = declared_tls;
    }
    if (ntls > declared_tls)
      ntls = declared_tls;

    const int previous = ext.active_timelevels(gi);
    if (status)
      status[n] = previous;

    if (inc and requested_tls[n] == 0) {
      min_num_timelevels = std::min(min_num_timelevels, previous);
      continue;
    }

    hierarchy_writer::record_storage(gi, ntls);

    const int current = ext.active_timelevels(gi);
    min_num_timelevels = std::min(min_num_timelevels, current);
  }
  if (min_num_timelevels == INT_MAX)
    min_num_timelevels = 0;

  return min_num_timelevels;
}
} // namespace

int GroupStorageIncrease(const cGH *cctkGH, int n_groups, const int *groups,
                         const int *tls, int *status) {
  DECLARE_CCTK_PARAMETERS;

  return GroupStorageCrease(cctkGH, n_groups, groups, tls, status, true);
}

int GroupStorageDecrease(const cGH *cctkGH, int n_groups, const int *groups,
                         const int *tls, int *status) {
  DECLARE_CCTK_PARAMETERS;

  return GroupStorageCrease(cctkGH, n_groups, groups, tls, status, false);
}

int QueryGroupStorageB(const cGH *cctkGH, int gi, const char *groupname) {
  assert(cctkGH);
  const GHExt &ext = driver();
  if (gi < 0) {
    if (!groupname) {
      CCTK_WARN(CCTK_WARN_ALERT,
                "QueryGroupStorageB called with negative group index and "
                "no group name");
      return -1;
    }
    gi = CCTK_GroupIndex(groupname);
    if (gi < 0) {
      CCTK_VWARN(CCTK_WARN_ALERT, "Group '%s' not found", groupname);
      return -1;
    }
  }
  assert(gi < CCTK_NumGroups());
  return ext.record(gi_t{gi}).has_storage() ? 1 : 0;
}

int EnableGroupStorage(const cGH *cctkGH, const char *groupname) {
  const int group = CCTK_GroupIndex(groupname);
  assert(group >= 0 and group < CCTK_NumGroups());
  const int tls = CCTK_DeclaredTimeLevelsGI(group);
  int status;
  GroupStorageIncrease(cctkGH, 1, &group, &tls, &status);
  return status;
}

int DisableGroupStorage(const cGH *cctkGH, const char *groupname) {
  const int group = CCTK_GroupIndex(groupname);
  assert(group >= 0 and group < CCTK_NumGroups());
  const int tls = 0;
  int status;
  GroupStorageDecrease(cctkGH, 1, &group, &tls, &status);
  return status;
}

} // namespace CarpetX
