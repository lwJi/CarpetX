#include "integrated_groups.hxx"

#include <cctk.h>
#include <util_Table.h>

#include <algorithm>
#include <cassert>
#include <cstring>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

namespace ODESolvers {

namespace {

std::string group_tag(const CarpetX::gi_t gi, const char *const key) {
  const int tags = CCTK_GroupTagsTableI(gi.index());
  assert(tags >= 0);
  std::vector<char> buf(1000);
  const int iret = Util_TableGetString(tags, buf.size(), buf.data(), key);
  if (iret == UTIL_ERROR_TABLE_NO_SUCH_KEY)
    return std::string();
  assert(iret >= 0);
  return std::string(buf.data());
}

std::optional<CarpetX::gi_t> group_named(const CarpetX::gi_t relative_to,
                                         std::string name) {
  if (name.find(':') == std::string::npos) {
    const char *const thorn_or_impl =
        CCTK_GroupImplementationI(relative_to.index());
    assert(thorn_or_impl);
    const char *const impl = CCTK_ThornImplementation(thorn_or_impl);
    const char *const thorn = CCTK_ImplementationThorn(thorn_or_impl);
    assert(impl || thorn);
    assert(!impl || !thorn || std::strcmp(impl, thorn) == 0);
    name = std::string(impl ? impl : thorn) + "::" + name;
  }
  const int gi = CCTK_GroupIndex(name.c_str());
  if (gi < 0)
    return std::nullopt;
  return CarpetX::gi_t{gi};
}

std::optional<CarpetX::gi_t> rhs_of(const CarpetX::gi_t gi) {
  const std::string name = group_tag(gi, "rhs");
  if (name.empty())
    return std::nullopt;
  const std::optional<CarpetX::gi_t> rhs = group_named(gi, name);
  if (!rhs)
    CCTK_VERROR("Variable group \"%s\" declares a RHS group \"%s\". "
                "That group does not exist.",
                CCTK_FullGroupName(gi.index()), name.c_str());
  assert(*rhs != gi);
  return rhs;
}

std::vector<CarpetX::gi_t> dependents_of(const CarpetX::gi_t gi) {
  std::vector<CarpetX::gi_t> dependents;
  std::istringstream names(group_tag(gi, "dependents"));
  std::string name;
  while (names >> name) {
    const std::optional<CarpetX::gi_t> dependent = group_named(gi, name);
    if (!dependent)
      CCTK_VERROR("Variable group \"%s\" declares a dependent group \"%s\". "
                  "That group does not exist.",
                  CCTK_FullGroupName(gi.index()), name.c_str());
    dependents.push_back(*dependent);
  }
  return dependents;
}

integrated_groups_t find_integrated_groups() {
  integrated_groups_t found{};
  std::vector<CarpetX::gi_t> rhs_groups;
  for (int gi0 = 0; gi0 < CCTK_NumGroups(); ++gi0) {
    if (CCTK_GroupTypeI(gi0) != CCTK_GF)
      continue;
    const CarpetX::gi_t gi{gi0};
    const std::optional<CarpetX::gi_t> rhs = rhs_of(gi);
    if (!rhs)
      continue;
    assert(CCTK_NumVarsInGroupI(rhs->index()) == CCTK_NumVarsInGroupI(gi0));
    found.pairs.push_back(CarpetX::rk_pair_t{gi, *rhs});
    found.nvars += CCTK_NumVarsInGroupI(gi0);
    rhs_groups.push_back(*rhs);
    const std::vector<CarpetX::gi_t> dependents = dependents_of(gi);
    found.invalidated_by_update.insert(found.invalidated_by_update.end(),
                                       dependents.begin(), dependents.end());
  }

  std::sort(rhs_groups.begin(), rhs_groups.end());
  assert(std::adjacent_find(rhs_groups.begin(), rhs_groups.end()) ==
         rhs_groups.end());

  std::vector<CarpetX::gi_t> &invalidated = found.invalidated_by_update;
  invalidated.insert(invalidated.end(), rhs_groups.begin(), rhs_groups.end());
  std::sort(invalidated.begin(), invalidated.end());
  invalidated.erase(std::unique(invalidated.begin(), invalidated.end()),
                    invalidated.end());

  for (const CarpetX::rk_pair_t &pair : found.pairs)
    assert(!std::binary_search(invalidated.begin(), invalidated.end(),
                               pair.state));
  return found;
}

} // namespace

const integrated_groups_t &integrated_groups() {
  static const integrated_groups_t groups = find_integrated_groups();
  return groups;
}

} // namespace ODESolvers
