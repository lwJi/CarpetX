#include "io/selection.hxx"

#include "hierarchy/driver.hxx"
#include "util/refuse.hxx"
#include <carpetx.hxx>

#include <cctk.h>
#include <cctk_Parameters.h>

#include <algorithm>
#include <functional>
#include <optional>
#include <set>
#include <source_location>

namespace CarpetX {

group_selection_t
group_selection_t::select_groups(const char *const parameter_name,
                                 const char *const value) {
  std::set<gi_t> named;
  const int num_named = CCTK_TraverseString(
      value,
      [](const int var, const char *, void *const arg) {
        static_cast<std::set<gi_t> *>(arg)->insert(
            driver().record_of(var_t{var}).groupindex);
      },
      &named, CCTK_GROUP_OR_VAR);
  if (num_named < 0)
    refuse(std::source_location::current(),
           "%s is \"%s\", which names a group or variable that does not exist "
           "or carries a malformed option string",
           parameter_name, value);
  return select_records([&](const GHExt::GroupRecord &record) {
    return named.contains(record.groupindex);
  });
}

group_selection_t group_selection_t::select_records(
    const std::function<bool(const GHExt::GroupRecord &)> &keep) {
  group_selection_t selection;
  for (const GHExt::GroupRecord &record : driver().records())
    if (keep(record))
      selection.selected.push_back(&record);
  return selection;
}

bool group_selection_t::contains(const gi_t gi) const {
  const auto found = std::lower_bound(
      selected.begin(), selected.end(), gi,
      [](const GHExt::GroupRecord *const record, const gi_t wanted) {
        return record->groupindex < wanted;
      });
  return found != selected.end() && (*found)->groupindex == gi;
}

void group_selection_t::log(const char *const method,
                            const std::optional<group_kind> kind) const {
  DECLARE_CCTK_PARAMETERS;
  if (!verbose)
    return;
  CCTK_VINFO("%s output for groups:", method);
  for (const GHExt::GroupRecord &record : records(kind))
    CCTK_VINFO("  %s", record.groupname.c_str());
}

} // namespace CarpetX
