#include "hierarchy/integrator.hxx"

#include "hierarchy/driver.hxx"
#include "util/refuse.hxx"

#include <cctk.h>
#include <cctk_Parameters.h>

#include <cassert>
#include <optional>
#include <source_location>
#include <vector>

namespace CarpetX {

bool group_is_integrated(const gi_t gi) {
  assert(gi.index() >= 0 && gi.index() < CCTK_NumGroups());
  const std::optional<GHExt::integrator_t> &integrator = driver().integrator();
  return integrator && integrator->rhs.contains(gi);
}

stage_range registered_stages() {
  const std::optional<GHExt::integrator_t> &integrator = driver().integrator();
  return integrator ? integrator->stages() : all_rk_stages;
}

rk_dense_output_t registered_dense_output() {
  const std::optional<GHExt::integrator_t> &integrator = driver().integrator();
  return integrator ? integrator->dense_output : rk_dense_output_t::none;
}

static bool group_has_flux_tag(const GHExt::GroupRecord &record) {
  return record.kind == group_kind::grid_function &&
         record.grid_function().fluxes.has_value();
}

bool group_has_flux_register(const gi_t gi) {
  const GHExt &ext = driver();
  return ext.do_reflux && group_is_integrated(gi) &&
         group_has_flux_tag(ext.record(gi));
}

void warn_inert_flux_tags() {
  DECLARE_CCTK_PARAMETERS;
  const GHExt &ext = driver();
  const std::optional<GHExt::integrator_t> &integrator = ext.integrator();
  if (max_num_levels <= 1)
    return;
  if (CCTK_MyProc(nullptr) != 0)
    return;
  if (!ext.do_reflux) {
    for (const gi_t gi : index_range<gi_t>(gi_t{0}, gi_t{CCTK_NumGroups()})) {
      if (integrator && integrator->rhs.contains(gi) &&
          group_has_flux_tag(ext.record(gi))) {
        CCTK_INFO("Grid functions integrated by ODESolvers carry fluxes= "
                  "tags, but refluxing at coarse-fine boundaries is off; set "
                  "CarpetX::do_reflux = yes to enable it");
        break;
      }
    }
    return;
  }
  for (const gi_t gi : index_range<gi_t>(gi_t{0}, gi_t{CCTK_NumGroups()})) {
    if (!group_has_flux_tag(ext.record(gi)) || group_has_flux_register(gi))
      continue;
    CCTK_VWARN(CCTK_WARN_ALERT,
               "Group \"%s\" has a fluxes= tag but is not integrated by "
               "ODESolvers: no flux register is allocated for it, and it is "
               "not refluxed at coarse-fine boundaries (see "
               "CarpetX::do_reflux)",
               CCTK_FullGroupName(gi.index()));
  }
}

std::vector<gi_t> integrated_states(const GHExt::integrator_t &integrator) {
  std::vector<gi_t> states;
  for (const auto &[state, rhs] : integrator.rhs)
    states.push_back(state);
  return states;
}

integrated_groups_t
integrated_groups_with_storage(const level_t level,
                               const GHExt::integrator_t &integrator) {
  const GHExt &ext = driver();
  ext.level(level);
  integrated_groups_t groups;
  for (const auto &[state, rhs] : integrator.rhs) {
    if (!ext.record(state).has_storage())
      continue;
    if (!ext.record(rhs).has_storage())
      refuse(std::source_location::current(),
             "integrated group %s has storage on patch %d, refinement level "
             "%d, but its right-hand side %s has none",
             CCTK_FullGroupName(state.index()), level.patch().index(),
             level.index(), CCTK_FullGroupName(rhs.index()));
    groups.states.push_back(state);
    groups.rhss.push_back(rhs);
  }
  return groups;
}

} // namespace CarpetX
