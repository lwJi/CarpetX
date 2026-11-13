#ifndef CARPETX_CARPETX_HIERARCHY_INTEGRATOR_HXX
#define CARPETX_CARPETX_HIERARCHY_INTEGRATOR_HXX

#include "internal.hxx"

#include "hierarchy/driver.hxx"

#include <carpetx.hxx>

#include <vector>

namespace CarpetX {

bool group_is_integrated(gi_t gi);
stage_range registered_stages();
rk_dense_output_t registered_dense_output();
bool group_has_flux_register(gi_t gi);
void warn_inert_flux_tags();

struct integrated_groups_t {
  std::vector<gi_t> states, rhss;
};

std::vector<gi_t> integrated_states(const GHExt::integrator_t &integrator);
integrated_groups_t
integrated_groups_with_storage(level_t level,
                               const GHExt::integrator_t &integrator);

} // namespace CarpetX

#endif
