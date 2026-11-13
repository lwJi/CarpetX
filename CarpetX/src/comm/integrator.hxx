#ifndef CARPETX_CARPETX_COMM_INTEGRATOR_HXX
#define CARPETX_CARPETX_COMM_INTEGRATOR_HXX

#include "internal.hxx"

#include "hierarchy/driver.hxx"
#include <carpetx.hxx>

#include <vector>

namespace CarpetX {

bool group_is_integrated(gi_t gi);
stage_range registered_stages();
rk_dense_output_t registered_dense_output();

struct integrated_groups_t {
  std::vector<gi_t> states, rhss;
};

struct rk_level_t {
  level_t level;
  integrated_groups_t groups;
};

std::vector<gi_t> integrated_states(const GHExt::integrator_t &integrator);
integrated_groups_t
integrated_groups_with_storage(level_t level,
                               const GHExt::integrator_t &integrator);
std::vector<rk_level_t>
integrated_levels(window_t window, const GHExt::integrator_t &integrator);

} // namespace CarpetX

#endif
