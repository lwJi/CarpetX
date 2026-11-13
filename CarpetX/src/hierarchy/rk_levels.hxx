#ifndef CARPETX_CARPETX_HIERARCHY_RK_LEVELS_HXX
#define CARPETX_CARPETX_HIERARCHY_RK_LEVELS_HXX

#include "internal.hxx"

#include "hierarchy/driver.hxx"
#include "hierarchy/integrator.hxx"

#include <carpetx.hxx>

#include <vector>

namespace CarpetX {

struct rk_level_t {
  level_t level;
  integrated_groups_t groups;
};

std::vector<rk_level_t>
integrated_levels(window_t window, const GHExt::integrator_t &integrator);

} // namespace CarpetX

#endif
