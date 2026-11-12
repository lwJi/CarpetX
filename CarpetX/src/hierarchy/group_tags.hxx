#ifndef CARPETX_CARPETX_HIERARCHY_GROUP_TAGS_HXX
#define CARPETX_CARPETX_HIERARCHY_GROUP_TAGS_HXX

#include "internal.hxx"

#include "hierarchy/driver.hxx"

#include <carpetx.hxx>

#include <cctk.h>

#include <array>
#include <optional>
#include <string>
#include <vector>

namespace CarpetX {

std::array<std::array<symmetry_t, dim>, 2>
get_symmetries(std::optional<patch_t> patch);
std::array<std::array<boundary_t, dim>, 2> get_default_boundaries();

bool get_group_checkpoint_flag(gi_t gi);
std::optional<bool> get_group_restrict_tag(gi_t gi);
bool get_group_evolve_flag(gi_t gi);

bool group_is_integrated(gi_t gi);
stage_range registered_stages();
rk_dense_output_t registered_dense_output();
bool group_has_flux_register(gi_t gi);
void warn_inert_flux_tags();

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

std::string get_group_prolongation_type(gi_t gi);
int get_group_prolongation_order(gi_t gi);

} // namespace CarpetX

#endif
