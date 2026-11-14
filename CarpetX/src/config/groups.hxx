#ifndef CARPETX_CARPETX_CONFIG_GROUPS_HXX
#define CARPETX_CARPETX_CONFIG_GROUPS_HXX

#include "internal.hxx"

#include "config/boundaries.hxx"
#include "config/indices.hxx"
#include <carpetx.hxx>

#include <cctk.h>

#include <array>
#include <optional>
#include <string>
#include <vector>

namespace CarpetX {

enum class group_kind { grid_function, array, scalar };

bool get_group_checkpoint_flag(gi_t gi);
std::optional<bool> get_group_restrict_tag(gi_t gi);
bool get_group_evolve_flag(gi_t gi);

std::string get_group_prolongation_type(gi_t gi);
int get_group_prolongation_order(gi_t gi);

struct grid_function_t {
  std::array<int, dim> indextype;
  std::array<int, dim> nghostzones;
  std::array<std::array<boundary_t, dim>, 2> boundaries;
  std::vector<std::array<int, dim>> parities;
  std::vector<CCTK_REAL> dirichlet_values;
  std::vector<CCTK_REAL> robin_values;
  std::optional<std::array<gi_t, dim>> fluxes;
};
grid_function_t declared_grid_function(gi_t gi, int numvars);

} // namespace CarpetX

#endif
