#ifndef CARPETX_CARPETX_CONFIG_GROUPS_HXX
#define CARPETX_CARPETX_CONFIG_GROUPS_HXX

#include "internal.hxx"

#include "config/boundaries.hxx"
#include "config/indices.hxx"
#include "util/refuse.hxx"
#include "util/var_type.hxx"
#include <carpetx.hxx>

#include <cctk.h>

#include <array>
#include <cassert>
#include <optional>
#include <source_location>
#include <string>
#include <vector>

namespace CarpetX {

enum class group_kind { grid_function, array, scalar };

struct grid_function_t {
  std::array<int, dim> indextype;
  std::array<int, dim> nghostzones;
  std::array<std::array<boundary_t, dim>, 2> boundaries;
  std::vector<std::array<int, dim>> parities;
  std::vector<CCTK_REAL> dirichlet_values;
  std::vector<CCTK_REAL> robin_values;
  std::optional<std::array<gi_t, dim>> fluxes;
  std::string prolongation_type;
  int prolongation_order;

private:
  grid_function_t(gi_t gi, int numvars);
  friend struct group_declaration_t;
};

struct group_declaration_t {
  std::string groupname;
  gi_t groupindex;
  int firstvarindex;
  int numvars;
  bool do_checkpoint;
  bool do_evolve;
  bool do_restrict;
  bool restrict_on_sync;
  group_kind kind;
  CarpetX::var_type var_type;

  const grid_function_t &grid_function(
      const std::source_location site = std::source_location::current()) const {
    if (kind != group_kind::grid_function)
      refuse(site, "group %s is not a grid function group", groupname.c_str());
    assert(grid_function_);
    return *grid_function_;
  }

private:
  explicit group_declaration_t(gi_t gi);
  std::optional<grid_function_t> grid_function_;
  friend struct GHExt;
};

} // namespace CarpetX

#endif
