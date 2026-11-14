#ifndef CARPETX_CARPETX_CONFIG_BOUNDARIES_HXX
#define CARPETX_CARPETX_CONFIG_BOUNDARIES_HXX

#include "internal.hxx"

#include "config/indices.hxx"
#include <carpetx.hxx>

#include <array>
#include <optional>
#include <ostream>

namespace CarpetX {

enum class boundary_t {
  none,
  symmetry_boundary,
  dirichlet,
  linear_extrapolation,
  neumann,
  robin,
};
std::ostream &operator<<(std::ostream &os, boundary_t boundary);

std::array<std::array<symmetry_t, dim>, 2>
get_symmetries(std::optional<patch_t> patch);
std::array<std::array<boundary_t, dim>, 2> get_default_boundaries();

} // namespace CarpetX

#endif
