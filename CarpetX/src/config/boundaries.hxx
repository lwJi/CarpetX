#ifndef CARPETX_CARPETX_CONFIG_BOUNDARIES_HXX
#define CARPETX_CARPETX_CONFIG_BOUNDARIES_HXX

#include "internal.hxx"

#include "config/indices.hxx"
#include <carpetx.hxx>

#include <array>
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

template <typename T> using faces_t = std::array<std::array<T, dim>, 2>;

} // namespace CarpetX

#endif
