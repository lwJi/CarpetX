#include "config/boundaries.hxx"

#include <carpetx.hxx>

#include <cassert>
#include <ostream>

namespace CarpetX {

std::ostream &operator<<(std::ostream &os, const symmetry_t symmetry) {
  switch (symmetry) {
  case symmetry_t::none:
    return os << "none";
  case symmetry_t::interpatch:
    return os << "interpatch";
  case symmetry_t::periodic:
    return os << "periodic";
  case symmetry_t::reflection:
    return os << "reflection";
  default:
    assert(0);
  }
  return os;
}

std::ostream &operator<<(std::ostream &os, const boundary_t boundary) {
  switch (boundary) {
  case boundary_t::none:
    return os << "none";
  case boundary_t::symmetry_boundary:
    return os << "symmetry_boundary";
  case boundary_t::dirichlet:
    return os << "dirichlet";
  case boundary_t::linear_extrapolation:
    return os << "linear_extrapolation";
  case boundary_t::neumann:
    return os << "neumann";
  case boundary_t::robin:
    return os << "robin";
  default:
    assert(0);
  }
  return os;
}

} // namespace CarpetX
