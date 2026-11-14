#include "config/boundaries.hxx"

#include <carpetx.hxx>

#include <cctk.h>
#include <cctk_Parameters.h>

#include <array>
#include <cassert>
#include <optional>
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

std::array<std::array<symmetry_t, dim>, 2>
get_symmetries(const std::optional<patch_t> patch) {
  DECLARE_CCTK_PARAMETERS;

  std::array<std::array<bool, 3>, 2> is_interpatch{
      {{{false, false, false}}, {{false, false, false}}}};
  if (patch && CCTK_IsFunctionAliased("MultiPatch_GetBoundarySpecification2")) {
    CCTK_INT is_interpatch_boundary[2 * dim];
    const int ierr = MultiPatch_GetBoundarySpecification2(
        patch->index(), 2 * dim, is_interpatch_boundary);
    assert(!ierr);
    for (int f = 0; f < 2; ++f)
      for (int d = 0; d < dim; ++d)
        is_interpatch[f][d] = is_interpatch_boundary[2 * d + f];
  }
  const std::array<std::array<bool, 3>, 2> is_periodic{{
      {{bool(periodic && periodic_x), bool(periodic && periodic_y),
        bool(periodic && periodic_z)}},
      {{bool(periodic && periodic_x), bool(periodic && periodic_y),
        bool(periodic && periodic_z)}},
  }};
  const std::array<std::array<bool, 3>, 2> is_reflection{{
      {{bool(reflection_x), bool(reflection_y), bool(reflection_z)}},
      {{bool(reflection_upper_x), bool(reflection_upper_y),
        bool(reflection_upper_z)}},
  }};

  for (int f = 0; f < 2; ++f)
    for (int d = 0; d < dim; ++d)
      assert(is_interpatch[f][d] + is_periodic[f][d] + is_reflection[f][d] <=
             1);

  std::array<std::array<symmetry_t, dim>, 2> symmetries;
  for (int f = 0; f < 2; ++f)
    for (int d = 0; d < dim; ++d)
      symmetries[f][d] = is_interpatch[f][d]   ? symmetry_t::interpatch
                         : is_periodic[f][d]   ? symmetry_t::periodic
                         : is_reflection[f][d] ? symmetry_t::reflection
                                               : symmetry_t::none;

  return symmetries;
}

std::array<std::array<boundary_t, dim>, 2> get_default_boundaries() {
  DECLARE_CCTK_PARAMETERS;

  const std::array<std::array<symmetry_t, dim>, 2> symmetries =
      get_symmetries(std::nullopt);
  std::array<std::array<bool, 3>, 2> is_symmetry;
  for (int f = 0; f < 2; ++f)
    for (int d = 0; d < dim; ++d)
      is_symmetry[f][d] = symmetries[f][d] != symmetry_t::none;

  const std::array<std::array<bool, 3>, 2> is_dirichlet{{
      {{
          bool(CCTK_EQUALS(boundary_x, "dirichlet")),
          bool(CCTK_EQUALS(boundary_y, "dirichlet")),
          bool(CCTK_EQUALS(boundary_z, "dirichlet")),
      }},
      {{
          bool(CCTK_EQUALS(boundary_upper_x, "dirichlet")),
          bool(CCTK_EQUALS(boundary_upper_y, "dirichlet")),
          bool(CCTK_EQUALS(boundary_upper_z, "dirichlet")),
      }},
  }};
  const std::array<std::array<bool, 3>, 2> is_linear_extrapolation{{
      {{
          bool(CCTK_EQUALS(boundary_x, "linear extrapolation")),
          bool(CCTK_EQUALS(boundary_y, "linear extrapolation")),
          bool(CCTK_EQUALS(boundary_z, "linear extrapolation")),
      }},
      {{
          bool(CCTK_EQUALS(boundary_upper_x, "linear extrapolation")),
          bool(CCTK_EQUALS(boundary_upper_y, "linear extrapolation")),
          bool(CCTK_EQUALS(boundary_upper_z, "linear extrapolation")),
      }},
  }};
  const std::array<std::array<bool, 3>, 2> is_neumann{{
      {{
          bool(CCTK_EQUALS(boundary_x, "neumann")),
          bool(CCTK_EQUALS(boundary_y, "neumann")),
          bool(CCTK_EQUALS(boundary_z, "neumann")),
      }},
      {{
          bool(CCTK_EQUALS(boundary_upper_x, "neumann")),
          bool(CCTK_EQUALS(boundary_upper_y, "neumann")),
          bool(CCTK_EQUALS(boundary_upper_z, "neumann")),
      }},
  }};
  const std::array<std::array<bool, 3>, 2> is_robin{{
      {{
          bool(CCTK_EQUALS(boundary_x, "robin")),
          bool(CCTK_EQUALS(boundary_y, "robin")),
          bool(CCTK_EQUALS(boundary_z, "robin")),
      }},
      {{
          bool(CCTK_EQUALS(boundary_upper_x, "robin")),
          bool(CCTK_EQUALS(boundary_upper_y, "robin")),
          bool(CCTK_EQUALS(boundary_upper_z, "robin")),
      }},
  }};
  for (int f = 0; f < 2; ++f)
    for (int d = 0; d < dim; ++d)
      assert(is_symmetry[f][d] + is_dirichlet[f][d] +
                 is_linear_extrapolation[f][d] + is_neumann[f][d] +
                 is_robin[f][d] <=
             1);

  std::array<std::array<boundary_t, dim>, 2> boundaries;
  for (int f = 0; f < 2; ++f)
    for (int d = 0; d < dim; ++d)
      boundaries[f][d] = is_symmetry[f][d]    ? boundary_t::symmetry_boundary
                         : is_dirichlet[f][d] ? boundary_t::dirichlet
                         : is_linear_extrapolation[f][d]
                             ? boundary_t::linear_extrapolation
                         : is_neumann[f][d] ? boundary_t::neumann
                         : is_robin[f][d]   ? boundary_t::robin
                                            : boundary_t::none;

  return boundaries;
}

} // namespace CarpetX
