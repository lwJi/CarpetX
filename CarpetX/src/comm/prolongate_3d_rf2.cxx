#include "comm/prolongate_3d_rf2.hxx"

#include "hierarchy/group_tags.hxx"

#include <cctk.h>

#include <array>
#include <cassert>
#include <iostream>
#include <map>
#include <string>

namespace CarpetX {

std::ostream &operator<<(std::ostream &os, const centering_t cent) {
  switch (cent) {
  case centering_t::vertex:
    return os << "vertex";
  case centering_t::cell:
    return os << "cell";
  }
  assert(false);
}

std::ostream &operator<<(std::ostream &os, const interpolation_t intp) {
  switch (intp) {
  case interpolation_t::poly:
    return os << "poly";
  case interpolation_t::hermite:
    return os << "hermite";
  case interpolation_t::cons:
    return os << "cons";
  case interpolation_t::eno:
    return os << "eno";
  case interpolation_t::minmod:
    return os << "minmod";
  }
  assert(false);
}

std::ostream &operator<<(std::ostream &os, const fallback_t fb) {
  switch (fb) {
  case fallback_t::none:
    return os << "none";
  case fallback_t::linear:
    return os << "linear";
  }
  assert(false);
}

amrex::Interpolater *get_interpolator(const std::string prolongation_type,
                                      const int prolongation_order,
                                      const std::array<int, dim> indextype) {
  const int indextype_scalar =
      (indextype[0] << 2) | (indextype[1] << 1) | (indextype[2] << 0);

  const std::map<int, std::array<amrex::Interpolater *, 8> > *operators =
      nullptr;

  if (prolongation_type == "interpolate")
    operators = &prolongate_poly_3d_rf2;
  else if (prolongation_type == "conservative")
    operators = &prolongate_cons_3d_rf2;
  else if (prolongation_type == "ddf")
    operators = &prolongate_ddf_3d_rf2;
  else if (prolongation_type == "eno")
    operators = &prolongate_eno_3d_rf2;
  else if (prolongation_type == "minmod")
    operators = &prolongate_minmod_3d_rf2;
  else if (prolongation_type == "hermite")
    operators = &prolongate_hermite_3d_rf2;
  else if (prolongation_type == "natural")
    operators = &prolongate_natural_3d_rf2;
  else if (prolongation_type == "poly-cons3lfb")
    operators = &prolongate_poly_cons3lfb_3d_rf2;
  else if (prolongation_type == "poly-eno3lfb")
    operators = &prolongate_poly_eno3lfb_3d_rf2;
  else
    CCTK_VERROR("Unsupported prolongation type %s", prolongation_type.c_str());

  if (operators->count(prolongation_order) == 0)
    CCTK_VERROR("Unsupported prolongation order %d for prolongation type %s",
                prolongation_order, prolongation_type.c_str());

  return operators->at(prolongation_order).at(indextype_scalar);
}

} // namespace CarpetX
