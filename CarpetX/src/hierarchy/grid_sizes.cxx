#include "hierarchy/grid_sizes.hxx"

#include <cctk.h>
#include <cctk_Parameters.h>

#include <AMReX_IntVect.H>
#include <AMReX_Vector.H>

#include <type_traits>

namespace CarpetX {

namespace {

constexpr int max_parameter_levels =
    std::extent_v<decltype(PRIVATE_CARPETX_STRUCT.max_grid_sizes_x)>;
static_assert(max_parameter_levels ==
              std::extent_v<decltype(PRIVATE_CARPETX_STRUCT.max_grid_sizes_y)>);
static_assert(max_parameter_levels ==
              std::extent_v<decltype(PRIVATE_CARPETX_STRUCT.max_grid_sizes_z)>);

int size_or_default(const CCTK_INT size, const CCTK_INT default_size) {
  return size == -1 ? default_size : size;
}

} // namespace

amrex::Vector<amrex::IntVect> per_level_max_grid_sizes() {
  DECLARE_CCTK_PARAMETERS;

  if (max_num_levels > max_parameter_levels)
    CCTK_VERROR("CarpetX::max_num_levels is %d, but CarpetX::max_grid_sizes_x, "
                "max_grid_sizes_y and max_grid_sizes_z have %d entries, one "
                "per level: run with at most %d levels, or enlarge the three "
                "arrays in CarpetX's param.ccl",
                int(max_num_levels), max_parameter_levels,
                max_parameter_levels);

  amrex::Vector<amrex::IntVect> sizes;
  sizes.reserve(max_num_levels);
  for (int n = 0; n < max_num_levels; ++n)
    sizes.push_back(
        amrex::IntVect{size_or_default(max_grid_sizes_x[n], max_grid_size_x),
                       size_or_default(max_grid_sizes_y[n], max_grid_size_y),
                       size_or_default(max_grid_sizes_z[n], max_grid_size_z)});
  return sizes;
}

} // namespace CarpetX
