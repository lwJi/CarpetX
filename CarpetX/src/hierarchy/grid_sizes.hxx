#ifndef CARPETX_CARPETX_HIERARCHY_GRID_SIZES_HXX
#define CARPETX_CARPETX_HIERARCHY_GRID_SIZES_HXX

#include "internal.hxx"

#include <AMReX_IntVect.H>
#include <AMReX_Vector.H>

namespace CarpetX {

amrex::Vector<amrex::IntVect> per_level_max_grid_sizes();

} // namespace CarpetX

#endif
