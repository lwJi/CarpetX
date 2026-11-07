#ifndef CARPETX_CARPETX_OPS_REDUCTION_HXX
#define CARPETX_CARPETX_OPS_REDUCTION_HXX

#include "internal.hxx"

#include <carpetx.hxx>

#include <defs.hxx>
#include <vect.hxx>

#include <cctk.h>

#include <mpi.h>

#include <cmath>
#include <ostream>

namespace CarpetX {

template <typename T, int D> struct combine {
  using value_type = typename reduction<T, D>::tuple_type;
  constexpr AMREX_GPU_DEVICE value_type init() const {
    return (value_type)reduction<T, D>();
  }
  constexpr AMREX_GPU_DEVICE value_type operator()(const value_type &x,
                                                   const value_type &y) const {
    return (value_type)reduction<T, D>(reduction<T, D>(x), reduction<T, D>(y));
  }
};

typedef reduction<CCTK_REAL, dim> reduction_CCTK_REAL;
#ifndef __NVCOMPILER
#pragma omp declare reduction(reduction:reduction_CCTK_REAL : omp_out += omp_in)
#endif

MPI_Datatype reduction_mpi_datatype_CCTK_REAL();
MPI_Op reduction_mpi_op();

} // namespace CarpetX

#endif
