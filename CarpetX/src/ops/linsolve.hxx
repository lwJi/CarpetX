#ifndef CARPETX_CARPETX_OPS_LINSOLVE_HXX
#define CARPETX_CARPETX_OPS_LINSOLVE_HXX

#include "internal.hxx"

#include <carpetx.hxx>

#include <cctk.h>

#include <optional>

namespace CarpetX {

void solve_poisson(gi_t gi_sol, gi_t gi_rhs, std::optional<gi_t> gi_res,
                   CCTK_REAL reltol, CCTK_REAL abstol,
                   CCTK_REAL *restrict res_initial,
                   CCTK_REAL *restrict res_final);

} // namespace CarpetX

#endif
