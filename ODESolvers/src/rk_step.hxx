#ifndef CARPETX_ODESOLVERS_RK_STEP_HXX
#define CARPETX_ODESOLVERS_RK_STEP_HXX

#include "rk_methods.hxx"
#include "solve.hxx"

#include <carpetx.hxx>

#include <cctk.h>

#include <span>

namespace ODESolvers {

void step(const cGH *cctkGH, const rk_method_t &method,
          CarpetX::window_t window, statecomp_t &var, const statecomp_t &rhs,
          std::span<const CarpetX::gi_t> dep_groups);

} // namespace ODESolvers

#endif
