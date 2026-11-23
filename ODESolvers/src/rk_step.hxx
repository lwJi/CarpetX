#ifndef CARPETX_ODESOLVERS_RK_STEP_HXX
#define CARPETX_ODESOLVERS_RK_STEP_HXX

#include "rk_methods.hxx"
#include "state.hxx"

#include <carpetx.hxx>

#include <cctk.h>

#include <span>

namespace ODESolvers {

void step(const cGH *cctkGH, const rk_method_t &method,
          CarpetX::window_t window, const grid_state_t &var,
          const grid_state_t &rhs,
          std::span<const CarpetX::gi_t> invalidated_by_update);

} // namespace ODESolvers

#endif
