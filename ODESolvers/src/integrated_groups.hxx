#ifndef CARPETX_ODESOLVERS_INTEGRATED_GROUPS_HXX
#define CARPETX_ODESOLVERS_INTEGRATED_GROUPS_HXX

#include <carpetx_rk.hxx>

#include <vector>

namespace ODESolvers {

struct integrated_groups_t {
  std::vector<CarpetX::rk_pair_t> pairs;
  std::vector<CarpetX::gi_t> invalidated_by_update;
  int nvars;
};

const integrated_groups_t &integrated_groups();

} // namespace ODESolvers

#endif
