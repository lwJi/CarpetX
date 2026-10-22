#ifndef CARPETX_CARPETX_INCLUDE_CARPETX_RK_HXX
#define CARPETX_CARPETX_INCLUDE_CARPETX_RK_HXX

#include <carpetx.hxx>

#include <cctk.h>

#include <vector>

namespace CarpetX {

inline constexpr int max_num_rk_stages = 4;

struct rk_pair_t {
  int state_gi, rhs_gi;
};

struct rk_config_t {
  int nstages;
  bool subcycling_ok, reflux_ok;
  std::vector<rk_pair_t> integrated;

  bool is_integrated(const int gi) const {
    for (const rk_pair_t &pair : integrated)
      if (pair.state_gi == gi)
        return true;
    return false;
  }
};

void register_integrator(const rk_config_t &config);

void run_schedule_group_at(const cGH *cctkGH, const char *group, CCTK_REAL time,
                           CCTK_REAL delta_time);

void accumulate_fluxes(level_t level, int stage, CCTK_REAL weight);
void reflux(window_t window);
bool window_is_restricted(window_t window);

} // namespace CarpetX

#endif
