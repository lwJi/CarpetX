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

class rk_step_t;

class rk_stage_t {
public:
  rk_stage_t(rk_stage_t &&other) noexcept;
  rk_stage_t(const rk_stage_t &) = delete;
  rk_stage_t &operator=(const rk_stage_t &) = delete;
  rk_stage_t &operator=(rk_stage_t &&) = delete;
  ~rk_stage_t();

  void rhs_computed(CCTK_REAL flux_weight);
  void fill_next_boundary(CCTK_REAL xsi, CCTK_REAL dtc);

private:
  struct impl;
  impl *p;
  explicit rk_stage_t(impl *p) noexcept;
  friend class rk_step_t;
};

class rk_step_t {
public:
  rk_step_t(rk_step_t &&other) noexcept;
  rk_step_t(const rk_step_t &) = delete;
  rk_step_t &operator=(const rk_step_t &) = delete;
  rk_step_t &operator=(rk_step_t &&) = delete;
  ~rk_step_t();

  rk_stage_t stage(int n);

private:
  struct impl;
  impl *p;
  explicit rk_step_t(impl *p) noexcept;
  friend rk_step_t open_rk_step(level_t level);
  friend class rk_stage_t;
};

rk_step_t open_rk_step(level_t level);
void rk_recover_boundary(level_t level, CCTK_REAL xsi, CCTK_REAL dtc);

void reflux(window_t window);
bool window_is_restricted(window_t window);

} // namespace CarpetX

#endif
