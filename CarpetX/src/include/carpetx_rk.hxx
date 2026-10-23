#ifndef CARPETX_CARPETX_INCLUDE_CARPETX_RK_HXX
#define CARPETX_CARPETX_INCLUDE_CARPETX_RK_HXX

#include <carpetx.hxx>

#include <cctk.h>

#include <memory>
#include <vector>

namespace CarpetX {

enum class rk_dense_output_t { none, ssprk3, rk4 };

constexpr int rk_dense_output_stages(const rk_dense_output_t dense_output) {
  switch (dense_output) {
  case rk_dense_output_t::ssprk3:
    return 3;
  case rk_dense_output_t::rk4:
    return 4;
  default:
    return 0;
  }
}

struct rk_pair_t {
  int state_gi, rhs_gi;
};

struct rk_config_t {
  rk_dense_output_t dense_output = rk_dense_output_t::none;
  std::vector<CCTK_REAL> b;
  std::vector<rk_pair_t> integrated;

  int nstages() const { return b.size(); }
  bool feeds_flux_registers() const { return !b.empty(); }
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

class rk_step_t {
public:
  rk_step_t(rk_step_t &&other) noexcept;
  rk_step_t(const rk_step_t &) = delete;
  rk_step_t &operator=(const rk_step_t &) = delete;
  rk_step_t &operator=(rk_step_t &&) = delete;
  ~rk_step_t();

  void rhs_computed(CCTK_REAL delta_time);
  void state_updated();

private:
  struct impl;
  std::unique_ptr<impl> p;
  explicit rk_step_t(std::unique_ptr<impl> p) noexcept;
  friend rk_step_t open_rk_step(window_t window);
};

rk_step_t open_rk_step(window_t window);

} // namespace CarpetX

#endif
