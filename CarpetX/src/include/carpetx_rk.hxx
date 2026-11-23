#ifndef CARPETX_CARPETX_INCLUDE_CARPETX_RK_HXX
#define CARPETX_CARPETX_INCLUDE_CARPETX_RK_HXX

#include <carpetx.hxx>

#include <cctk.h>

#include <array>
#include <memory>
#include <utility>
#include <vector>

namespace CarpetX {

struct rk_dense_output_method_t {
  int nstages, order;
  std::array<std::array<CCTK_REAL, 4>, 4> a;
  std::array<CCTK_REAL, 4> b, c;
};

constexpr rk_dense_output_method_t
rk_dense_output_method(const rk_dense_output_t dense_output) {
  switch (dense_output) {
  case rk_dense_output_t::ssprk3:
    return {3,
            3,
            {{{}, {1}, {1.0 / 4, 1.0 / 4}}},
            {1.0 / 6, 1.0 / 6, 2.0 / 3},
            {0, 1, 1.0 / 2}};
  case rk_dense_output_t::rk4:
    return {4,
            4,
            {{{}, {1.0 / 2}, {0, 1.0 / 2}, {0, 0, 1}}},
            {1.0 / 6, 1.0 / 3, 1.0 / 3, 1.0 / 6},
            {0, 1.0 / 2, 1.0 / 2, 1}};
  default:
    return {};
  }
}

constexpr int rk_dense_output_stages(const rk_dense_output_t dense_output) {
  return rk_dense_output_method(dense_output).nstages;
}

struct rk_pair_t {
  gi_t state, rhs;
};

struct rk_config_t {
  static rk_config_t with_dense_output(const rk_dense_output_t dense_output,
                                       std::vector<rk_pair_t> integrated) {
    const rk_dense_output_method_t method =
        rk_dense_output_method(dense_output);
    return {dense_output,
            std::vector<CCTK_REAL>(method.b.begin(),
                                   method.b.begin() + method.nstages),
            std::move(integrated)};
  }
  static rk_config_t with_weights(std::vector<CCTK_REAL> b,
                                  std::vector<rk_pair_t> integrated) {
    return {rk_dense_output_t::none, std::move(b), std::move(integrated)};
  }

  rk_dense_output_t dense_output = rk_dense_output_t::none;
  std::vector<CCTK_REAL> b;
  std::vector<rk_pair_t> integrated;

  int nstages() const { return b.size(); }
  bool feeds_flux_registers() const { return !b.empty(); }
  bool is_integrated(const gi_t gi) const {
    for (const rk_pair_t &pair : integrated)
      if (pair.state == gi)
        return true;
    return false;
  }
};

void register_integrator(const rk_config_t &config);

void run_schedule_group_at(const cGH *cctkGH, const char *group, CCTK_REAL time,
                           CCTK_REAL delta_time);

class [[nodiscard]] rk_step_t {
public:
  rk_step_t(rk_step_t &&other) noexcept;
  rk_step_t(const rk_step_t &) = delete;
  rk_step_t &operator=(const rk_step_t &) = delete;
  rk_step_t &operator=(rk_step_t &&) = delete;
  ~rk_step_t();

  void rhs_computed();
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
