#ifndef CARPETX_ODESOLVERS_STATE_HXX
#define CARPETX_ODESOLVERS_STATE_HXX

#include <carpetx.hxx>

#include <cctk.h>

#include <AMReX_MultiFab.H>

#include <span>
#include <vector>

namespace ODESolvers {

using mfabs_t = std::vector<const amrex::MultiFab *>;

struct grid_state_t {
  std::vector<CarpetX::group_t> groups;

  mfabs_t mfabs() const;
};

class scratch_state_t {
public:
  static scratch_state_t scaled(const grid_state_t &src, CCTK_REAL factor);
  static scratch_state_t copy(const grid_state_t &src) {
    return scaled(src, 1);
  }

  void add(CCTK_REAL factor, const grid_state_t &src);

  mfabs_t mfabs() const;

private:
  scratch_state_t() = default;
  std::vector<amrex::MultiFab *> destinations();

  std::vector<CarpetX::scratch_t> scratch_;
};

struct term_t {
  CCTK_REAL factor;
  mfabs_t mfabs;
};

void update(const grid_state_t &dst, CCTK_REAL scale,
            std::span<const term_t> terms, CarpetX::valid_t leaves_valid);

} // namespace ODESolvers

#endif
