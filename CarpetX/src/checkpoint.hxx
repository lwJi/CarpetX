#ifndef CARPETX_CARPETX_CHECKPOINT_HXX
#define CARPETX_CARPETX_CHECKPOINT_HXX

#include "internal.hxx"

#include <carpetx.hxx>

#include <cctk.h>

#include <AMReX_Box.H>

#include <optional>
#include <vector>

namespace CarpetX {

struct level_record_t {
  std::vector<amrex::Box> boxes;
  std::optional<rat64> iteration;
  std::optional<int> last_step_iteration;
};

struct checkpoint_record_t {
  int iteration;
  CCTK_REAL time;
  std::vector<std::vector<level_record_t> > levels;
};

bool finer_level_is_behind(const checkpoint_record_t &record, level_t level);

void apply_structure(cGH *cctkGH, const checkpoint_record_t &record);

} // namespace CarpetX

#endif
