#ifndef CARPETX_CARPETX_CONFIG_STAGES_HXX
#define CARPETX_CARPETX_CONFIG_STAGES_HXX

#include "internal.hxx"

#include "config/indices.hxx"
#include <carpetx.hxx>

namespace CarpetX {

inline constexpr int max_num_rk_stages = 4;

struct stage_t : index_t<stage_t> {
  using index_t::index_t;
};

using stage_range = index_range<stage_t>;

constexpr stage_range rk_stages(const int nstages) {
  return stage_range{stage_t{1}, stage_t{nstages + 1}};
}
inline constexpr stage_range all_rk_stages = rk_stages(max_num_rk_stages);

inline constexpr int time_refinement = 2;

constexpr int substeps_per_coarsest_step(const int levels_below) {
  int substeps = 1;
  for (int n = 0; n < levels_below; ++n)
    substeps *= time_refinement;
  return substeps;
}

struct substep_t {
  rat64 begin, end;
  constexpr bool aligned() const { return end == 1; }
};

} // namespace CarpetX

#endif
