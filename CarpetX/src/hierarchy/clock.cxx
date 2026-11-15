#include "hierarchy/clock.hxx"
#include "hierarchy/model.hxx"

#include <cctk.h>

#include <cassert>

namespace CarpetX {

void publish_clock(const GHExt &ext) {
  const GHExt::run_clock_t &clock = ext.run_clock();
  const auto publish = [&clock](cGH *const cctkGH) {
    assert(cctkGH);
    cctkGH->cctk_iteration = clock.iteration;
    cctkGH->cctk_time = clock.time;
    cctkGH->cctk_delta_time = clock.delta_time;
    cctkGH->cctk_timefac = clock.timefac;
  };
  publish(ext.flesh_cctkGH());
  ext.for_each_cctkGH(publish);
}

} // namespace CarpetX
