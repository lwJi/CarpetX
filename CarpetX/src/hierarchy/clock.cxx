#include "hierarchy/clock.hxx"
#include "hierarchy/model.hxx"

#include <cctk.h>

#include <bit>
#include <cassert>
#include <cstdint>
#include <functional>
#include <string>
#include <type_traits>

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

namespace {
using real_bits_t =
    std::conditional_t<sizeof(CCTK_REAL) == sizeof(std::uint64_t),
                       std::uint64_t, std::uint32_t>;
static_assert(sizeof(real_bits_t) == sizeof(CCTK_REAL));

bool same_bits(const CCTK_REAL shown, const CCTK_REAL held) {
  return std::bit_cast<real_bits_t>(shown) == std::bit_cast<real_bits_t>(held);
}

constexpr const char *clock_stale_reason =
    "something other than CarpetX's clock writer assigned it";
} // namespace

void require_current_clock(const GHExt &ext, const cGH *const cctkGH,
                           const std::function<std::string()> &who) {
  assert(cctkGH);
  const GHExt::run_clock_t &clock = ext.run_clock();
  const auto refuse_int = [&](const char *const field, const int shown,
                              const int held) {
    CCTK_VERROR("%s returned a cGH whose %s is %d, but the run clock's is %d; "
                "%s",
                who().c_str(), field, shown, held, clock_stale_reason);
  };
  const auto refuse_real = [&](const char *const field, const CCTK_REAL shown,
                               const CCTK_REAL held) {
    CCTK_VERROR("%s returned a cGH whose %s is %.17g, but the run clock's is "
                "%.17g; %s",
                who().c_str(), field, double(shown), double(held),
                clock_stale_reason);
  };
  if (cctkGH->cctk_iteration != clock.iteration)
    refuse_int("cctk_iteration", cctkGH->cctk_iteration, clock.iteration);
  if (!same_bits(cctkGH->cctk_time, clock.time))
    refuse_real("cctk_time", cctkGH->cctk_time, clock.time);
  if (!same_bits(cctkGH->cctk_delta_time, clock.delta_time))
    refuse_real("cctk_delta_time", cctkGH->cctk_delta_time, clock.delta_time);
  if (cctkGH->cctk_timefac != clock.timefac)
    refuse_int("cctk_timefac", cctkGH->cctk_timefac, clock.timefac);
}

} // namespace CarpetX
