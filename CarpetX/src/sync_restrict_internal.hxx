#ifndef CARPETX_CARPETX_SYNC_RESTRICT_INTERNAL_HXX
#define CARPETX_CARPETX_SYNC_RESTRICT_INTERNAL_HXX

#include "internal.hxx"

#include "driver.hxx"

#include <carpetx.hxx>

#include <cctk.h>

#include <optional>
#include <source_location>
#include <span>
#include <vector>

namespace CarpetX {

std::vector<gi_t> collect_restrictable_groups();

struct timelevels_t {
  static constexpr timelevels_t one(const tl_t tl) { return timelevels_t{tl}; }
  static constexpr timelevels_t all_but_oldest() {
    return timelevels_t{std::nullopt};
  }
  tl_range
  range(const GHExt::GroupRecord &record,
        std::source_location site = std::source_location::current()) const;

private:
  constexpr explicit timelevels_t(const std::optional<tl_t> tl) : tl_(tl) {}
  std::optional<tl_t> tl_;
};

struct active_levels_t;

void Restrict(const cGH *cctkGH, int min_level, int max_level,
              std::span<const gi_t> groups);
void ProlongateAfterRestriction(const active_levels_t &active_levels,
                                const cGH *cctkGH);

} // namespace CarpetX

#endif // #ifndef CARPETX_CARPETX_SYNC_RESTRICT_INTERNAL_HXX
