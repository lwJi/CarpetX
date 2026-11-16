#ifndef CARPETX_CARPETX_COMM_TIMELEVELS_HXX
#define CARPETX_CARPETX_COMM_TIMELEVELS_HXX

#include "internal.hxx"

#include "hierarchy/model.hxx"

#include <carpetx.hxx>

#include <optional>
#include <source_location>

namespace CarpetX {

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

} // namespace CarpetX

#endif
