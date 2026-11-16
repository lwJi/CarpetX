#include "comm/timelevels.hxx"

#include "hierarchy/model.hxx"
#include "util/refuse.hxx"
#include <carpetx.hxx>

#include <source_location>

namespace CarpetX {

tl_range timelevels_t::range(const GHExt::GroupRecord &record,
                             const std::source_location site) const {
  const int ntls = record.num_timelevels();
  if (!tl_)
    return tl_range{tl_t{0}, tl_t{ntls > 1 ? ntls - 1 : ntls}};
  if (!record.timelevels().contains(*tl_))
    refuse(site,
           "time level %d of group %s does not exist: it has %d time levels",
           tl_->index(), record.groupname.c_str(), ntls);
  return tl_range{*tl_, tl_->next()};
}

} // namespace CarpetX
