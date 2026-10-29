#ifndef CARPETX_CARPETX_IO_SELECTION_HXX
#define CARPETX_CARPETX_IO_SELECTION_HXX

#include "internal.hxx"

#include "hierarchy/driver.hxx"
#include <carpetx.hxx>

#include <functional>
#include <optional>
#include <vector>

namespace CarpetX {

struct group_selection_t {
private:
  struct kind_filter {
    std::optional<group_kind> kind;
    bool operator()(const GHExt::GroupRecord *const record) const {
      return !kind || record->kind == *kind;
    }
  };

public:
  static group_selection_t select_groups(const char *parameter_name,
                                         const char *value);
  static group_selection_t
  select_records(const std::function<bool(const GHExt::GroupRecord &)> &keep);

  bool contains(gi_t gi) const;
  bool empty() const { return selected.empty(); }
  auto records(const std::optional<group_kind> kind = std::nullopt) const {
    return storage_range<
        const GHExt::GroupRecord,
        std::vector<const GHExt::GroupRecord *>::const_iterator, kind_filter>(
        selected.begin(), selected.end(), kind_filter{kind});
  }
  void log(const char *method, std::optional<group_kind> kind) const;
  bool operator==(const group_selection_t &) const = default;

private:
  std::vector<const GHExt::GroupRecord *> selected;
};

} // namespace CarpetX

#endif
