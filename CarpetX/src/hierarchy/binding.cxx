#include "hierarchy/binding.hxx"
#include "hierarchy/driver.hxx"

#include <cctk.h>

#include <cassert>

namespace CarpetX {

namespace {

void *array_ptr(const GHExt::GroupRecord &group, const tl_t tl, const vi_t vi) {
  const auto &restrict arraygroupdata = group.array();
  return const_cast<void *>(arraygroupdata.timelevel(tl).data.data_at(
      vi.index() * arraygroupdata.array_size));
}

void bind_array_group(cGH *const cctkGH, const gi_t gi) {
  assert(cctkGH);
  const GHExt::GroupRecord &group = driver().record(gi);
  for (const tl_t tl : group.timelevels())
    for (const vi_t vi : group.variables())
      cctkGH->data[group.var(vi).index()][tl.index()] =
          array_ptr(group, tl, vi);
}

void assert_array_group_bound(const cGH *const cctkGH, const gi_t gi) {
  assert(cctkGH);
  const GHExt::GroupRecord &group = driver().record(gi);
  for (const tl_t tl : group.timelevels())
    for (const vi_t vi : group.variables())
      assert(cctkGH->data[group.var(vi).index()][tl.index()] ==
             array_ptr(group, tl, vi));
}

} // namespace

void cctkGH_binding::bind_arrays(cGH *const cctkGH) {
  for (const auto &group : driver().records()) {
    if (group.kind == group_kind::grid_function)
      continue;
    bind_array_group(cctkGH, group.groupindex);
  }
}

void cctkGH_binding::rotate(const gi_t gi) {
  GHExt &ext = GHExt::mutable_instance();
  assert(ext.flesh_cctkGH);
  ext.record(gi).array().rotate_timelevels();
  bind_array_group(ext.flesh_cctkGH, gi);
  ext.for_each_cctkGH([&](cGH *const copy) { bind_array_group(copy, gi); });
  assert_array_group_bound(ext.flesh_cctkGH, gi);
  ext.for_each_cctkGH(
      [&](const cGH *const copy) { assert_array_group_bound(copy, gi); });
}

} // namespace CarpetX
