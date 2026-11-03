#include "hierarchy/binding.hxx"
#include "hierarchy/driver.hxx"
#include "hierarchy/loops.hxx"

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

void unbind_array_group(cGH *const cctkGH, const gi_t gi) {
  assert(cctkGH);
  const GHExt::GroupRecord &group = driver().record(gi);
  for (const tl_t tl : group.timelevels())
    for (const vi_t vi : group.variables())
      cctkGH->data[group.var(vi).index()][tl.index()] = nullptr;
}

void assert_array_group_bound(const cGH *const cctkGH, const gi_t gi) {
  assert(cctkGH);
  const GHExt::GroupRecord &group = driver().record(gi);
  for (const tl_t tl : group.timelevels())
    for (const vi_t vi : group.variables())
      assert(cctkGH->data[group.var(vi).index()][tl.index()] ==
             array_ptr(group, tl, vi));
}

void bind_grid_group(cGH *const cctkGH,
                     const GHExt::PatchData::LevelData::GroupData &groupdata,
                     const GridPtrDesc1 &grid1, const int component) {
  assert(cctkGH);
  for (const tl_t tl : groupdata.record.timelevels()) {
    const amrex::Array4<CCTK_REAL> vars =
        groupdata.timelevel(tl).mfab->array(component);
    for (const vi_t vi : groupdata.record.variables())
      cctkGH->data[groupdata.record.var(vi).index()][tl.index()] =
          grid1.ptr(vars, vi);
  }
}

void assert_grid_group_bound(
    const cGH *const cctkGH,
    const GHExt::PatchData::LevelData::GroupData &groupdata,
    const GridPtrDesc1 &grid1, const int component) {
  assert(cctkGH);
  for (const tl_t tl : groupdata.record.timelevels()) {
    const amrex::Array4<CCTK_REAL> vars =
        groupdata.timelevel(tl).mfab->array(component);
    for (const vi_t vi : groupdata.record.variables())
      assert(cctkGH->data[groupdata.record.var(vi).index()][tl.index()] ==
             grid1.ptr(vars, vi));
  }
}

} // namespace

void cctkGH_binding::bind_arrays(cGH *const cctkGH) {
  for (const auto &group : driver().records()) {
    if (group.kind == group_kind::grid_function)
      continue;
    bind_array_group(cctkGH, group.groupindex);
  }
}

void cctkGH_binding::unbind_arrays(cGH *const cctkGH) {
  for (const auto &group : driver().records()) {
    if (group.kind == group_kind::grid_function)
      continue;
    unbind_array_group(cctkGH, group.groupindex);
  }
}

void cctkGH_binding::bind_local(cGH *const cctkGH,
                                const GHExt::PatchData::LevelData &leveldata,
                                const MFPointer &mfp) {
  assert(in_local_mode(cctkGH));
  assert(cctkGH->cctk_component == mfp.index());
  for (const auto &groupdata : leveldata.groups()) {
    const GridPtrDesc1 grid1(leveldata, groupdata, mfp);
    bind_grid_group(cctkGH, groupdata, grid1, mfp.index());
  }
}

void cctkGH_binding::unbind_local(cGH *const cctkGH,
                                  const GHExt::PatchData::LevelData &leveldata,
                                  const MFPointer &mfp) {
  assert(in_local_mode(cctkGH));
  assert(cctkGH->cctk_component == mfp.index());
  for (const auto &groupdata : leveldata.groups())
    for (const tl_t tl : groupdata.record.timelevels())
      for (const vi_t vi : groupdata.record.variables())
        cctkGH->data[groupdata.record.var(vi).index()][tl.index()] = nullptr;
}

void cctkGH_binding::rotate(const window_t window, const gi_t gi) {
  const int ntls0 = driver().record(gi).num_timelevels();
  loop_serially(window, [&](const GHExt::PatchData::LevelData &leveldata) {
    auto &restrict groupdata =
        GHExt::mutable_instance().level(leveldata.level).group(gi);
    const int ntls = groupdata.num_timelevels();
    assert(ntls == ntls0);
    if (ntls > 1) {
      groupdata.rotate_timelevels();
      for (const auto &local_cctkGH_ptr : leveldata.local_cctkGHs) {
        cGH *const local_cctkGH = local_cctkGH_ptr.get();
        const int component = local_cctkGH->cctk_component;
        const GridPtrDesc1 grid1(leveldata, groupdata, component);
        bind_grid_group(local_cctkGH, groupdata, grid1, component);
        assert_grid_group_bound(local_cctkGH, groupdata, grid1, component);
      }
    }
  });
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
