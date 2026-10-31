#include "hierarchy/loops.hxx"
#include "hierarchy/driver.hxx"
#include "util/task_manager.hxx"

#include <cctk.h>

#include <AMReX_Gpu.H>
#include <AMReX_MFIter.H>

#if defined _OPENMP
#include <omp.h>
#elif defined __HIPCC__
#define omp_get_max_threads() 1
#define omp_get_num_threads() 1
#define omp_get_thread_num() 0
#define omp_in_parallel() 0
#else
static inline int omp_get_max_threads() { return 1; }
static inline int omp_get_num_threads() { return 1; }
static inline int omp_get_thread_num() { return 0; }
static inline int omp_in_parallel() { return 0; }
#endif

#include <algorithm>
#include <cassert>
#include <functional>
#include <vector>

namespace CarpetX {

template <class Order>
static std::vector<level_t> window_levels(const window_t window,
                                          const Order order) {
  const auto [min_level, max_level] = window.level_bounds();
  const auto [min_patch, max_patch] = window.patch_bounds();
  std::vector<level_t> levels;
  if (min_level >= max_level)
    return levels;
  for (const patch_t patch :
       index_range<patch_t>(patch_t{min_patch}, patch_t{max_patch}))
    for (const level_t level : driver().patch(patch).level_handles())
      if (min_level <= level.index() && level.index() < max_level)
        levels.push_back(level);
  std::stable_sort(levels.begin(), levels.end(),
                   [&](const level_t a, const level_t b) {
                     return order(a.index(), b.index());
                   });
  return levels;
}

static void assert_consistent_iterations(const std::vector<level_t> &levels) {
  rat64 good_iteration = -1;
  for (const level_t level : levels) {
    const auto &iteration = driver().level(level).iteration;
    if (good_iteration == -1)
      good_iteration = iteration;
    assert(iteration == good_iteration);
  }
}

void loop_coarse_to_fine(
    const window_t window,
    const std::function<void(const GHExt::PatchData::LevelData &leveldata)>
        &kernel) {
  assert(omp_get_num_threads() == 1);
  const std::vector<level_t> levels = window_levels(window, std::less<int>());
  assert_consistent_iterations(levels);
  for (const level_t level : levels)
    kernel(driver().level(level));
}

void loop_fine_to_coarse(
    const window_t window,
    const std::function<void(const GHExt::PatchData::LevelData &leveldata)>
        &kernel) {
  assert(omp_get_num_threads() == 1);
  const std::vector<level_t> levels =
      window_levels(window, std::greater<int>());
  assert_consistent_iterations(levels);
  for (const level_t level : levels)
    kernel(driver().level(level));
}

void loop_parallel(
    const window_t window,
    const std::function<void(const GHExt::PatchData::LevelData &leveldata,
                             int index, int component, const cGH *cctkGH)>
        &kernel) {
  assert(omp_get_num_threads() == 1);
  task_manager tasks;

  loop_coarse_to_fine(window, [&](auto &restrict leveldata) {
    int component = 0;
    const auto mfitinfo = amrex::MFItInfo().DisableDeviceSync().EnableTiling();
    const auto &fab0 = *leveldata.fab;
    for (amrex::MFIter mfi(fab0, mfitinfo); mfi.isValid(); ++mfi, ++component) {
      const int index = mfi.index();
      tasks.submit_serially([&kernel, &leveldata, index, component]() {
        cGH *restrict const localGH = leveldata.get_local_cctkGH(component);
        kernel(leveldata, index, component, localGH);
      });
    }
  });

  tasks.run_tasks();
}

void loop_serially(
    const window_t window,
    const std::function<void(const GHExt::PatchData::LevelData &leveldata,
                             int index, int component, const cGH *cctkGH)>
        &kernel) {
  loop_coarse_to_fine(window, [&](auto &restrict leveldata) {
    int component = 0;
    const auto mfitinfo = amrex::MFItInfo().DisableDeviceSync().EnableTiling();
    const auto &fab0 = *leveldata.fab;
    for (amrex::MFIter mfi(fab0, mfitinfo); mfi.isValid(); ++mfi, ++component) {
      const int index = mfi.index();
      cGH *restrict const localGH = leveldata.get_local_cctkGH(component);
      kernel(leveldata, index, component, localGH);
    }
  });
}

void synchronize() {
#ifdef AMREX_USE_GPU
  amrex::Gpu::streamSynchronizeAll();
  AMREX_GPU_ERROR_CHECK();
#endif
}

void update_cctkGHs(cGH *restrict const cctkGH) {
  const GHExt &ext = driver();
  update_cctkGH(ext.global_cctkGH.get(), cctkGH);
  for (auto &restrict level_cctkGH : ext.level_cctkGHs) {
    update_cctkGH(level_cctkGH.get(), cctkGH);
  }
  for (auto &patch : ext.patches()) {
    for (auto &restrict level : patch.levels()) {
      update_cctkGH(level.patch_cctkGH.get(), cctkGH);
    }
  }
  for (auto &patch : ext.patches()) {
    for (auto &restrict level : patch.levels()) {
      for (auto &restrict local_cctkGH : level.local_cctkGHs) {
        update_cctkGH(local_cctkGH.get(), cctkGH);
      }
    }
  }
}

} // namespace CarpetX
