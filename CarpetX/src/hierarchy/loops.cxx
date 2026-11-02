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

#include <cassert>
#include <functional>

namespace CarpetX {

template <class Kernel>
static void loop_patches_at(const window_t window, const int level_index,
                            const Kernel &kernel) {
  const auto [min_patch, max_patch] = window.patch_bounds();
  for (const patch_t patch :
       index_range<patch_t>{patch_t{min_patch}, patch_t{max_patch}}) {
    const auto &patchdata = driver().patch(patch);
    const level_t level{patch, level_index};
    if (patchdata.level_handles().contains(level))
      kernel(patchdata.level(level));
  }
}

static void assert_consistent_iterations(const window_t window) {
  const auto [min_level, max_level] = window.level_bounds();
  rat64 good_iteration = -1;
  for (int level_index = min_level; level_index < max_level; ++level_index)
    loop_patches_at(window, level_index, [&](const auto &leveldata) {
      if (good_iteration == -1)
        good_iteration = leveldata.iteration;
      assert(leveldata.iteration == good_iteration);
    });
}

void loop_coarse_to_fine(
    const window_t window,
    const std::function<void(const GHExt::PatchData::LevelData &leveldata)>
        &kernel) {
  assert(omp_get_num_threads() == 1);
  assert_consistent_iterations(window);
  const auto [min_level, max_level] = window.level_bounds();
  for (int level_index = min_level; level_index < max_level; ++level_index)
    loop_patches_at(window, level_index, kernel);
}

void loop_fine_to_coarse(
    const window_t window,
    const std::function<void(const GHExt::PatchData::LevelData &leveldata)>
        &kernel) {
  assert(omp_get_num_threads() == 1);
  assert_consistent_iterations(window);
  const auto [min_level, max_level] = window.level_bounds();
  for (int level_index = max_level - 1; level_index >= min_level; --level_index)
    loop_patches_at(window, level_index, kernel);
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
