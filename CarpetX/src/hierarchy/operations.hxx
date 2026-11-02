#ifndef CARPETX_CARPETX_HIERARCHY_OPERATIONS_HXX
#define CARPETX_CARPETX_HIERARCHY_OPERATIONS_HXX

#include "internal.hxx"

#include "hierarchy/driver.hxx"
#include <carpetx.hxx>
#include <rational.hxx>

#include <cctk.h>

#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_IntVect.H>
#include <AMReX_Vector.H>

#include <cstddef>
#include <functional>
#include <optional>
#include <source_location>
#include <string>

namespace CarpetX {

GHExt &create_hierarchy(GHExt::config_t config);
void destroy_hierarchy();

struct hierarchy_writer {
  static const GHExt::PatchData &add_patch(patch_t patch);
  static const GHExt::PatchData::LevelData &
  add_level(level_t level, const amrex::BoxArray &ba,
            const amrex::DistributionMapping &dm,
            const std::function<std::string()> &why, level_buffers_ptr buffers);
  static GHExt::PatchData::LevelData
  remake_level(level_t level, const amrex::BoxArray &ba,
               const amrex::DistributionMapping &dm,
               const std::function<std::string()> &why,
               level_buffers_ptr buffers);
  static void erase_levels_from(level_t level);

  static void
  set_max_grid_size(patch_t patch,
                    const amrex::Vector<amrex::IntVect> &max_grid_sizes);
  static void make_new_grids(patch_t patch, CCTK_REAL time);
  static void mark_initialized();
  static void regrid(level_t base_level, CCTK_REAL time);
  static void set_finest_level(level_t finest_level);
  static void recover_level(level_t level, amrex::BoxArray ba, bool rechop);

  static void record_storage(gi_t gi, int timelevels);
  static void freeze_storage();
  static void set_global_cctkGH(const cGH *cctkGH);
  static void *array_data(gi_t gi, tl_t tl);

  static std::size_t
  acquire_scratch(group_t group,
                  std::source_location site = std::source_location::current());
  static void release_scratch(group_t group, std::size_t slot);

  static void advance_clocks(window_t window);
  static void stamp_last_step(window_t window, int iteration);
  static void stamp_last_finalized(window_t window, int iteration);
  static void restore_level_clock(level_t level, rat64 iteration,
                                  std::optional<int> last_step_iteration);
  static void rotate_timelevels(window_t window, gi_t gi);
  static void rotate_array_timelevels(gi_t gi);
};

} // namespace CarpetX

#endif
