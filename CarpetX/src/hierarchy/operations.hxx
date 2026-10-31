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

GHExt &mutable_driver();

GHExt &create_hierarchy(GHExt::config_t config);
void destroy_hierarchy();

const GHExt::PatchData &add_patch(patch_t patch);
const GHExt::PatchData::LevelData &
add_level(level_t level, const amrex::BoxArray &ba,
          const amrex::DistributionMapping &dm,
          const std::function<std::string()> &why);
GHExt::PatchData::LevelData
remake_level(level_t level, const amrex::BoxArray &ba,
             const amrex::DistributionMapping &dm,
             const std::function<std::string()> &why);
void erase_levels_from(level_t level);

void set_max_grid_size(patch_t patch,
                       const amrex::Vector<amrex::IntVect> &max_grid_sizes);
void make_new_grids(patch_t patch, CCTK_REAL time);
void mark_initialized();
void regrid(level_t base_level, CCTK_REAL time);
void set_finest_level(level_t finest_level);
void recover_level(level_t level, amrex::BoxArray ba, bool rechop);

void record_storage(gi_t gi, int timelevels);
void freeze_storage();
void set_global_cctkGH(const cGH *cctkGH);
void *array_data(gi_t gi, tl_t tl);

std::size_t
acquire_scratch(group_t group,
                std::source_location site = std::source_location::current());
void release_scratch(group_t group, std::size_t slot);

void advance_clocks(window_t window);
void stamp_last_step(window_t window, int iteration);
void stamp_last_finalized(window_t window, int iteration);
void restore_level_clock(level_t level, rat64 iteration,
                         std::optional<int> last_step_iteration);
void rotate_timelevels(window_t window, gi_t gi);
void rotate_array_timelevels(gi_t gi);

} // namespace CarpetX

#endif
