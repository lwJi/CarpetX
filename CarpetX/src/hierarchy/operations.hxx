#ifndef CARPETX_CARPETX_HIERARCHY_OPERATIONS_HXX
#define CARPETX_CARPETX_HIERARCHY_OPERATIONS_HXX

#include "internal.hxx"

#include "hierarchy/driver.hxx"
#include <carpetx.hxx>

#include <cctk.h>

#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_IntVect.H>
#include <AMReX_Vector.H>

#include <cstddef>
#include <functional>
#include <memory>
#include <optional>
#include <source_location>
#include <string>

namespace CarpetX {

struct hierarchy_writers_t;
[[nodiscard]] std::unique_ptr<hierarchy_writers_t>
create_hierarchy(GHExt::config_t config);

class clock_writer_t {
public:
  clock_writer_t(const clock_writer_t &) = delete;
  clock_writer_t(clock_writer_t &&) = delete;

  void advance_clocks(window_t window);
  void stamp_last_step(window_t window, int iteration);
  void stamp_last_finalized(window_t window, int iteration);
  void restore_level_clock(level_t level, rat64 iteration,
                           std::optional<int> last_step_iteration);
  void rotate_timelevels(window_t window, gi_t gi);
  void rotate_array_timelevels(gi_t gi);

private:
  explicit clock_writer_t(GHExt &ext) : ext(ext) {}
  GHExt &ext;
  friend struct hierarchy_writers_t;
};

class level_writer_t {
public:
  level_writer_t(const level_writer_t &) = delete;
  level_writer_t(level_writer_t &&) = delete;

  const GHExt::PatchData::LevelData &
  add_level(level_t level, const amrex::BoxArray &ba,
            const amrex::DistributionMapping &dm,
            const std::function<std::string()> &why);
  const GHExt::PatchData::LevelData &
  make_level(level_t level, const amrex::BoxArray &ba,
             const amrex::DistributionMapping &dm, const std::string &why);
  GHExt::PatchData::LevelData
  remake_level(level_t level, const amrex::BoxArray &ba,
               const amrex::DistributionMapping &dm,
               const std::function<std::string()> &why);
  void erase_levels_from(level_t level);

private:
  explicit level_writer_t(GHExt &ext) : ext(ext) {}
  GHExt &ext;
  friend class structure_writer_t;
};

class structure_writer_t {
public:
  structure_writer_t(const structure_writer_t &) = delete;
  structure_writer_t(structure_writer_t &&) = delete;

  const GHExt::PatchData &add_patch(patch_t patch);
  void set_max_grid_size(patch_t patch,
                         const amrex::Vector<amrex::IntVect> &max_grid_sizes);
  void make_new_grids(patch_t patch, CCTK_REAL time);
  void mark_initialized();
  void regrid(level_t base_level, CCTK_REAL time);
  void set_finest_level(level_t finest_level);
  void recover_level(level_t level, amrex::BoxArray ba, bool rechop);

private:
  explicit structure_writer_t(GHExt &ext) : ext(ext), levels(ext) {}
  GHExt &ext;
  level_writer_t levels;
  friend struct hierarchy_writers_t;
};

class registry_writer_t {
public:
  registry_writer_t(const registry_writer_t &) = delete;
  registry_writer_t(registry_writer_t &&) = delete;

  void record_storage(gi_t gi, int timelevels);
  void freeze_storage();
  void record_integrator(
      GHExt::integrator_t integrator,
      std::source_location site = std::source_location::current());
  void freeze_integrator(
      std::source_location site = std::source_location::current());
  void adopt_flesh_cctkGH(cGH *cctkGH);

private:
  explicit registry_writer_t(GHExt &ext) : ext(ext) {}
  GHExt &ext;
  friend struct hierarchy_writers_t;
};

class scratch_writer_t {
public:
  scratch_writer_t(const scratch_writer_t &) = delete;
  scratch_writer_t(scratch_writer_t &&) = delete;

  std::size_t
  acquire(group_t group,
          std::source_location site = std::source_location::current());
  void release(group_t group, std::size_t slot);
  amrex::MultiFab &
  mfab(group_t group, std::size_t slot,
       std::source_location site = std::source_location::current());

private:
  explicit scratch_writer_t(GHExt &ext) : ext(ext) {}
  GHExt &ext;
  friend struct hierarchy_writers_t;
};

struct hierarchy_writers_t {
  clock_writer_t clock;
  structure_writer_t structure;
  registry_writer_t registry;
  scratch_writer_t scratch;

  ~hierarchy_writers_t();

private:
  explicit hierarchy_writers_t(std::unique_ptr<GHExt> made);
  std::unique_ptr<GHExt> hierarchy;
  friend std::unique_ptr<hierarchy_writers_t>
  create_hierarchy(GHExt::config_t config);
};

} // namespace CarpetX

#endif
