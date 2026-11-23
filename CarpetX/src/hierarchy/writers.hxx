#ifndef CARPETX_CARPETX_HIERARCHY_WRITERS_HXX
#define CARPETX_CARPETX_HIERARCHY_WRITERS_HXX

#include "internal.hxx"

#include "hierarchy/model.hxx"
#include <carpetx.hxx>
#include <rational.hxx>

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

class stage_writer_t {
public:
  stage_writer_t(const stage_writer_t &) = delete;
  stage_writer_t(stage_writer_t &&) = delete;

  void run_at(CCTK_REAL time, CCTK_REAL delta_time,
              const std::function<void()> &body);

private:
  explicit stage_writer_t(const GHExt &ext) : ext(ext) {}
  const GHExt &ext;
  friend struct hierarchy_writers_t;
};

class clock_writer_t {
public:
  clock_writer_t(const clock_writer_t &) = delete;
  clock_writer_t(clock_writer_t &&) = delete;

  void start_run(CCTK_REAL initial_time);
  void resume_run(int iteration, CCTK_REAL time);
  void set_delta_time(CCTK_REAL delta_time);
  void begin_iteration();
  void begin_step(window_t window);
  void show_level(level_t level);
  void replay_last_step(level_t level);
  void stamp_last_step(window_t window);
  void stamp_last_finalized(window_t window);
  void restore_level_clock(level_t level, rat64 iteration,
                           std::optional<int> last_step_iteration);
  void rotate_timelevels(window_t window, gi_t gi);
  void rotate_array_timelevels(gi_t gi);

private:
  explicit clock_writer_t(const GHExt &ext) : ext(ext) {}
  const GHExt &ext;
  void set(GHExt::run_clock_t next);
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
  explicit level_writer_t(const GHExt &ext) : ext(ext) {}
  const GHExt &ext;
  GHExt::PatchData::levels_t &changing(patch_t patch);
  void ensure_level_cctkGH(level_t level);
  void reset_modified(patch_t patch);
  friend class structure_writer_t;
};

class structure_writer_t {
public:
  structure_writer_t(const structure_writer_t &) = delete;
  structure_writer_t(structure_writer_t &&) = delete;

  const GHExt::PatchData &
  add_patch(patch_t patch, const faces_t<symmetry_t> &symmetries,
            const faces_t<boundary_t> &default_boundaries);
  void set_max_grid_size(patch_t patch,
                         const amrex::Vector<amrex::IntVect> &max_grid_sizes);
  void make_new_grids(patch_t patch, CCTK_REAL time);
  void mark_initialized();
  void regrid(level_t base_level, CCTK_REAL time);
  void set_finest_level(level_t finest_level);
  void recover_level(level_t level, amrex::BoxArray ba, bool rechop);

private:
  explicit structure_writer_t(const GHExt &ext) : ext(ext), levels(ext) {}
  const GHExt &ext;
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
  explicit registry_writer_t(const GHExt &ext) : ext(ext) {}
  const GHExt &ext;
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
  explicit scratch_writer_t(const GHExt &ext) : ext(ext) {}
  const GHExt &ext;
  friend struct hierarchy_writers_t;
};

struct hierarchy_writers_t {
private:
  GHExt hierarchy;

public:
  clock_writer_t clock;
  stage_writer_t stage;
  structure_writer_t structure;
  registry_writer_t registry;
  scratch_writer_t scratch;

  ~hierarchy_writers_t();

private:
  explicit hierarchy_writers_t(GHExt::config_t config);
  friend std::unique_ptr<hierarchy_writers_t>
  create_hierarchy(GHExt::config_t config);
};

} // namespace CarpetX

#endif
