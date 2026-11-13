#ifndef CARPETX_CARPETX_HIERARCHY_DRIVER_HXX
#define CARPETX_CARPETX_HIERARCHY_DRIVER_HXX

#include "internal.hxx"

#include <loop.hxx>
#include "hierarchy/indices.hxx"
#include "hierarchy/valid.hxx"
#include "util/refuse.hxx"
#include "util/var_type.hxx"

#include <carpetx.hxx>
#include <rational.hxx>
#include <tuple.hxx>

#include <cctk.h>
#include <cctk_Schedule.h>

#include <AMReX.H>
#include <AMReX_AmrCore.H>
#include <AMReX_MultiFab.H>
#include <AMReX_iMultiFab.H>

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <ostream>
#include <source_location>
#include <span>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace CarpetX {

using Loop::dim;

inline constexpr int max_num_rk_stages = 4;

struct stage_t : index_t<stage_t> {
  using index_t::index_t;
};

using tl_range = index_range<tl_t>;
using level_range = index_range<level_t>;

using stage_range = index_range<stage_t>;

constexpr stage_range rk_stages(const int nstages) {
  return stage_range{stage_t{1}, stage_t{nstages + 1}};
}
inline constexpr stage_range all_rk_stages = rk_stages(max_num_rk_stages);

template <class T> class const_propagating_ptr {
public:
  const_propagating_ptr() = default;
  const_propagating_ptr(std::unique_ptr<T> ptr) : ptr_(std::move(ptr)) {}
  const_propagating_ptr &operator=(std::unique_ptr<T> ptr) {
    ptr_ = std::move(ptr);
    return *this;
  }

  T *get() { return ptr_.get(); }
  const T *get() const { return ptr_.get(); }
  T *operator->() { return ptr_.get(); }
  const T *operator->() const { return ptr_.get(); }
  T &operator*() { return *ptr_; }
  const T &operator*() const { return *ptr_; }

private:
  std::unique_ptr<T> ptr_;
};

enum class group_kind { grid_function, array, scalar };

template <typename U> U &storage_entry(U &entry) { return entry; }
template <typename U> U &storage_entry(U *const entry) { return *entry; }
template <typename U> U &storage_entry(std::unique_ptr<U> &entry) {
  return *entry;
}
template <typename U> U &storage_entry(const std::unique_ptr<U> &entry) {
  return *entry;
}

template <typename T, typename Iter, typename Keep> class storage_range {
public:
  class iterator {
  public:
    iterator(Iter it, Iter last, Keep keep) : it(it), last(last), keep(keep) {
      settle();
    }
    T &operator*() const { return storage_entry(*it); }
    iterator &operator++() {
      ++it;
      settle();
      return *this;
    }
    bool operator!=(const iterator &other) const { return it != other.it; }

  private:
    void settle() {
      while (it != last && !keep(*it))
        ++it;
    }
    Iter it, last;
    Keep keep;
  };

  storage_range(Iter first, Iter last, Keep keep)
      : first(first), last(last), keep(keep) {}
  iterator begin() const { return iterator(first, last, keep); }
  iterator end() const { return iterator(last, last, keep); }

private:
  Iter first, last;
  Keep keep;
};

// TODO: It seems that AMReX now also has `RB90`, `RB180`, and
// `PolarB` boundary conditions. Make these available as well.

// Boundary conditions are group properties. They are valid only for faces where
// the domain symmetry is `none`.
enum class boundary_t {
  none,
  symmetry_boundary,
  dirichlet,
  linear_extrapolation,
  neumann,
  robin,
};
std::ostream &operator<<(std::ostream &os, const boundary_t boundary);

static_assert(AMREX_SPACEDIM == dim,
              "AMReX's AMREX_SPACEDIM must be the same as Cactus's cctk_dim");

static_assert(std::is_same<amrex::Real, CCTK_REAL>::value,
              "AMReX's Real type must be the same as Cactus's CCTK_REAL");

////////////////////////////////////////////////////////////////////////////////

class level_writer_t;
using amrcore_factory = std::unique_ptr<amrex::AmrCore> (*)(
    level_writer_t &levels, patch_t patch, const amrex::RealBox &domain, int,
    const amrex::Vector<int> &ncells,
    const amrex::Array<int, dim> &is_periodic);

struct level_buffers_t;
using level_buffers_ptr =
    std::unique_ptr<level_buffers_t, void (*)(level_buffers_t *)>;
using level_buffers_factory =
    level_buffers_ptr (*)(level_t level, const amrex::BoxArray &ba,
                          const amrex::DistributionMapping &dm);

struct GHExt {

  struct config_t {
    bool use_subcycling;
    bool do_reflux;
    amrcore_factory make_core;
    level_buffers_factory make_buffers;
  };
  explicit GHExt(config_t config);
  GHExt(const GHExt &) = delete;
  GHExt(GHExt &&) = delete;
  GHExt &operator=(const GHExt &) = delete;
  GHExt &operator=(GHExt &&) = delete;

  struct cctkGHptr {
    cGH *cctkGH;
    cctkGHptr(const cctkGHptr &) = delete;
    cctkGHptr(cctkGHptr &&ptr) : cctkGH(ptr.cctkGH) { ptr.cctkGH = nullptr; }
    cctkGHptr &operator=(const cctkGHptr &) = delete;
    cctkGHptr &operator=(cctkGHptr &&ptr);
    cctkGHptr() : cctkGH(nullptr) {}
    cctkGHptr(cGH *&&cctkGH) : cctkGH(cctkGH) {}
    cctkGHptr &operator=(cGH *&&cctkGH);
    ~cctkGHptr();
    operator bool() const { return bool(cctkGH); }
    cGH *get() const { return cctkGH; }
  };

  struct GroupRecord;

  struct GlobalData {

    class AnyTypeVector {

    public:
      // access to a single element of a AnyTypeVector
      class AnyTypeScalarRef {
      public:
        AnyTypeScalarRef() = delete;
        AnyTypeScalarRef(const AnyTypeVector &vect_, size_t idx_)
            : _vect(vect_), _idx(idx_) {}

      private:
        const AnyTypeVector &_vect;
        const size_t _idx;

        friend YAML::Emitter &
        operator<<(YAML::Emitter &yaml,
                   const AnyTypeScalarRef &anytypescalarref);
        friend std::ostream &operator<<(std::ostream &os,
                                        const AnyTypeScalarRef &scalar);
      };

      AnyTypeVector()
          : _type(std::nullopt), _typesize(-1), _count(0), _data(nullptr) {};
      AnyTypeVector(const var_type type_, size_t count_)
          : _type(std::nullopt), _typesize(-1), _count(0), _data(nullptr) {
        alloc(type_, count_);
        assert(_type == type_);
        assert(_typesize != -1);
        assert(_count == count_);
        assert(_data != nullptr);
      };
      AnyTypeVector(const AnyTypeVector &) = delete;
      AnyTypeVector &operator=(const AnyTypeVector &) = delete;
      AnyTypeVector &operator=(AnyTypeVector &&other) {
        swap(other);
        return *this;
      }
      AnyTypeVector(AnyTypeVector &&other)
          : _type(other._type), _typesize(other._typesize),
            _count(other._count), _data(other._data) {
        other._type = std::nullopt;
        other._typesize = -1;
        other._count = 0;
        other._data = nullptr;
      }
      void swap(AnyTypeVector &other) {
        std::swap(this->_type, other._type);
        std::swap(this->_typesize, other._typesize);
        std::swap(this->_count, other._count);
        std::swap(this->_data, other._data);
      }

      ~AnyTypeVector() {
        if (_data != nullptr) {
          assert(_type);
          assert(_typesize != -1);
          amrex::The_Arena()->free(_data);
          _type = std::nullopt;
          _typesize = -1;
          _count = 0;
          _data = nullptr;
        }
        assert(!_type);
        assert(_typesize == -1);
        assert(_count == 0);
        assert(_data == nullptr);
      };

      void alloc(const var_type type_, size_t count_) {
        assert(!_type);
        assert(_typesize == -1);
        assert(_count == 0);
        assert(_data == nullptr);

        _type = type_;
        _typesize = size_of(type_);
        assert(_typesize > 0);
        _count = count_;
        _data = amrex::The_Arena()->alloc(_typesize * _count);
      }

      void free() {
        assert(_type);
        assert(_typesize != -1);
        assert(_data != nullptr);
        amrex::The_Arena()->free(_data);
        _type = std::nullopt;
        _typesize = -1;
        _count = 0;
        _data = nullptr;
      }

      var_type type() const {
        assert(_type);
        return *_type;
      };
      int typesize() const { return _typesize; };

      const void *data_at(size_t i) const {
#ifdef CCTK_DEBUG
        if (i >= _count) {
          CCTK_VERROR("invalid index %zd exceeds %zd", i, _count);
        }
#endif
        assert(i < _count);
        return (char *)_data + i * _typesize;
      };

      void *data_at(size_t i) {
#ifdef CCTK_DEBUG
        if (i >= _count) {
          CCTK_VERROR("invalid index %zu exceeds %zu", i, _count);
        }
#endif
        assert(i < _count);
        return (char *)_data + i * _typesize;
      };

      AnyTypeScalarRef operator[](size_t idx) const {
        return AnyTypeScalarRef(*this, idx);
      }

      size_t size() const { return _count; };

      friend YAML::Emitter &operator<<(YAML::Emitter &yaml,
                                       const AnyTypeVector &anytypevector);

    private:
      std::optional<var_type> _type;
      int _typesize;
      size_t _count;
      void *_data;
    };

    struct ArrayGroupData {
      ArrayGroupData() = delete;
      ArrayGroupData(const ArrayGroupData &) = delete;
      ArrayGroupData &operator=(const ArrayGroupData &) = delete;
      ArrayGroupData(ArrayGroupData &&) = delete;
      ArrayGroupData &operator=(ArrayGroupData &&) = delete;

      explicit ArrayGroupData(const GroupRecord &record);

      const GroupRecord &record;

      struct timelevel_t {
        timelevel_t(AnyTypeVector data, int numvars,
                    const std::function<std::string()> &why)
            : data(std::move(data)),
              valid_(
                  numvars,
                  why_valid_t(make_valid_outer() | make_valid_ghosts(), why)) {}

        AnyTypeVector data;

        valid_t valid(const vi_t vi) const {
          return valid_.at(vi.index()).get();
        }
        const why_valid_t &why_valid(const vi_t vi) const {
          return valid_.at(vi.index());
        }

      private:
        std::vector<why_valid_t> valid_;
        friend class array_write_scope_t;
      };
      int array_size;
      int dimension;
      int lsh[dim];
      int ash[dim];
      int gsh[dim];
      int lbnd[dim];
      int ubnd[dim];
      int bbox[2 * dim];
      int nghostzones[dim];

      const timelevel_t &timelevel(const tl_t tl,
                                   const std::source_location site =
                                       std::source_location::current()) const {
        if (tl.index() < 0 || tl.index() >= int(tls.size()))
          refuse(site,
                 "time level %d of group %s does not exist: it has %d time "
                 "levels",
                 tl.index(), record.groupname.c_str(), int(tls.size()));
        return tls[tl.index()];
      }
      timelevel_t &timelevel(
          const tl_t tl,
          const std::source_location site = std::source_location::current()) {
        return const_cast<timelevel_t &>(
            std::as_const(*this).timelevel(tl, site));
      }

    private:
      std::vector<timelevel_t> tls;
      void rotate_timelevels();

      friend struct cctkGH_binding;

      friend YAML::Emitter &operator<<(YAML::Emitter &yaml,
                                       const ArrayGroupData &arraygroupdata);
    };
  };

  struct GroupRecord {
    GroupRecord(gi_t gi, int num_timelevels);

    std::string groupname;
    gi_t groupindex;
    int firstvarindex;
    int numvars;
    bool do_checkpoint;
    bool do_evolve;
    bool do_restrict;
    bool restrict_on_sync;
    group_kind kind;
    CarpetX::var_type var_type;

    int num_timelevels() const { return num_timelevels_; }
    tl_range timelevels() const {
      return tl_range{tl_t{0}, tl_t{num_timelevels_}};
    }
    index_range<vi_t> variables() const {
      return index_range<vi_t>{vi_t{0}, vi_t{numvars}};
    }
    var_t var(const vi_t vi) const {
      assert(variables().contains(vi));
      return var_t{firstvarindex + vi.index()};
    }
    bool has_storage() const { return num_timelevels_ > 0; }

    const GlobalData::ArrayGroupData &
    array(const std::source_location site =
              std::source_location::current()) const {
      if (kind == group_kind::grid_function)
        refuse(site, "group %s is not a grid array or scalar group",
               groupname.c_str());
      assert(array_);
      return *array_;
    }
    GlobalData::ArrayGroupData &
    array(const std::source_location site = std::source_location::current()) {
      return const_cast<GlobalData::ArrayGroupData &>(
          std::as_const(*this).array(site));
    }

    struct grid_function_t {
      std::array<int, dim> indextype;
      std::array<int, dim> nghostzones;
      std::array<std::array<boundary_t, dim>, 2> boundaries;
      std::vector<std::array<int, dim>> parities;
      std::vector<CCTK_REAL> dirichlet_values;
      std::vector<CCTK_REAL> robin_values;
      std::optional<std::array<gi_t, dim>> fluxes;
    };
    const grid_function_t &
    grid_function(const std::source_location site =
                      std::source_location::current()) const {
      if (kind != group_kind::grid_function)
        refuse(site, "group %s is not a grid function group",
               groupname.c_str());
      assert(grid_function_);
      return *grid_function_;
    }

  private:
    int num_timelevels_;
    std::unique_ptr<GlobalData::ArrayGroupData> array_;
    std::optional<grid_function_t> grid_function_;
    static grid_function_t make_grid_function(gi_t gi, int numvars);
  };

  struct record_filter {
    std::optional<group_kind> kind;
    bool operator()(const GroupRecord &group) const {
      return group.has_storage() && (!kind || group.kind == *kind);
    }
  };

  const GroupRecord &record(
      const gi_t gi,
      const std::source_location site = std::source_location::current()) const {
    if (!storage_frozen())
      refuse(site, "CarpetX group %d has no record yet: storage is not frozen",
             gi.index());
    if (gi.index() < 0 || gi.index() >= int(group_records.size()))
      refuse(site, "group index %d does not exist: there are %d groups",
             gi.index(), int(group_records.size()));
    return group_records[gi.index()];
  }
  GroupRecord &
  record(const gi_t gi,
         const std::source_location site = std::source_location::current()) {
    return const_cast<GroupRecord &>(std::as_const(*this).record(gi, site));
  }
  const GroupRecord &
  record_of(var_t var,
            std::source_location site = std::source_location::current()) const;
  auto records(const std::optional<group_kind> kind = std::nullopt) const {
    return storage_range<const GroupRecord,
                         std::vector<GroupRecord>::const_iterator,
                         record_filter>(
        group_records.begin(), group_records.end(), record_filter{kind});
  }
  int active_timelevels(gi_t gi) const;
  bool storage_frozen() const { return !pending_timelevels_; }

  struct integrator_t {
    rk_dense_output_t dense_output;
    std::vector<CCTK_REAL> weights;
    std::map<gi_t, gi_t> rhs;
    stage_range stages() const { return rk_stages(int(weights.size())); }
  };
  const std::optional<integrator_t> &
  integrator(std::source_location site = std::source_location::current()) const;

  bool is_initialized() const { return initialized_; }
  CCTK_INT epoch() const { return epoch_; }

  struct PatchData {
    PatchData() = delete;
    PatchData(const PatchData &) = delete;
    PatchData &operator=(const PatchData &) = delete;
    PatchData(PatchData &&) = default;
    PatchData &operator=(PatchData &&) = default;

    PatchData(level_writer_t &levels, patch_t patch, amrcore_factory make_core);

    patch_t patch;

    bool is_cartesian;

    std::array<std::array<symmetry_t, dim>, 2> symmetries;

    const_propagating_ptr<amrex::AmrCore> amrcore;

    struct LevelData {
      LevelData() = delete;
      LevelData(const LevelData &) = delete;
      LevelData &operator=(const LevelData &) = delete;
      LevelData(LevelData &&) = default;
      LevelData &operator=(LevelData &&) = default;

      struct clock_t {
        rat64 iteration, delta_iteration;
        int last_step_iteration, last_finalized_iteration;
      };
      static clock_t coarsest_clock();
      clock_t child_clock(bool use_subcycling) const;

      LevelData(level_t level, const amrex::BoxArray &ba,
                const amrex::DistributionMapping &dm,
                const std::function<std::string()> &why, clock_t clock,
                level_buffers_ptr buffers, std::span<const GroupRecord> records,
                const amrex::Geometry &geom);

      level_t level;

      // Iteration and time at which this cycle level is valid
      rat64 iteration, delta_iteration;

      // The cctk_iteration in which this level's clock group reached its
      // current clock: stamped by Evolve after every batch's EVOL on the
      // widened window, 0 on a fresh start, the parent's value when a regrid
      // creates or remakes the level, and the checkpointed value on
      // recovery. Recovery replays CCTK_POST_RECOVER_VARIABLES in this
      // context. Checkpointed next to the level's clock.
      int last_step_iteration;

      // The cctk_iteration of the last end-of-step window (POSTSTEP,
      // ANALYSIS) that covered this level, i.e. when it was last finalized:
      // stamped by Evolve after each such window, 0 on a fresh start and on
      // recovery, the parent's value when a regrid creates or remakes the
      // level. Not checkpointed.
      int last_finalized_iteration;

      // Fabamrex::ArrayBase object holding a cell-centred BoxArray for
      // iterating over grid functions. This stores the grid structure
      // and its distribution over all processes, but holds no data.
      std::unique_ptr<amrex::FabArrayBase> fab;

      level_buffers_ptr buffers;

      cctkGHptr patch_cctkGH;
      std::vector<cctkGHptr> local_cctkGHs;

      cGH *get_patch_cctkGH() const { return patch_cctkGH.get(); }
      cGH *get_local_cctkGH(const tile_t tile) const {
        cGH *const cctkGH = local_cctkGHs.at(tile.index()).get();
        assert(cctkGH->cctk_component == tile.component().index());
        return cctkGH;
      }

      using ghost_mask_key_t =
          std::pair<std::array<int, dim>, std::array<int, dim>>;
      static constexpr int ghost_mask_interior = 0;
      static constexpr int ghost_mask_same_level = 1;
      static constexpr int ghost_mask_coarse_fine = 2;
      static constexpr int ghost_mask_physical_boundary = 3;

      struct GroupData {
        GroupData() = delete;
        GroupData(const GroupData &) = delete;
        GroupData &operator=(const GroupData &) = delete;
        GroupData(GroupData &&) = delete;
        GroupData &operator=(GroupData &&) = delete;

        GroupData(const GroupRecord &record, level_t level,
                  const amrex::BoxArray &ba,
                  const amrex::DistributionMapping &dm,
                  const std::function<std::string()> &why,
                  const amrex::Geometry &geom);

        const GroupRecord &record;
        const GroupRecord::grid_function_t &grid_function;
        const level_t level;

        struct timelevel_t {
          timelevel_t(std::unique_ptr<amrex::MultiFab> mfab, int numvars,
                      const std::function<std::string()> &why)
              : mfab(std::move(mfab)), valid_(numvars, why_valid_t(why)) {}

          const_propagating_ptr<amrex::MultiFab> mfab;
          int open_writes = 0;

          valid_t valid(const vi_t vi) const {
            return valid_.at(vi.index()).get();
          }
          const why_valid_t &why_valid(const vi_t vi) const {
            return valid_.at(vi.index());
          }

        private:
          std::vector<why_valid_t> valid_;
          friend class write_scope_t;
        };
        const timelevel_t &
        timelevel(const tl_t tl, const std::source_location site =
                                     std::source_location::current()) const {
          if (tl.index() < 0 || tl.index() >= int(tls.size()))
            refuse(site,
                   "time level %d of group %s does not exist on patch %d, "
                   "refinement level %d: it has %d time levels",
                   tl.index(), record.groupname.c_str(), level.patch().index(),
                   level.index(), int(tls.size()));
          return tls[tl.index()];
        }
        timelevel_t &timelevel(
            const tl_t tl,
            const std::source_location site = std::source_location::current()) {
          return const_cast<timelevel_t &>(
              std::as_const(*this).timelevel(tl, site));
        }
        int num_timelevels() const { return tls.size(); }
        int open_write_count() const {
          int count = 0;
          for (const timelevel_t &tl : tls)
            count += tl.open_writes;
          return count;
        }

        bool all_faces_have_symmetries_or_boundaries() const;

        struct scratch_slot_t {
          const_propagating_ptr<amrex::MultiFab> mfab;
          bool in_use = false;
        };
        std::vector<scratch_slot_t> scratch_pool;

      private:
        std::vector<timelevel_t> tls;
        void rotate_timelevels();

        friend struct cctkGH_binding;
        friend YAML::Emitter &operator<<(YAML::Emitter &yaml,
                                         const GroupData &groupdata);
      };

      struct group_storage_filter {
        bool operator()(const std::unique_ptr<GroupData> &groupdata) const {
          return groupdata && groupdata->num_timelevels() > 0;
        }
      };

      const GroupData &group(const gi_t gi,
                             const std::source_location site =
                                 std::source_location::current()) const {
        if (gi.index() < 0 || gi.index() >= int(groupdata.size()))
          refuse(site, "group index %d does not exist: there are %d groups",
                 gi.index(), int(groupdata.size()));
        const GroupData *const gd = groupdata[gi.index()].get();
        if (!gd)
          refuse(site, "group %s is not a grid function group",
                 CCTK_FullGroupName(gi.index()));
        if (gd->num_timelevels() == 0)
          refuse(site,
                 "group %s has no storage on patch %d, refinement level %d",
                 CCTK_FullGroupName(gi.index()), level.patch().index(),
                 level.index());
        return *gd;
      }
      GroupData &
      group(const gi_t gi,
            const std::source_location site = std::source_location::current()) {
        return const_cast<GroupData &>(std::as_const(*this).group(gi, site));
      }
      auto groups() const {
        return storage_range<
            const GroupData,
            std::vector<std::unique_ptr<GroupData>>::const_iterator,
            group_storage_filter>(groupdata.begin(), groupdata.end(),
                                  group_storage_filter{});
      }
      auto groups() {
        return storage_range<GroupData,
                             std::vector<std::unique_ptr<GroupData>>::iterator,
                             group_storage_filter>(
            groupdata.begin(), groupdata.end(), group_storage_filter{});
      }

      const amrex::iMultiFab &ghost_mask(const GroupData &groupdata) const {
        return ghost_masks.at(
            ghost_mask_key_t{groupdata.grid_function.indextype,
                             groupdata.grid_function.nghostzones});
      }

    private:
      std::vector<std::unique_ptr<GroupData>> groupdata;
      std::map<ghost_mask_key_t, amrex::iMultiFab> ghost_masks;

      friend YAML::Emitter &operator<<(YAML::Emitter &yaml,
                                       const LevelData &leveldata);
    };

    int num_levels() const { return leveldata.size(); }
    level_range level_handles() const {
      return level_range{level_t{patch, 0}, level_t{patch, num_levels()}};
    }
    const LevelData &level(const level_t level,
                           const std::source_location site =
                               std::source_location::current()) const {
      assert(level.patch() == patch);
      if (level.index() < 0 || level.index() >= num_levels())
        refuse(site,
               "CarpetX level %d does not exist on patch %d: it has %d levels",
               level.index(), patch.index(), num_levels());
      return leveldata[level.index()];
    }
    LevelData &
    level(const level_t level,
          const std::source_location site = std::source_location::current()) {
      return const_cast<LevelData &>(std::as_const(*this).level(level, site));
    }
    std::span<const LevelData> levels() const { return leveldata; }
    std::span<LevelData> levels() { return leveldata; }
    std::optional<window_t::bounds_t> modified_levels() const;

  private:
    std::vector<LevelData> leveldata;
    std::vector<bool> level_modified;

    void reset_level_modified();
    LevelData &add_level(LevelData &&fresh);
    LevelData replace_level(level_t level, LevelData &&fresh);
    void erase_levels_from(level_t level);

    friend class structure_writer_t;
    friend struct GHExt;
    friend YAML::Emitter &operator<<(YAML::Emitter &yaml,
                                     const PatchData &patchdata);
  };

  int num_patches() const { return patchdata.size(); }
  index_range<patch_t> patch_handles() const {
    return index_range<patch_t>{patch_t{0}, patch_t{num_patches()}};
  }
  const PatchData &patch(
      const patch_t patch,
      const std::source_location site = std::source_location::current()) const {
    if (patch.index() < 0 || patch.index() >= num_patches())
      refuse(site, "CarpetX patch %d does not exist: there are %d patches",
             patch.index(), num_patches());
    return patchdata[patch.index()];
  }
  PatchData &
  patch(const patch_t patch,
        const std::source_location site = std::source_location::current()) {
    return const_cast<PatchData &>(std::as_const(*this).patch(patch, site));
  }
  const PatchData::LevelData &level(
      const level_t level,
      const std::source_location site = std::source_location::current()) const {
    return this->patch(level.patch(), site).level(level, site);
  }
  PatchData::LevelData &
  level(const level_t level,
        const std::source_location site = std::source_location::current()) {
    return const_cast<PatchData::LevelData &>(
        std::as_const(*this).level(level, site));
  }
  const PatchData::LevelData::GroupData &group(
      const group_t group,
      const std::source_location site = std::source_location::current()) const {
    return this->level(group.level, site).group(group.gi, site);
  }
  PatchData::LevelData::GroupData &
  group(const group_t group,
        const std::source_location site = std::source_location::current()) {
    return const_cast<PatchData::LevelData::GroupData &>(
        std::as_const(*this).group(group, site));
  }
  std::span<const PatchData> patches() const { return patchdata; }
  std::span<PatchData> patches() { return patchdata; }

  const bool use_subcycling;

  const bool do_reflux;

  int num_levels(const patch_t patch) const {
    return this->patch(patch).num_levels();
  }
  int num_levels() const {
    int nlevels = 0;
    using std::max;
    for (const auto &pd : patches())
      nlevels = max(nlevels, pd.num_levels());
    return nlevels;
  }

  cGH *get_global_cctkGH() const { return global_cctkGH.get(); }
  cGH *get_level_cctkGH(const level_t level) const {
    return level_cctkGHs.at(level.index()).get();
  }

  template <typename F> void for_each_cctkGH(F &&f) const {
    f(global_cctkGH.get());
    for (const auto &level_cctkGH : level_cctkGHs)
      f(level_cctkGH.get());
    for (const auto &patch : patches())
      for (const auto &level : patch.levels())
        f(level.get_patch_cctkGH());
    for (const auto &patch : patches())
      for (const auto &level : patch.levels())
        for (const auto &local_cctkGH : level.local_cctkGHs)
          f(local_cctkGH.get());
  }

private:
  cGH *flesh_cctkGH = nullptr;
  cctkGHptr global_cctkGH;
  std::vector<cctkGHptr> level_cctkGHs;
  std::optional<std::vector<int>> pending_timelevels_;
  bool initialized_ = false;
  CCTK_INT epoch_ = 0;
  const amrcore_factory make_core_;
  const level_buffers_factory make_buffers_;
  std::vector<GroupRecord> group_records;
  std::optional<integrator_t> integrator_;
  bool integrator_open_ = true;
  std::vector<PatchData> patchdata;

  PatchData &add_patch(level_writer_t &levels, patch_t patch);
  PatchData::LevelData &add_level(PatchData::LevelData &&fresh);
  PatchData::LevelData replace_level(PatchData::LevelData &&fresh);
  void erase_levels_from(level_t level);
  void record_storage(gi_t gi, int timelevels);
  void freeze_storage();
  void record_integrator(integrator_t integrator, std::source_location site);
  void freeze_integrator(std::source_location site);
  void mark_initialized();

  friend class clock_writer_t;
  friend class structure_writer_t;
  friend class level_writer_t;
  friend class registry_writer_t;
  friend class scratch_writer_t;
  friend struct cctkGH_binding;
  friend class write_scope_t;
  friend class array_write_scope_t;

  friend YAML::Emitter &operator<<(YAML::Emitter &yaml, const GHExt &ext);
  friend std::ostream &operator<<(std::ostream &os, const GHExt &ext);
};

const GHExt &driver();
void assert_hierarchy_destroyed();

class scope_door_t {
  static GHExt &open();
  friend class write_scope_t;
  friend class array_write_scope_t;
};

class hierarchy_slot_t {
  static void install(GHExt &ext);
  static void remove(GHExt &ext);
  friend struct hierarchy_writers_t;
};

} // namespace CarpetX

#endif
