#ifndef CARPETX_CARPETX_DRIVER_HXX
#define CARPETX_CARPETX_DRIVER_HXX

#include "internal.hxx"

#include "loop.hxx"
#include "valid.hxx"

#include <carpetx.hxx>
#include <carpetx_rk.hxx>
#include <rational.hxx>
#include <tuple.hxx>

#include <cctk.h>
#include <cctk_Schedule.h>

#include <AMReX.H>
#include <AMReX_AmrCore.H>
#include <AMReX_FluxRegister.H>
#include <AMReX_Interpolater.H>
#include <AMReX_MultiFab.H>
#include <AMReX_iMultiFab.H>

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <functional>
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
using namespace Arith;

using Loop::dim;

inline constexpr int max_num_rk_stages = 4;

template <class Index> struct index_range {
  struct iterator {
    Index current;
    constexpr Index operator*() const { return current; }
    constexpr iterator &operator++() {
      current = Index{current.index() + 1};
      return *this;
    }
    friend constexpr bool operator==(iterator, iterator) = default;
  };

  constexpr index_range(const Index begin, const Index end)
      : begin_(begin), end_(end) {}
  constexpr iterator begin() const { return iterator{begin_}; }
  constexpr iterator end() const { return iterator{end_}; }
  constexpr bool contains(const Index i) const {
    return begin_ <= i && i < end_;
  }

private:
  Index begin_, end_;
};
using tl_range = index_range<tl_t>;

[[noreturn]] void refuse(std::source_location site, const char *fmt, ...)
    CCTK_ATTRIBUTE_FORMAT(printf, 2, 3);

class operation_scope_t {
public:
  explicit operation_scope_t(const char *name);
  ~operation_scope_t();
  operation_scope_t(const operation_scope_t &) = delete;
  operation_scope_t &operator=(const operation_scope_t &) = delete;

private:
  const char *const previous;
};

const char *current_operation();

enum class group_kind { grid_function, array, scalar };

template <typename U> U &storage_entry(U &entry) { return entry; }
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

class CactusAmrCore final : public amrex::AmrCore {
  patch_t patch;

public:
  bool cactus_is_initialized = false;
  std::vector<bool> level_modified;

  CactusAmrCore(patch_t patch, const amrex::RealBox *rb, int max_level_in,
                const amrex::Vector<int> &n_cell_in, int coord = -1,
                amrex::Vector<amrex::IntVect> ref_ratios =
                    amrex::Vector<amrex::IntVect>(),
                const int *is_per = nullptr);
  CactusAmrCore(patch_t patch, const amrex::RealBox &rb, int max_level_in,
                const amrex::Vector<int> &n_cell_in, int coord,
                amrex::Vector<amrex::IntVect> const &ref_ratios,
                amrex::Array<int, AMREX_SPACEDIM> const &is_per);
  CactusAmrCore(const amrex::AmrCore &rhs) = delete;
  CactusAmrCore &operator=(const amrex::AmrCore &rhs) = delete;

  virtual ~CactusAmrCore() override;

  virtual void ErrorEst(int level, amrex::TagBoxArray &tags, amrex::Real time,
                        int ngrow) override;
  void SetupLevel(level_t level, const amrex::BoxArray &ba,
                  const amrex::DistributionMapping &dm,
                  const std::function<std::string()> &why);
  amrex::BoxArray RechopLevel(level_t level, amrex::BoxArray ba) const;
  void make_level(level_t level, const amrex::BoxArray &ba,
                  const amrex::DistributionMapping &dm, const std::string &why);
  virtual void
  MakeNewLevelFromScratch(int level, amrex::Real time,
                          const amrex::BoxArray &ba,
                          const amrex::DistributionMapping &dm) override;
  virtual void
  MakeNewLevelFromCoarse(int level, amrex::Real time, const amrex::BoxArray &ba,
                         const amrex::DistributionMapping &dm) override;
  virtual void RemakeLevel(int level, amrex::Real time,
                           const amrex::BoxArray &ba,
                           const amrex::DistributionMapping &dm) override;
  virtual void ClearLevel(int level) override;
};

struct GHExt {

  GHExt() = default;
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

  cctkGHptr global_cctkGH;
  std::vector<cctkGHptr> level_cctkGHs; // [reflevel]

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

      AnyTypeVector() : _type(-1), _typesize(-1), _count(0), _data(nullptr) {};
      AnyTypeVector(int type_, size_t count_)
          : _type(-1), _typesize(-1), _count(0), _data(nullptr) {
        alloc(type_, count_);
        assert(_type == type_);
        assert(_typesize != -1);
        assert(_count == count_);
        assert(_data != nullptr);
      };
      // Noncopyable for now
      AnyTypeVector(const AnyTypeVector &) = delete;
      AnyTypeVector &operator=(const AnyTypeVector &) = delete;
      AnyTypeVector &operator=(AnyTypeVector &&other) {
        swap(other);
        return *this;
      }
      AnyTypeVector(AnyTypeVector &&other)
          : _type(other._type), _typesize(other._typesize),
            _count(other._count), _data(other._data) {
        other._type = -1;
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
          assert(_type != -1);
          assert(_typesize != -1);
          amrex::The_Arena()->free(_data);
          _type = -1;
          _typesize = -1;
          _count = 0;
          _data = nullptr;
        }
        assert(_type == -1);
        assert(_typesize == -1);
        assert(_count == 0);
        assert(_data == nullptr);
      };

      void alloc(int type_, size_t count_) {
        assert(type_ == CCTK_VARIABLE_INT || type_ == CCTK_VARIABLE_REAL ||
               type_ == CCTK_VARIABLE_COMPLEX);

        assert(_type == -1);
        assert(_typesize == -1);
        assert(_count == 0);
        assert(_data == nullptr);

        _type = type_;
        _typesize = CCTK_VarTypeSize(_type);
        assert(_typesize > 0);
        _count = count_;
        _data = amrex::The_Arena()->alloc(_typesize * _count);
      }

      void free() {
        assert(_type != -1);
        assert(_typesize != -1);
        assert(_data != nullptr);
        amrex::The_Arena()->free(_data);
        _type = -1;
        _typesize = -1;
        _count = 0;
        _data = nullptr;
      }

      int type() const { return _type; };
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
      int _type, _typesize;
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
      void rotate_timelevels();

    private:
      std::vector<timelevel_t> tls;

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

  private:
    int num_timelevels_;
    std::unique_ptr<GlobalData::ArrayGroupData> array_;
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
    if (!storage_frozen)
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
  auto records(const std::optional<group_kind> kind = std::nullopt) const {
    return storage_range<const GroupRecord,
                         std::vector<GroupRecord>::const_iterator,
                         record_filter>(
        group_records.begin(), group_records.end(), record_filter{kind});
  }
  void freeze_storage();

  struct PatchData {
    PatchData() = delete;
    PatchData(const PatchData &) = delete;
    PatchData &operator=(const PatchData &) = delete;
    PatchData(PatchData &&) = default;
    PatchData &operator=(PatchData &&) = default;

    PatchData(patch_t patch);

    patch_t patch;

    bool is_cartesian;

    std::array<std::array<symmetry_t, dim>, 2> symmetries;

    // AMReX grid structure
    // TODO: convert this from unique_ptr to optional
    std::unique_ptr<CactusAmrCore> amrcore;

    struct LevelData {
      LevelData() = delete;
      LevelData(const LevelData &) = delete;
      LevelData &operator=(const LevelData &) = delete;
      LevelData(LevelData &&) = default;
      LevelData &operator=(LevelData &&) = default;

      LevelData(level_t level, const amrex::BoxArray &ba,
                const amrex::DistributionMapping &dm,
                const std::function<std::string()> &why);

      level_t level;

      // This level uses subcycling with respect to the next coarser
      // level. (Ignored for the coarsest level.)
      bool is_subcycling_level;

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

      // Cached flux geometry of this level for the flux-register (reflux)
      // path, built by amrex::Geometry::GetFaceArea and GetVolume:
      // face_area[d] is a one-component, zero-ghost MultiFab on this
      // level's BoxArray converted to the d-face centring (the centring of
      // the flux group in direction d), holding the constant face area
      // prod_{j != d} dx_j; cell_volume is its cell-centred counterpart
      // holding dx_0 dx_1 dx_2. All four share the level's
      // DistributionMapping with every GroupData MultiFab, so that the
      // (mflx, area, ...) and (mf, volume, ...) overloads of
      // amrex::FluxRegister can index them with the flux's or the state's
      // MFIter. EnsureFluxGeometry builds them together, lazily, on the
      // first AccumulateFluxes or Reflux that finds a register on this
      // level, so that neither call allocates a CarpetX-side MultiFab in
      // steady state. Constant per level: CarpetX levels are Cartesian, and
      // the layout only changes when a regrid remakes or clears the level,
      // which destroys this LevelData and the cache with it; there is no
      // other invalidation. Null on levels that never see a register; pure
      // geometry, never checkpointed and not valid-tracked.
      mutable std::array<std::unique_ptr<amrex::MultiFab>, dim> face_area;
      mutable std::unique_ptr<amrex::MultiFab> cell_volume;

      cctkGHptr patch_cctkGH;
      std::vector<cctkGHptr> local_cctkGHs; // [component]

      cGH *get_patch_cctkGH() const { return patch_cctkGH.get(); }
      cGH *get_local_cctkGH(const int component) const {
        return local_cctkGHs.at(component).get();
      }

      struct GroupData {
        GroupData() = delete;
        GroupData(const GroupData &) = delete;
        GroupData &operator=(const GroupData &) = delete;
        GroupData(GroupData &&) = delete;
        GroupData &operator=(GroupData &&) = delete;

        GroupData(const GroupRecord &record, level_t level,
                  const amrex::BoxArray &ba,
                  const amrex::DistributionMapping &dm,
                  const std::function<std::string()> &why);

        const GroupRecord &record;
        const level_t level;

        struct timelevel_t {
          timelevel_t(std::unique_ptr<amrex::MultiFab> mfab, int numvars,
                      const std::function<std::string()> &why)
              : mfab(std::move(mfab)), valid_(numvars, why_valid_t(why)) {}

          std::unique_ptr<amrex::MultiFab> mfab;
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
        void rotate_timelevels();
        int open_write_count() const {
          int count = 0;
          for (const timelevel_t &tl : tls)
            count += tl.open_writes;
          return count;
        }

        std::array<int, dim> indextype;
        std::array<int, dim> nghostzones;

        amrex::Interpolater *interpolator;

        std::array<std::array<boundary_t, dim>, 2> boundaries;
        bool all_faces_have_symmetries_or_boundaries() const;
        std::vector<std::array<int, dim> > parities;
        std::vector<CCTK_REAL> dirichlet_values;
        std::vector<CCTK_REAL> robin_values;
        amrex::Vector<amrex::BCRec> bcrecs;

        // Apply outer (physical) boundary conditions to a MultiFab
        void apply_boundary_conditions(amrex::MultiFab &mfab) const;

        // The RK buffers of the subcycling boundary fill (FillRKBoundary) into
        // this group on this (refined) level. All of them are owned by this
        // level and have the geometry of one FPinfo, that of this level's
        // MultiFab of the group for the group's interpolator and ghost width.
        // EnsureRKBuffers allocates them together, lazily, at the parent's
        // first StoreRKOldState after this level was made (or in the recovery
        // pre-pass). They die with this LevelData when a regrid remakes or
        // clears the level; there is no other invalidation. Null on level 0,
        // for groups that are not integrated, and where the coarse-fine
        // footprint is empty.

        // Coarse-fine source bands holding the parent level's subcycling RK
        // stage derivatives (zero-ghost, numvars comps) on the coarse cells
        // under this level's cf-ghost footprint: FPinfo::ba_crse_patch on
        // FPinfo::dm_patch, i.e. in the parent's index space and with exactly
        // the layout of rk_crse_patch below. Indexed by RK stage; filled by
        // StoreRKStage on the parent and read by FillRKBoundary on this level,
        // which evaluates the dense-output polynomial on them and prolongates
        // the resulting coarse state. Data with history: they hold the
        // parent's in-progress step for both of this level's substeps, and are
        // serialized at mid-cycle checkpoints under the parent level's name
        // (see rk_source_band).
        mutable std::array<std::unique_ptr<amrex::MultiFab>, max_num_rk_stages>
            ks_source_band;

        // Coarse-fine source band holding the parent's subcycling old state
        // u(t_n), a single snapshot (not RK-stage indexed) with the geometry
        // and lifecycle of the ks bands above. Filled from the parent's
        // var(tl) by StoreRKOldState at solve start; the u(t_n) base of the
        // dense output.
        mutable std::unique_ptr<amrex::MultiFab> old_source_band;

        // Persistent work buffers of the fill: the dense output is evaluated
        // from the bands straight into rk_crse_patch (FPinfo::ba_crse_patch on
        // FPinfo::dm_patch, the layout of the bands), which is then
        // interpolated into rk_fine_patch (FPinfo::ba_fine_patch on the same
        // dm_patch). Zero ghosts, numvars comps. Pure scratch, fully
        // overwritten before each read: never checkpointed and not
        // valid-tracked.
        mutable std::unique_ptr<amrex::MultiFab> rk_crse_patch, rk_fine_patch;

        std::unique_ptr<amrex::FluxRegister> freg;
        // Coarse-side staging for AccumulateFluxes: the parent's raw flux is
        // parallel-copied here, scaled, and added into freg. Allocated
        // together with freg and with the same arguments, so each face FabSet
        // has freg's BoxArray, DistributionMapping and component count.
        // Contents are meaningless between calls (zeroed before each use);
        // not checkpointed (not a freg_* band) and not valid-tracked; rebuilt
        // with freg when the level is remade on regrid.
        std::unique_ptr<amrex::FluxRegister> freg_scratch;
        std::optional<std::array<gi_t, dim> > fluxes;

        struct scratch_slot_t {
          std::unique_ptr<amrex::MultiFab> mfab;
          bool in_use = false;
        };
        std::vector<scratch_slot_t> scratch_pool;

      private:
        std::vector<timelevel_t> tls;

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
            std::vector<std::unique_ptr<GroupData> >::const_iterator,
            group_storage_filter>(groupdata.begin(), groupdata.end(),
                                  group_storage_filter{});
      }
      auto groups() {
        return storage_range<GroupData,
                             std::vector<std::unique_ptr<GroupData> >::iterator,
                             group_storage_filter>(
            groupdata.begin(), groupdata.end(), group_storage_filter{});
      }

    private:
      std::vector<std::unique_ptr<GroupData> > groupdata;

      friend YAML::Emitter &operator<<(YAML::Emitter &yaml,
                                       const LevelData &leveldata);
    };

    int num_levels() const { return leveldata.size(); }
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

    LevelData &add_level(level_t level, const amrex::BoxArray &ba,
                         const amrex::DistributionMapping &dm,
                         const std::function<std::string()> &why);
    LevelData replace_level(level_t level, LevelData &&fresh);
    void erase_levels_from(level_t level);

  private:
    std::vector<LevelData> leveldata;

    friend YAML::Emitter &operator<<(YAML::Emitter &yaml,
                                     const PatchData &patchdata);
  };

  int num_patches() const { return patchdata.size(); }
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

  PatchData &add_patch(patch_t patch);

  // Number of active timelevels per group, populated from schedule.ccl STORAGE
  // statements during CCTKi_ScheduleGHInit. Indexed by group index; 0 means no
  // storage. Frozen between schedule init and the first allocation pass.
  std::vector<int> active_timelevels; // [group index]
  bool storage_frozen = false;

  bool use_subcycling = false;

  // RecomputePerClockGroup is traversing CCTK_POST_RECOVER_VARIABLES, once
  // per clock group. Set only inside that loop. Levels outside the window may
  // then be at other times, so error messages read it to explain that a
  // result read across levels must be checkpointed.
  bool in_recovery_traversal = false;

  // The bin the driver is traversing on the end-of-step window (CCTK_POSTSTEP
  // and CCTK_ANALYSIS in Evolve and Initialise, CCTK_CHECKPOINT in Evolve,
  // CCTK_TERMINATE in Shutdown), or nullptr. Set only by
  // tail_traversal_scope. Under subcycling, SyncGroupsByDirISubcycling aborts
  // while it is set unless the window holds every level: otherwise a sync
  // would read the coarse level outside the window, which is not finalized.
  const char *tail_bin = nullptr;
  // The routine CallFunction dispatched last, set on entry and again on exit,
  // so it still names the routine when the flesh fires its SYNC clause after
  // the routine has returned. Read only by the error message of the
  // end-of-step sync check.
  const cFunctionData *last_called_function = nullptr;

  bool do_reflux = false;

  std::optional<rk_config_t> rk_config;
  mutable std::atomic<bool> rk_config_read{false};
  const std::optional<rk_config_t> &rk() const {
    rk_config_read.store(true, std::memory_order_relaxed);
    return rk_config;
  }

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

private:
  std::vector<GroupRecord> group_records;
  std::vector<PatchData> patchdata;

  friend YAML::Emitter &operator<<(YAML::Emitter &yaml, const GHExt &ghext);
  friend std::ostream &operator<<(std::ostream &os, const GHExt &ghext);
};

extern std::unique_ptr<GHExt> ghext;

GHExt &driver();

int num_rk_stages();
rk_dense_output_t registered_dense_output();

std::optional<std::array<gi_t, dim> > get_group_fluxes(gi_t gi);

void warn_inert_flux_tags();

bool get_group_checkpoint_flag(gi_t gi);

bool get_group_evolve_flag(gi_t gi);

// True iff every level of every patch sits at the same subcycling iteration,
// i.e. the checkpoint is time-aligned. Always true without subcycling. When
// false, the coarse source bands hold the in-progress coarse step (u(t_n) and
// the stage derivatives) that must be serialized.
bool all_levels_synchronized();

// Subcycling source-band kinds serialized at unsynchronized checkpoints:
// ks_source is the RK stages 0..max_num_rk_stages-1, old_source the u(t_n)
// snapshot, flux_register the six face FabSets of the child's flux register
// (GroupData::freg), each a zero-ghost MultiFab in the parent's index space
// holding the partially accumulated flux mismatch of the pair (level,
// level+1). For flux_register `stage` is the face index 0..5, namely
// 2*dir + (high ? 1 : 0).
enum class band_kind { ks_source, old_source, flux_register };

// Token identifying a source band in checkpoint names: "kss_s00".."kss_s03"
// for ks_source, "olds" for old_source, "freg_xlo".."freg_zhi" for
// flux_register. Shared by both IO backends.
std::string subcycling_band_tag(band_kind kind, int stage = -1);

amrex::MultiFab *rk_source_band(level_t level, gi_t gi, band_kind kind,
                                int stage = -1);

// Monotonically increasing counter. Incremented whenever the AMR grid
// hierarchy is invalidated (regridding, recovery). Starts at 0.
extern std::atomic<CCTK_INT> carpetx_epoch;

extern "C" CCTK_INT CarpetX_GetEpoch(void);

} // namespace CarpetX

#endif // #ifndef CARPETX_CARPETX_DRIVER_HXX
