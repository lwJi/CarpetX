#ifndef CARPETX_CARPETX_DRIVER_HXX
#define CARPETX_CARPETX_DRIVER_HXX

#include "loop.hxx"
#include "valid.hxx"

#include <rational.hxx>
#include <tuple.hxx>

#include <cctk.h>

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
#include <string>
#include <type_traits>
#include <vector>

namespace CarpetX {
using namespace Arith;

using Loop::dim;

using rat64 = rational<int64_t>;

// Compile-time capacity of the subcycling band arrays. The runtime-active
// count lives in GHExt::num_rk_stages (RK4 uses 4, SSPRK3 uses 3).
inline constexpr int max_num_rk_stages = 4;

// TODO: It seems that AMReX now also has `RB90`, `RB180`, and
// `PolarB` boundary conditions. Make these available as well.

// Symmetries are domain properties
enum class symmetry_t {
  none,
  interpatch,
  periodic,
  reflection,
};
std::ostream &operator<<(std::ostream &os, const symmetry_t symmetry);

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

// AMR driver
class CactusAmrCore final : public amrex::AmrCore {
  int patch;

public:
  bool cactus_is_initialized = false;
  std::vector<bool> level_modified;

  CactusAmrCore();
  CactusAmrCore(int patch, const amrex::RealBox *rb, int max_level_in,
                const amrex::Vector<int> &n_cell_in, int coord = -1,
                amrex::Vector<amrex::IntVect> ref_ratios =
                    amrex::Vector<amrex::IntVect>(),
                const int *is_per = nullptr);
  CactusAmrCore(int patch, const amrex::RealBox &rb, int max_level_in,
                const amrex::Vector<int> &n_cell_in, int coord,
                amrex::Vector<amrex::IntVect> const &ref_ratios,
                amrex::Array<int, AMREX_SPACEDIM> const &is_per);
  CactusAmrCore(const amrex::AmrCore &rhs) = delete;
  CactusAmrCore &operator=(const amrex::AmrCore &rhs) = delete;

  virtual ~CactusAmrCore() override;

  virtual void ErrorEst(int level, amrex::TagBoxArray &tags, amrex::Real time,
                        int ngrow) override;
  void SetupLevel(int level, const amrex::BoxArray &ba,
                  const amrex::DistributionMapping &dm,
                  const std::function<std::string()> &why);
  // Re-partition a recovered level's covered region into a fresh box
  // decomposition matching the current max_grid_size / node count. Pure
  // geometry: the returned BoxArray covers exactly the same cell union as the
  // input, so no field data is changed. Level 0 regenerates the canonical
  // full-domain decomposition; levels > 0 coalesce then re-tile + load-balance.
  amrex::BoxArray RechopLevel(int level, amrex::BoxArray ba) const;
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

// Cactus grid hierarchy extension
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

  struct CommonGroupData {
    std::string groupname;
    int groupindex;
    int firstvarindex;
    int numvars;

    bool do_checkpoint; // whether to checkpoint
    bool do_evolve;     // whether this is an evolved state variable
    bool do_restrict;   // whether to restrict
    // whether this group may be restricted at all, including during a SYNC
    // (restrict_during_sync): unlike do_restrict, which defaults to
    // do_checkpoint, false only for an explicit restrict="no" tag
    bool allow_restrict;

    std::vector<std::vector<why_valid_t> > valid; // [time level][var index]

    // TODO: add poison_invalid and check_valid functions

    friend YAML::Emitter &operator<<(YAML::Emitter &yaml,
                                     const CommonGroupData &commongroupdata);
  };

  struct GlobalData {
    // all data that exists on all levels

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

    // For subcycling in time, there really should be one copy of each
    // integrated grid scalar per level. We don't do that yet; instead,
    // we assume that grid scalars only hold "analysis" data.

    struct ArrayGroupData : public CommonGroupData {
      std::vector<AnyTypeVector>
          data; // [time level][var index + grid point index]
      int array_size;
      int dimension;
      int activetimelevels;
      int lsh[dim];
      int ash[dim];
      int gsh[dim];
      int lbnd[dim];
      int ubnd[dim];
      int bbox[2 * dim];
      int nghostzones[dim];

      ArrayGroupData() {
        array_size = -1;
        dimension = -1;
        activetimelevels = -1;
        for (int d = 0; d < dim; d++) {
          lsh[d] = -1;
          ash[d] = -1;
          gsh[d] = -1;
          lbnd[d] = -1;
          ubnd[d] = -1;
          bbox[2 * d] = bbox[2 * d + 1] = -1;
          nghostzones[d] = -1;
        }
      }

      friend YAML::Emitter &operator<<(YAML::Emitter &yaml,
                                       const ArrayGroupData &arraygroupdata);
    };
    // TODO: right now this is sized for the total number of groups
    std::vector<std::unique_ptr<ArrayGroupData> >
        arraygroupdata; // [group index]

    friend YAML::Emitter &operator<<(YAML::Emitter &yaml,
                                     const GlobalData &globaldata);
  };
  GlobalData globaldata;

  struct PatchData {
    PatchData() = delete;
    PatchData(const PatchData &) = delete;
    PatchData &operator=(const PatchData &) = delete;
    PatchData(PatchData &&) = default;
    PatchData &operator=(PatchData &&) = default;

    PatchData(int patch);

    int patch;

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

      LevelData(const int patch, const int level, const amrex::BoxArray &ba,
                const amrex::DistributionMapping &dm,
                const std::function<std::string()> &why);

      int patch, level;

      // This level uses subcycling with respect to the next coarser
      // level. (Ignored for the coarsest level.)
      bool is_subcycling_level;

      // Iteration and time at which this cycle level is valid
      rat64 iteration, delta_iteration;

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

      struct GroupData : public CommonGroupData {
        GroupData() = delete;
        GroupData(const GroupData &) = delete;
        GroupData &operator=(const GroupData &) = delete;
        GroupData(GroupData &&) = delete;
        GroupData &operator=(GroupData &&) = delete;

        GroupData(int patch, int level, int gi, const amrex::BoxArray &ba,
                  const amrex::DistributionMapping &dm,
                  const std::function<std::string()> &why);

        int patch, level;

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

        // each amrex::MultiFab has numvars components
        std::vector<std::unique_ptr<amrex::MultiFab> > mfab; // [time level]

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

        // Flux register of the pair (level - 1, level), owned by this (fine)
        // level: allocated only where it is fed, i.e. for a group with a
        // fluxes= tag that ODESolvers integrates (ghext->rk_integrated_group)
        // under do_reflux, with and without subcycling; zeroed on creation,
        // and reset by the parent's first RK stage (AccumulateFluxes).
        //
        // Contract of the fluxes= tag: on an ODESolvers-integrated group it
        // promises that the flux groups are computed in ODESolvers_RHS, so
        // that every RK stage can accumulate them; on any other group the tag
        // is validated for shape (LevelData constructor) and otherwise inert:
        // no register, no accumulation, no validity check of the fluxes.
        std::unique_ptr<amrex::FluxRegister> freg;
        // Coarse-side staging for AccumulateFluxes: the parent's raw flux is
        // parallel-copied here, scaled, and added into freg. Allocated
        // together with freg and with the same arguments, so each face FabSet
        // has freg's BoxArray, DistributionMapping and component count.
        // Contents are meaningless between calls (zeroed before each use);
        // not checkpointed (not a freg_* band) and not valid-tracked; rebuilt
        // with freg when the level is remade on regrid.
        std::unique_ptr<amrex::FluxRegister> freg_scratch;
        // associated flux group indices
        std::array<int, dim> fluxes; // [dir]

        // CarpetX can allocate and free (temporary) multifabs that
        // are associated with a Cactus grid function group. These
        // multifabs remain allocated when they are freed, which makes
        // it efficient when they are re-allocated later. However,
        // they are freed when the current level changes during
        // regridding (and the shape of the multifab presumably
        // changes). This is used e.g. by ODESolvers for its
        // temporaries.
      private:
        mutable std::vector<std::unique_ptr<amrex::MultiFab> > tmp_mfabs;
        mutable std::size_t next_tmp_mfab;

      public:
        void init_tmp_mfabs() const;
        amrex::MultiFab *alloc_tmp_mfab() const;
        void free_tmp_mfabs() const;

        friend YAML::Emitter &operator<<(YAML::Emitter &yaml,
                                         const GroupData &groupdata);
      };
      // TODO: right now this is sized for the total number of groups
      std::vector<std::unique_ptr<GroupData> > groupdata; // [group index]

      friend YAML::Emitter &operator<<(YAML::Emitter &yaml,
                                       const LevelData &leveldata);
    };
    std::vector<LevelData> leveldata; // [reflevel]

    friend YAML::Emitter &operator<<(YAML::Emitter &yaml,
                                     const PatchData &patchdata);
  };
  std::vector<PatchData> patchdata; // [patch]

  // Number of active timelevels per group, populated from schedule.ccl STORAGE
  // statements during CCTKi_ScheduleGHInit. Indexed by group index; 0 means no
  // storage. Frozen between schedule init and the first allocation pass.
  std::vector<int> active_timelevels; // [group index]
  bool storage_frozen = false;

  bool use_subcycling = false;

  // CarpetX::do_reflux, cached in InitGH next to use_subcycling. Read from
  // here rather than through DECLARE_CCTK_PARAMETERS in the GroupData
  // constructor, where the parameter block would shadow the do_restrict
  // member; recovery restores the parameter before InitGH runs.
  bool do_reflux = false;

  // Stage count of ODESolvers::method, published at WRAGH from the method
  // table (ODESolvers/src/rk_methods.hxx). Under subcycling it is <=
  // max_num_rk_stages (ODESolvers_CheckMethod rejects the others at
  // PARAMCHECK) and sizes the bands; without subcycling it may exceed that
  // (RKF78: 11, DP87: 13) and only bounds the stage AccumulateFluxes accepts.
  int num_rk_stages = 4;

  // Groups advanced by the time integrator, published by it at WRAGH
  // (ODESolvers: the groups with an "rhs" tag). These are exactly the groups
  // that own coarse-fine source bands under subcycling, both during evolution
  // and when recovery rebuilds the bands. Empty means none.
  std::vector<bool> rk_integrated_group; // [group index]

  // Per-level iteration values read from checkpoint; consumed by recovery fixup
  // in schedule.cxx. Indexed [patch][level]. Empty outside of recovery window.
  std::vector<std::vector<std::optional<rat64> > > recovered_level_iterations;

  int num_patches() const { return patchdata.size(); }
  int num_levels(const int patch) const {
    return patchdata.at(patch).leveldata.size();
  }
  int num_levels() const {
    int nlevels = 0;
    using std::max;
    for (const auto &pd : patchdata)
      nlevels = max(nlevels, int(pd.leveldata.size()));
    return nlevels;
  }

  cGH *get_global_cctkGH() const { return global_cctkGH.get(); }
  cGH *get_level_cctkGH(const int level) const {
    return level_cctkGHs.at(level).get();
  }
  cGH *get_patch_cctkGH(const int level, const int patch) const {
    return patchdata.at(patch).leveldata.at(level).patch_cctkGH.get();
  }
  cGH *get_local_cctkGH(const int level, const int patch,
                        const int component) const {
    return patchdata.at(patch)
        .leveldata.at(level)
        .local_cctkGHs.at(component)
        .get();
  }

  friend YAML::Emitter &operator<<(YAML::Emitter &yaml, const GHExt &ghext);
  friend std::ostream &operator<<(std::ostream &os, const GHExt &ghext);
};

extern std::unique_ptr<GHExt> ghext;

// Print one CCTK_WARN_ALERT for each grid-function group whose fluxes= tag
// cannot take effect: flux-tagged, but rejected by group_has_flux_register
// (subcycling.hxx), i.e. not integrated by ODESolvers, so it never gets a
// flux register and is never refluxed. With CarpetX::do_reflux = no, print
// instead a single CCTK_INFO if some integrated grid function carries a
// fluxes= tag, since every such tag is then inert. Silent when
// CarpetX::max_num_levels <= 1, since then no group gets a register. Warns
// on the root process only. Called once by Initialise, after
// PARAMCHECK (it needs ghext->do_reflux from InitGH and
// ghext->rk_integrated_group from WRAGH) and before any level exists, so it
// fires once per run, fresh start or recovery.
void warn_inert_flux_tags();

// Group tag "checkpoint"; absent -> yes. Reads only the tags table, so valid
// from PARAMCHECK on. Defined in driver.cxx.
bool get_group_checkpoint_flag(int gi);

// Group tag "evolve"; absent -> the "checkpoint" tag (absent -> yes). Reads
// only the tags table, so valid from PARAMCHECK on. Defined in driver.cxx.
bool get_group_evolve_flag(int gi);

// True iff every level of every patch sits at the same subcycling iteration,
// i.e. the checkpoint is time-aligned. Always true without subcycling. When
// false, the coarse source bands hold the in-progress coarse step (u(t_n) and
// the stage derivatives) that must be serialized.
bool all_levels_synchronized();

// True when (patch, level) is a coarse level ahead of one of its children in
// the checkpoint being recovered, so its evolved groups must carry olds/kss_*.
// Reads ghext->recovered_level_iterations; a missing entry (checkpoint without
// iteration_num/den) means time-aligned, hence false. Always false without
// subcycling and on the finest level. Only meaningful during RecoverGH, while
// the recovered iterations are still populated.
bool recovered_level_needs_rk_bands(int patch, int level);

// True when the flux register of the pair (level, level + 1) is live in the
// checkpoint being recovered: level `level` has begun a coarse step whose
// reflux has not yet been applied, i.e. some finer level is behind it, so the
// checkpoint must carry its freg_* bands. Same preconditions as
// recovered_level_needs_rk_bands.
bool recovered_flux_register_is_live(int patch, int level);

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

// The source band that level `level` fills as a parent: owned by the child
// level's GroupData. Null on the finest level, for groups that are not
// integrated, and where the coarse-fine footprint is empty. For
// flux_register, null unless the child owns a register for this group (an
// integrated group with a fluxes= tag under do_reflux).
amrex::MultiFab *rk_source_band(int patch, int level, int gi, band_kind kind,
                                int stage = -1);

// Monotonically increasing counter. Incremented whenever the AMR grid
// hierarchy is invalidated (regridding, recovery). Starts at 0.
extern std::atomic<CCTK_INT> carpetx_epoch;

extern "C" CCTK_INT CarpetX_GetEpoch(void);

} // namespace CarpetX

#endif // #ifndef CARPETX_CARPETX_DRIVER_HXX
