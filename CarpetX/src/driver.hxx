#ifndef CARPETX_CARPETX_DRIVER_HXX
#define CARPETX_CARPETX_DRIVER_HXX

#include "loop.hxx"
#include "valid.hxx"

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
#include <string>
#include <type_traits>
#include <vector>

namespace CarpetX {
using namespace Arith;

using Loop::dim;

using rat64 = rational<int64_t>;

inline constexpr int max_num_rk_stages = 4;

enum class symmetry_t {
  none,
  interpatch,
  periodic,
  reflection,
};
std::ostream &operator<<(std::ostream &os, const symmetry_t symmetry);

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
  std::vector<cctkGHptr> level_cctkGHs;

  struct CommonGroupData {
    std::string groupname;
    int groupindex;
    int firstvarindex;
    int numvars;

    bool do_checkpoint;
    bool do_evolve;
    bool do_restrict;
    bool restrict_on_sync;

    std::vector<std::vector<why_valid_t> > valid;

    friend YAML::Emitter &operator<<(YAML::Emitter &yaml,
                                     const CommonGroupData &commongroupdata);
  };

  struct GlobalData {
    class AnyTypeVector {

    public:
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

    struct ArrayGroupData : public CommonGroupData {
      std::vector<AnyTypeVector> data;
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
    std::vector<std::unique_ptr<ArrayGroupData> > arraygroupdata;

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

      bool is_subcycling_level;

      rat64 iteration, delta_iteration;

      int last_step_iteration;

      int last_finalized_iteration;

      std::unique_ptr<amrex::FabArrayBase> fab;

      mutable std::array<std::unique_ptr<amrex::MultiFab>, dim> face_area;
      mutable std::unique_ptr<amrex::MultiFab> cell_volume;

      cctkGHptr patch_cctkGH;
      std::vector<cctkGHptr> local_cctkGHs;

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

        void apply_boundary_conditions(amrex::MultiFab &mfab) const;

        std::vector<std::unique_ptr<amrex::MultiFab> > mfab;

        mutable std::array<std::unique_ptr<amrex::MultiFab>, max_num_rk_stages>
            ks_source_band;

        mutable std::unique_ptr<amrex::MultiFab> old_source_band;

        mutable std::unique_ptr<amrex::MultiFab> rk_crse_patch, rk_fine_patch;

        std::unique_ptr<amrex::FluxRegister> freg;
        std::unique_ptr<amrex::FluxRegister> freg_scratch;
        std::array<int, dim> fluxes;

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
      std::vector<std::unique_ptr<GroupData> > groupdata;

      friend YAML::Emitter &operator<<(YAML::Emitter &yaml,
                                       const LevelData &leveldata);
    };
    std::vector<LevelData> leveldata;

    friend YAML::Emitter &operator<<(YAML::Emitter &yaml,
                                     const PatchData &patchdata);
  };
  std::vector<PatchData> patchdata;

  std::vector<int> active_timelevels;
  bool storage_frozen = false;

  bool use_subcycling = false;

  int recovered_iteration = -1;

  bool in_recovery_traversal = false;

  const char *tail_bin = nullptr;
  const cFunctionData *last_called_function = nullptr;

  bool do_reflux = false;

  int num_rk_stages = 4;

  std::vector<bool> rk_integrated_group;

  struct recovered_level_t {
    std::optional<rat64> iteration;
    std::optional<int> last_step_iteration;
  };
  std::vector<std::vector<recovered_level_t> > recovered_levels;

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

void warn_inert_flux_tags();

bool get_group_checkpoint_flag(int gi);

bool get_group_evolve_flag(int gi);

bool all_levels_synchronized();

bool recovered_finer_level_is_behind(int patch, int level);

enum class band_kind { ks_source, old_source, flux_register };

std::string subcycling_band_tag(band_kind kind, int stage = -1);

amrex::MultiFab *rk_source_band(int patch, int level, int gi, band_kind kind,
                                int stage = -1);

extern std::atomic<CCTK_INT> carpetx_epoch;

extern "C" CCTK_INT CarpetX_GetEpoch(void);

} // namespace CarpetX

#endif
