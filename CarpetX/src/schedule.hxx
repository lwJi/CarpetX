#ifndef CARPETX_CARPETX_SCHEDULE_HXX
#define CARPETX_CARPETX_SCHEDULE_HXX

#include "internal.hxx"

#include "driver.hxx"
#include "loop.hxx"

#include <cctk.h>
#include <cctk_Schedule.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <functional>
#include <optional>
#include <span>
#include <type_traits>
#include <vector>

namespace CarpetX {
using namespace Loop;

int Initialise(tFleshConfig *config);
int Evolve(tFleshConfig *config);
int Shutdown(tFleshConfig *config);

void RestrictNoPoison(const cGH *cctkGH, int min_level, int max_level,
                      std::span<const gi_t> groups);

struct active_levels_t;

int SyncGroupsByDirI(const active_levels_t &active_levels,
                     const cGH *restrict cctkGH, std::span<const gi_t> groups);

int SyncGroupsByDirISubcycling(const active_levels_t &active_levels,
                               const cGH *restrict cctkGH,
                               std::span<const gi_t> groups);

int SyncGroups(const active_levels_t &active_levels, const cGH *restrict cctkGH,
               std::span<const gi_t> groups);

int SyncGroupsOnActiveLevels(const cGH *restrict cctkGH, int numgroups,
                             const int *groups, const int *directions);

int SyncGroupsByDirIProlongateOnly(const active_levels_t &active_levels,
                                   const cGH *restrict cctkGH,
                                   std::span<const gi_t> groups, int tl = -1);

int SyncGroupsByDirIProlongateOnlyAligned(const active_levels_t &active_levels,
                                          const cGH *restrict cctkGH,
                                          std::span<const gi_t> groups,
                                          int tl = -1);

int SyncGroupsByDirIGhostOnly(const active_levels_t &active_levels,
                              const cGH *restrict cctkGH,
                              std::span<const gi_t> groups, int tl = -1);

int CallFunction(void *function, cFunctionData *attribute, void *data);

int GroupStorageIncrease(const cGH *cctkGH, int n_groups, const int *groups,
                         const int *tls, int *status);
int GroupStorageDecrease(const cGH *cctkGH, int n_groups, const int *groups,
                         const int *tls, int *status);
int EnableGroupStorage(const cGH *cctkGH, const char *groupname);
int DisableGroupStorage(const cGH *cctkGH, const char *groupname);
int QueryGroupStorageB(const cGH *cctkGH, int gi, const char *groupname);

////////////////////////////////////////////////////////////////////////////////

struct active_levels_t {
  int min_level, max_level;
  int min_patch, max_patch;

  active_levels_t(const int min_level, const int max_level, const int min_patch,
                  const int max_patch);
  active_levels_t(const int min_level, const int max_level);
  active_levels_t();
  explicit active_levels_t(window_t window);
  window_t window() const;

private:
  void assert_consistent_iterations() const;

public:
  // Loop over all active patches of all active levels from coarsest
  // to finest
  void loop_coarse_to_fine(
      const std::function<void(GHExt::PatchData::LevelData &level)> &kernel)
      const;

  // Loop over all active patches of all active levels from finest to
  // coarsest
  void loop_fine_to_coarse(
      const std::function<void(GHExt::PatchData::LevelData &level)> &kernel)
      const;

  // Loop over all active patches of all active levels serially
  void loop_serially(
      const std::function<void(GHExt::PatchData::LevelData &level)> &kernel)
      const {
    loop_coarse_to_fine(kernel);
  }

  // Loop over all components of all active patches of all active
  // levels in parallel
  void loop_parallel(
      const std::function<void(int patch, int level, int index, int component,
                               const cGH *cctkGH)> &kernel) const;

  // Loop over all components of all active patches of all active
  // levels serially
  void loop_serially(
      const std::function<void(int patch, int level, int index, int component,
                               const cGH *cctkGH)> &kernel) const;
};

const std::optional<active_levels_t> &scheduled_levels();

////////////////////////////////////////////////////////////////////////////////

// Like an MFIter, but does not support iteration, instead it can be copied
struct MFPointer {
  int m_index;
  amrex::Box m_validbox; // interior of component
  amrex::Box m_tilebox;  // interior of tile

  MFPointer() = delete;
  MFPointer(const MFPointer &) = default;
  MFPointer(MFPointer &&) = default;
  MFPointer &operator=(const MFPointer &) = default;
  MFPointer &operator=(MFPointer &&) = default;
  MFPointer(const amrex::MFIter &mfi)
      : m_index((assert(mfi.isValid()), mfi.index())),
        m_validbox(mfi.validbox()), m_tilebox(mfi.tilebox()) {}

  constexpr int index() const noexcept { return m_index; }

  constexpr amrex::Box validbox() const noexcept { return m_validbox; }

  constexpr amrex::Box tilebox() const noexcept { return m_tilebox; }

  const amrex::Box fabbox(const amrex::IntVect &ng) const noexcept {
    return amrex::grow(validbox(), ng);
  }

  const amrex::Box growntilebox(const amrex::IntVect &ng) const noexcept {
    amrex::Box bx = tilebox();
    const amrex::Box &vbx = validbox();
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
      if (bx.smallEnd(d) == vbx.smallEnd(d))
        bx.growLo(d, ng[d]);
      if (bx.bigEnd(d) == vbx.bigEnd(d))
        bx.growHi(d, ng[d]);
    }
    return bx;
  }
};

struct GridDesc : GridDescBase {

  GridDesc() = delete;
  GridDesc(const GHExt::PatchData::LevelData &leveldata, const MFPointer &mfp);
  GridDesc(const GHExt::PatchData::LevelData &leveldata,
           const int global_component);
  GridDesc(const cGH *cctkGH) : GridDescBase(cctkGH) {}
};

// TODO: remove this
struct GridPtrDesc : GridDesc {
  amrex::Dim3 cactus_offset;

  GridPtrDesc() = delete;
  GridPtrDesc(const GHExt::PatchData::LevelData &leveldata,
              const MFPointer &mfp);

  template <typename T> T *ptr(const amrex::Array4<T> &vars, int vi) const {
    return vars.ptr(cactus_offset.x, cactus_offset.y, cactus_offset.z, vi);
  }
  template <typename T>
  T &idx(const amrex::Array4<T> &vars, int i, int j, int k, int vi) const {
    return vars(cactus_offset.x + i, cactus_offset.y + i, cactus_offset.z + j,
                vi);
  }
};

struct GridPtrDesc1 : GridDesc {
  amrex::Dim3 cactus_offset;
  std::array<int, dim> gimin, gimax;
  std::array<int, dim> gash;

  GridPtrDesc1() = delete;
  GridPtrDesc1(const GridPtrDesc1 &) = delete;
  GridPtrDesc1 &operator=(const GridPtrDesc1 &) = delete;

  GridPtrDesc1(const GHExt::PatchData::LevelData &leveldata,
               const GHExt::PatchData::LevelData::GroupData &groupdata,
               const MFPointer &mfp);
  GridPtrDesc1(const GHExt::PatchData::LevelData &leveldata,
               const GHExt::PatchData::LevelData::GroupData &groupdata,
               int component);

  template <typename T> T *ptr(const amrex::Array4<T> &vars, int vi) const {
    return vars.ptr(cactus_offset.x + gimin[0], cactus_offset.y + gimin[1],
                    cactus_offset.z + gimin[2], vi);
  }
  template <typename T>
  T &idx(const amrex::Array4<T> &vars, int i, int j, int k, int vi) const {
    return vars(cactus_offset.x + gimin[0] + i, cactus_offset.y + gimin[1] + j,
                cactus_offset.z + gimin[2] + k, vi);
  }

  template <typename T>
  GF3D1<T> gf3d(const amrex::Array4<T> &vars, int vi) const {
    return GF3D1<T>(ptr(vars, vi), gimin, gimax, gash);
  }

  friend std::ostream &operator<<(std::ostream &os, const GridPtrDesc1 &p) {
    os << "GridPtrDesc1{" << (const GridDescBase &)p << ", "
       << "cactus_offset:"
       << "{" << p.cactus_offset.x << "," << p.cactus_offset.y << ","
       << p.cactus_offset.z << "}, "
       << "gimin:"
       << "{" << p.gimin[0] << "," << p.gimin[1] << "," << p.gimin[2] << "}, "
       << "gimax:"
       << "{" << p.gimax[0] << "," << p.gimax[1] << "," << p.gimax[2] << "}, "
       << "gash:"
       << "{" << p.gash[0] << "," << p.gash[1] << "," << p.gash[2] << "}";
    return os;
  }
};

////////////////////////////////////////////////////////////////////////////////

cGH *copy_cctkGH(const cGH *restrict const sourceGH);
void delete_cctkGH(cGH *cctkGH);

bool in_local_mode(const cGH *restrict cctkGH);
bool in_patch_mode(const cGH *restrict cctkGH);
bool in_level_mode(const cGH *restrict cctkGH);
bool in_global_mode(const cGH *restrict cctkGH);
bool in_meta_mode(const cGH *restrict cctkGH);

void update_cctkGH(cGH *restrict cctkGH, const cGH *restrict sourceGH);
void update_cctkGHs(cGH *restrict cctkGH);
void enter_global_mode(cGH *restrict cctkGH);
void leave_global_mode(cGH *restrict cctkGH);
void enter_level_mode(cGH *restrict cctkGH, int level);
void leave_level_mode(cGH *restrict cctkGH, int level);
void enter_patch_mode(cGH *restrict cctkGH,
                      const GHExt::PatchData::LevelData &restrict leveldata);
void leave_patch_mode(cGH *restrict cctkGH,
                      const GHExt::PatchData::LevelData &restrict leveldata);
void enter_local_mode(cGH *restrict cctkGH,
                      const GHExt::PatchData::LevelData &restrict leveldata,
                      const MFPointer &mfp);
void leave_local_mode(cGH *restrict cctkGH,
                      const GHExt::PatchData::LevelData &restrict leveldata,
                      const MFPointer &mfp);

void synchronize();

} // namespace CarpetX

#endif // #ifndef CARPETX_CARPETX_SCHEDULE_HXX
