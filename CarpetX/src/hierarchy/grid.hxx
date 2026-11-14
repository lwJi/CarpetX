#ifndef CARPETX_CARPETX_HIERARCHY_GRID_HXX
#define CARPETX_CARPETX_HIERARCHY_GRID_HXX

#include "internal.hxx"

#include "hierarchy/model.hxx"
#include <loop.hxx>

#include <cctk.h>

#include <array>
#include <cassert>
#include <ostream>

namespace CarpetX {

inline amrex::Orientation orient(int d, int f) {
  return amrex::Orientation(d, amrex::Orientation::Side(f));
}

struct MFPointer {
  component_t m_index;
  amrex::Box m_validbox;
  amrex::Box m_tilebox;

  MFPointer() = delete;
  MFPointer(const MFPointer &) = default;
  MFPointer(MFPointer &&) = default;
  MFPointer &operator=(const MFPointer &) = default;
  MFPointer &operator=(MFPointer &&) = default;
  MFPointer(const amrex::MFIter &mfi)
      : m_index((assert(mfi.isValid()), component_t{mfi.index()})),
        m_validbox(mfi.validbox()), m_tilebox(mfi.tilebox()) {}

  constexpr component_t index() const noexcept { return m_index; }

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

struct GridDesc : Loop::GridDescBase {

  GridDesc() = delete;
  GridDesc(const GHExt::PatchData::LevelData &leveldata, const MFPointer &mfp);
  GridDesc(const GHExt::PatchData::LevelData &leveldata, component_t component);
  GridDesc(const cGH *cctkGH) : Loop::GridDescBase(cctkGH) {}
};

struct GridPtrDesc : GridDesc {
  amrex::Dim3 cactus_offset;

  GridPtrDesc() = delete;
  GridPtrDesc(const GHExt::PatchData::LevelData &leveldata,
              const MFPointer &mfp);

  template <typename T>
  T *ptr(const amrex::Array4<T> &vars, const vi_t vi) const {
    return vars.ptr(cactus_offset.x, cactus_offset.y, cactus_offset.z,
                    vi.index());
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
               component_t component);

  template <typename T>
  T *ptr(const amrex::Array4<T> &vars, const vi_t vi) const {
    return vars.ptr(cactus_offset.x + gimin[0], cactus_offset.y + gimin[1],
                    cactus_offset.z + gimin[2], vi.index());
  }

  template <typename T>
  Loop::GF3D1<T> gf3d(const amrex::Array4<T> &vars, const vi_t vi) const {
    return Loop::GF3D1<T>(ptr(vars, vi), gimin, gimax, gash);
  }

  friend std::ostream &operator<<(std::ostream &os, const GridPtrDesc1 &p) {
    os << "GridPtrDesc1{" << (const Loop::GridDescBase &)p << ", "
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

} // namespace CarpetX

#endif
