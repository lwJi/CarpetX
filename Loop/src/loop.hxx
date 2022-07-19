#ifndef LOOP_HXX
#define LOOP_HXX

#include <AMReX_FArrayBox.H>

#define CCTK_DEVICE AMREX_GPU_DEVICE
#define CCTK_HOST AMREX_GPU_HOST

#include <simd.hxx>
#include <vect.hxx>

#include <cctk.h>

#include <algorithm>
#include <array>
// #include <cassert>
#include <cstddef>
// #include <cstdlib>
#include <climits>
// #include <cstring>
// #include <functional>
// #include <iostream>
// #include <limits>
#include <ostream>
#include <string>
#include <type_traits>

namespace Loop {

using Arith::vect;

template <typename F> CCTK_ATTRIBUTE_NOINLINE auto noinline(const F &f) {
  return f();
}

constexpr int dim = 3;

// Value for undefined cctkGH entries
// Note: Don't use a negative value, which tends to leave bugs undetected. Large
// positive values often lead to segfaults, exposing bugs.
constexpr int undefined = (INT_MAX / 2 + 1) + 666;

enum class where_t { everywhere, interior, boundary, ghosts_inclusive, ghosts };
std::ostream &operator<<(std::ostream &os, const where_t where);

// TODO: Add this `template <int CI, int CJ, int CK>` here, and to `GF3D2` etc.
struct PointDesc {
  // static constexpr vect<int, dim> centering{CI, CJ, CK};
  int imin, imax;
  int i, j, k;
  CCTK_REAL x, y, z;
  CCTK_REAL dx, dy, dz;
  int idx;
  static constexpr int di = 1;
  int dj; // stride in the j direction
  int dk; // stride in the k direction
  int np; // number of grid points
  vect<int, dim> I;
  vect<int, dim> NI; // outward boundary normal, or zero
  vect<int, dim> I0; // nearest interior point
  static constexpr vect<vect<int, dim>, dim> DI = {vect<int, dim>::unit(0),
                                                   vect<int, dim>::unit(1),
                                                   vect<int, dim>::unit(2)};
  vect<CCTK_REAL, dim> X;
  vect<CCTK_REAL, dim> DX;

  friend std::ostream &operator<<(std::ostream &os, const PointDesc &p) {
    os << "PointDesc{"
       << "imin,imax:{" << p.imin << "," << p.imax << "}, "
       << "ijk:"
       << "{" << p.i << "," << p.j << "," << p.k << "}, "
       << "xyz:"
       << "{" << p.x << "," << p.y << "," << p.z << "}, "
       << "dxyz:"
       << "{" << p.dx << "," << p.dy << "," << p.dz << "}, "
       << "idx:" << p.idx << ", "
       << "dijk:"
       << "{" << p.di << "," << p.dj << "," << p.dk << "},"
       << "np:" << p.np << ","
       << "nijk:"
       << "{" << p.NI[0] << "," << p.NI[1] << "," << p.NI[2] << "}"
       << "i0ijk:"
       << "{" << p.I0[0] << "," << p.I0[1] << "," << p.I0[2] << "}"
       << "}";
    return os;
  }
};

struct GridDescBase {
  std::array<int, dim> gsh;
  std::array<int, dim> lbnd, ubnd;
  std::array<int, dim> lsh;
  std::array<int, dim> ash;
  std::array<int, 2 * dim> bbox;
  std::array<int, dim> nghostzones;
  std::array<int, dim> tmin, tmax;

  // for current level
  std::array<CCTK_REAL, dim> x0;
  std::array<CCTK_REAL, dim> dx;

  template <typename T, std::size_t N>
  static void output(std::ostream &os, const std::string &str,
                     const std::array<T, N> &arr) {
    os << str << ":[";
    for (std::size_t n = 0; n < N; ++n) {
      if (n > 0)
        os << ",";
      os << arr[n];
    }
    os << "]";
  }
  friend std::ostream &operator<<(std::ostream &os, const GridDescBase &grid) {
    os << "GridDescBase{";
    output(os, "gsh", grid.gsh);
    output(os, ",lbnd", grid.lbnd);
    output(os, ",ubnd", grid.ubnd);
    output(os, ",lsh", grid.lsh);
    output(os, ",bbox", grid.bbox);
    output(os, ",nghostzones", grid.nghostzones);
    output(os, ",tmin", grid.tmin);
    output(os, ",tmax", grid.tmax);
    os << "}";
    return os;
  }

protected:
  GridDescBase();

public:
  GridDescBase(const cGH *cctkGH);

  template <int CI, int CJ, int CK>
  constexpr CCTK_ATTRIBUTE_ALWAYS_INLINE CCTK_DEVICE CCTK_HOST PointDesc
  point_desc(const vect<int, dim> &restrict NI,
             const vect<int, dim> &restrict I0, const int imin, const int imax,
             const int i, const int j, const int k) const {
    constexpr int di = 1;
    const int dj = di * (ash[0] - CI);
    const int dk = dj * (ash[1] - CJ);
    const int np = dk * (ash[2] - CK);
    const CCTK_REAL x = x0[0] + (lbnd[0] + i - CCTK_REAL(!CI) / 2) * dx[0];
    const CCTK_REAL y = x0[1] + (lbnd[1] + j - CCTK_REAL(!CJ) / 2) * dx[1];
    const CCTK_REAL z = x0[2] + (lbnd[2] + k - CCTK_REAL(!CK) / 2) * dx[2];
    const int idx = i * di + j * dj + k * dk;
    return PointDesc{imin,      imax,         //
                     i,         j,     k,     //
                     x,         y,     z,     //
                     dx[0],     dx[1], dx[2], //
                     idx,                     //
                     dj,        dk,    np,    //
                     {i, j, k},               //
                     NI,                      //
                     I0,                      //
                     {x, y, z},               //
                     dx};
  }

  template <int CI, int CJ, int CK>
  constexpr CCTK_ATTRIBUTE_ALWAYS_INLINE CCTK_DEVICE CCTK_HOST PointDesc
  point_desc(const PointDesc &p) const {
    return point_desc<CI, CJ, CK>(p.NI, p.I0, p.imin, p.imax, p.i, p.j, p.k);
  }

  // Loop over a given box
  template <int CI, int CJ, int CK, int VS = 1, typename F>
  // inline CCTK_ATTRIBUTE_ALWAYS_INLINE
  CCTK_ATTRIBUTE_NOINLINE void
  loop_box(const F &f, const std::array<int, dim> &restrict imin,
           const std::array<int, dim> &restrict imax,
           const std::array<int, dim> &restrict inormal) const {
    // TODO: convert array to vect (and everywhere nearby)
    static_assert(CI == 0 || CI == 1, "");
    static_assert(CJ == 0 || CJ == 1, "");
    static_assert(CK == 0 || CK == 1, "");
    static_assert(VS > 0, "");

    for (int d = 0; d < dim; ++d)
      if (imin[d] >= imax[d])
        return;

    bool has_normal = false;
    for (int d = 0; d < dim; ++d)
      has_normal |= inormal[d] != 0;

    if (!has_normal) {
      // Loop over an interior box

      constexpr std::array<int, dim> izero = {0, 0, 0};
      for (int k = imin[2]; k < imax[2]; ++k) {
        for (int j = imin[1]; j < imax[1]; ++j) {
#pragma omp simd
          for (int i = imin[0]; i < imax[0]; i += VS) {
            f(point_desc<CI, CJ, CK>(izero, izero, imin[0], imax[0], i, j, k));
          }
        }
      }

    } else {
      // Loop over a box at a boundary

      for (int k = imin[2]; k < imax[2]; ++k) {
        for (int j = imin[1]; j < imax[1]; ++j) {
#pragma omp simd
          for (int i = imin[0]; i < imax[0]; i += VS) {

            const std::array<int, dim> I = {i, j, k};
            std::array<int, dim> I0;
            for (int d = 0; d < dim; ++d) {
              if (inormal[d] == 0)
                // interior
                I0[d] = I[d];
              else if (inormal[d] < 0)
                // left boundary
                I0[d] = imax[d];
              else
                // right boundary
                I0[d] = imin[d] - 1;
            }
            f(point_desc<CI, CJ, CK>(inormal, I0, imin[0], imax[0], i, j, k));
          }
        }
      }
    }
  }

  // Box including all points
  template <int CI, int CJ, int CK>
  void box_all(const std::array<int, dim> &group_nghostzones,
               vect<int, dim> &restrict imin,
               vect<int, dim> &restrict imax) const {
    const std::array<int, dim> offset{CI, CJ, CK};
    for (int d = 0; d < dim; ++d) {
      int ghost_offset = nghostzones[d] - group_nghostzones[d];
      using std::max, std::min;
      imin[d] = max(tmin[d], ghost_offset);
      imax[d] = min(tmax[d], lsh[d] - offset[d] - ghost_offset);
    }
  }

  // Box including all interior points
  template <int CI, int CJ, int CK>
  void box_int(const std::array<int, dim> &group_nghostzones,
               vect<int, dim> &restrict imin,
               vect<int, dim> &restrict imax) const {
    // TODO: call box_int instead
    const std::array<int, dim> offset{CI, CJ, CK};
    for (int d = 0; d < dim; ++d) {
      using std::max, std::min;
      imin[d] = max(tmin[d], nghostzones[d]);
      imax[d] = min(tmax[d], lsh[d] - offset[d] - nghostzones[d]);
    }
  }

  // Loop over all points
  template <int CI, int CJ, int CK, typename F>
  inline CCTK_ATTRIBUTE_ALWAYS_INLINE void
  loop_all(const std::array<int, dim> &group_nghostzones, const F &f) const {
    // TODO: call box_all instead
    const std::array<int, dim> offset{CI, CJ, CK};
    std::array<int, dim> imin, imax;
    for (int d = 0; d < dim; ++d) {
      int ghost_offset = nghostzones[d] - group_nghostzones[d];
      using std::max, std::min;
      imin[d] = max(tmin[d], ghost_offset);
      imax[d] = min(tmax[d], lsh[d] - offset[d] - ghost_offset);
    }
    const std::array<int, dim> inormal{0, 0, 0};

    loop_box<CI, CJ, CK>(f, imin, imax, inormal);
  }

  // Loop over all interior points
  template <int CI, int CJ, int CK, typename F>
  inline CCTK_ATTRIBUTE_ALWAYS_INLINE void
  loop_int(const std::array<int, dim> &group_nghostzones, const F &f) const {
    // TODO: call box_int instead
    const std::array<int, dim> offset{CI, CJ, CK};
    std::array<int, dim> imin, imax;
    for (int d = 0; d < dim; ++d) {
      using std::max, std::min;
      imin[d] = max(tmin[d], nghostzones[d]);
      imax[d] = min(tmax[d], lsh[d] - offset[d] - nghostzones[d]);
    }
    const std::array<int, dim> inormal{0, 0, 0};

    loop_box<CI, CJ, CK>(f, imin, imax, inormal);
  }

  // Loop over a part of the domain. Loop over the interior first,
  // then faces, then edges, then corners.
  template <int CI, int CJ, int CK, typename F>
  inline CCTK_ATTRIBUTE_ALWAYS_INLINE void loop_there(
      const std::array<int, dim> &group_nghostzones,
      const std::array<std::array<std::array<bool, dim>, dim>, dim> &there,
      const F &f) const {
    const std::array<int, dim> offset{CI, CJ, CK};

    for (int rank = dim; rank >= 0; --rank) {

      for (int nk = -1; nk <= +1; ++nk) {
        for (int nj = -1; nj <= +1; ++nj) {
          for (int ni = -1; ni <= +1; ++ni) {
            if ((ni == 0) + (nj == 0) + (nk == 0) == rank) {

              if (there[ni + 1][nj + 1][nk + 1]) {

                const std::array<int, dim> inormal{ni, nj, nk};

                std::array<int, dim> imin, imax;
                for (int d = 0; d < dim; ++d) {
                  const int ghost_offset =
                      nghostzones[d] - group_nghostzones[d];
                  const int begin_bnd = ghost_offset;
                  const int begin_int = nghostzones[d];
                  const int end_int = lsh[d] - offset[d] - nghostzones[d];
                  const int end_bnd = lsh[d] - offset[d] - ghost_offset;
                  switch (inormal[d]) {
                  case -1: // lower boundary
                    imin[d] = begin_bnd;
                    imax[d] = begin_int;
                    break;
                  case 0: // interior
                    imin[d] = begin_int;
                    imax[d] = end_int;
                    break;
                  case +1: // upper boundary
                    imin[d] = end_int;
                    imax[d] = end_bnd;
                    break;
                  default:
                    assert(0);
                  }

                  imin[d] = std::max(tmin[d], imin[d]);
                  imax[d] = std::min(tmax[d], imax[d]);
                }

#ifdef CCTK_DEBUG
                bool isempty = false;
                for (int d = 0; d < dim; ++d)
                  isempty |= imin[d] >= imax[d];
                if (!isempty) {
                  vect<int, dim> all_imin, all_imax;
                  box_all<CI, CJ, CK>(group_nghostzones, all_imin, all_imax);
                  vect<int, dim> int_imin, int_imax;
                  box_int<CI, CJ, CK>(group_nghostzones, int_imin, int_imax);
                  for (int d = 0; d < dim; ++d) {
                    assert(all_imin[d] <= imin[d]);
                    assert(imax[d] <= all_imax[d]);
                  }
                  bool overlaps = true;
                  for (int d = 0; d < dim; ++d)
                    overlaps &=
                        !(imax[d] <= int_imin[d] || imin[d] >= int_imax[d]);
                  assert(!overlaps);
                }
#endif

                loop_box<CI, CJ, CK>(f, imin, imax, inormal);
              }
            } // if rank
          }
        }
      }

    } // for rank
  }

  // Loop over all outer boundary points. This excludes ghost faces, but
  // includes ghost edges/corners on non-ghost faces. Loop over faces first,
  // then edges, then corners.
  template <int CI, int CJ, int CK, typename F>
  inline CCTK_ATTRIBUTE_ALWAYS_INLINE void
  loop_bnd(const std::array<int, dim> &group_nghostzones, const F &f) const {
    const std::array<int, dim> offset{CI, CJ, CK};

    for (int rank = dim - 1; rank >= 0; --rank) {

      for (int nk = -1; nk <= +1; ++nk) {
        for (int nj = -1; nj <= +1; ++nj) {
          for (int ni = -1; ni <= +1; ++ni) {
            if ((ni == 0) + (nj == 0) + (nk == 0) == rank) {

              if ((ni != 0 && bbox[0 + (ni == -1 ? 0 : 1)]) ||
                  (nj != 0 && bbox[2 + (nj == -1 ? 0 : 1)]) ||
                  (nk != 0 && bbox[4 + (nk == -1 ? 0 : 1)])) {

                const std::array<int, dim> inormal{ni, nj, nk};

                std::array<int, dim> imin, imax;
                for (int d = 0; d < dim; ++d) {
                  const int ghost_offset =
                      nghostzones[d] - group_nghostzones[d];
                  const int begin_bnd = ghost_offset;
                  const int begin_int = nghostzones[d];
                  const int end_int = lsh[d] - offset[d] - nghostzones[d];
                  const int end_bnd = lsh[d] - offset[d] - ghost_offset;
                  switch (inormal[d]) {
                  case -1: // lower boundary
                    imin[d] = begin_bnd;
                    imax[d] = begin_int;
                    break;
                  case 0: // interior
                    imin[d] = begin_int;
                    imax[d] = end_int;
                    break;
                  case +1: // upper boundary
                    imin[d] = end_int;
                    imax[d] = end_bnd;
                    break;
                  default:
                    assert(0);
                  }

                  imin[d] = std::max(tmin[d], imin[d]);
                  imax[d] = std::min(tmax[d], imax[d]);
                }

#ifdef CCTK_DEBUG
                bool isempty = false;
                for (int d = 0; d < dim; ++d)
                  isempty |= imin[d] >= imax[d];
                if (!isempty) {
                  vect<int, dim> all_imin, all_imax;
                  box_all<CI, CJ, CK>(group_nghostzones, all_imin, all_imax);
                  vect<int, dim> int_imin, int_imax;
                  box_int<CI, CJ, CK>(group_nghostzones, int_imin, int_imax);
                  for (int d = 0; d < dim; ++d) {
                    assert(all_imin[d] <= imin[d]);
                    assert(imax[d] <= all_imax[d]);
                  }
                  bool overlaps = true;
                  for (int d = 0; d < dim; ++d)
                    overlaps &=
                        !(imax[d] <= int_imin[d] || imin[d] >= int_imax[d]);
                  assert(!overlaps);
                }
#endif

                loop_box<CI, CJ, CK>(f, imin, imax, inormal);
              }
            } // if rank
          }
        }
      }

    } // for rank
  }

  // Loop over all outer ghost points. This includes ghost edges/corners on
  // non-ghost faces. Loop over faces first, then edges, then corners.
  template <int CI, int CJ, int CK, typename F>
  inline CCTK_ATTRIBUTE_ALWAYS_INLINE void
  loop_ghosts_inclusive(const std::array<int, dim> &group_nghostzones,
                        const F &f) const {
    const std::array<int, dim> offset{CI, CJ, CK};

    for (int rank = dim - 1; rank >= 0; --rank) {

      for (int nk = -1; nk <= +1; ++nk) {
        for (int nj = -1; nj <= +1; ++nj) {
          for (int ni = -1; ni <= +1; ++ni) {
            if ((ni == 0) + (nj == 0) + (nk == 0) == rank) {

              if ((ni != 0 && !bbox[0 + (ni == -1 ? 0 : 1)]) ||
                  (nj != 0 && !bbox[2 + (nj == -1 ? 0 : 1)]) ||
                  (nk != 0 && !bbox[4 + (nk == -1 ? 0 : 1)])) {

                const std::array<int, dim> inormal{ni, nj, nk};

                std::array<int, dim> imin, imax;
                for (int d = 0; d < dim; ++d) {
                  const int ghost_offset =
                      nghostzones[d] - group_nghostzones[d];
                  const int begin_bnd = ghost_offset;
                  const int begin_int = nghostzones[d];
                  const int end_int = lsh[d] - offset[d] - nghostzones[d];
                  const int end_bnd = lsh[d] - offset[d] - ghost_offset;
                  switch (inormal[d]) {
                  case -1: // lower boundary
                    imin[d] = begin_bnd;
                    imax[d] = begin_int;
                    break;
                  case 0: // interior
                    imin[d] = begin_int;
                    imax[d] = end_int;
                    break;
                  case +1: // upper boundary
                    imin[d] = end_int;
                    imax[d] = end_bnd;
                    break;
                  default:
                    assert(0);
                  }

                  imin[d] = std::max(tmin[d], imin[d]);
                  imax[d] = std::min(tmax[d], imax[d]);
                }

#ifdef CCTK_DEBUG
                bool isempty = false;
                for (int d = 0; d < dim; ++d)
                  isempty |= imin[d] >= imax[d];
                if (!isempty) {
                  vect<int, dim> all_imin, all_imax;
                  box_all<CI, CJ, CK>(group_nghostzones, all_imin, all_imax);
                  vect<int, dim> int_imin, int_imax;
                  box_int<CI, CJ, CK>(group_nghostzones, int_imin, int_imax);
                  for (int d = 0; d < dim; ++d) {
                    assert(all_imin[d] <= imin[d]);
                    assert(imax[d] <= all_imax[d]);
                  }
                  bool overlaps = true;
                  for (int d = 0; d < dim; ++d)
                    overlaps &=
                        !(imax[d] <= int_imin[d] || imin[d] >= int_imax[d]);
                  assert(!overlaps);
                }
#endif

                loop_box<CI, CJ, CK>(f, imin, imax, inormal);
              }
            } // if rank
          }
        }
      }
    } // for rank
  }

  // Loop over all outer ghost points. This excludes ghost edges/corners on
  // non-ghost faces. Loop over faces first, then edges, then corners.
  template <int CI, int CJ, int CK, typename F>
  inline CCTK_ATTRIBUTE_ALWAYS_INLINE void
  loop_ghosts(const std::array<int, dim> &group_nghostzones, const F &f) const {
    const std::array<int, dim> offset{CI, CJ, CK};

    for (int rank = dim - 1; rank >= 0; --rank) {

      for (int nk = -1; nk <= +1; ++nk) {
        for (int nj = -1; nj <= +1; ++nj) {
          for (int ni = -1; ni <= +1; ++ni) {
            if ((ni == 0) + (nj == 0) + (nk == 0) == rank) {

              if ((ni == 0 || !bbox[0 + (ni == -1 ? 0 : 1)]) &&
                  (nj == 0 || !bbox[2 + (nj == -1 ? 0 : 1)]) &&
                  (nk == 0 || !bbox[4 + (nk == -1 ? 0 : 1)])) {

                const std::array<int, dim> inormal{ni, nj, nk};

                std::array<int, dim> imin, imax;
                for (int d = 0; d < dim; ++d) {
                  const int ghost_offset =
                      nghostzones[d] - group_nghostzones[d];
                  const int begin_bnd = ghost_offset;
                  const int begin_int = nghostzones[d];
                  const int end_int = lsh[d] - offset[d] - nghostzones[d];
                  const int end_bnd = lsh[d] - offset[d] - ghost_offset;
                  switch (inormal[d]) {
                  case -1: // lower boundary
                    imin[d] = begin_bnd;
                    imax[d] = begin_int;
                    break;
                  case 0: // interior
                    imin[d] = begin_int;
                    imax[d] = end_int;
                    break;
                  case +1: // upper boundary
                    imin[d] = end_int;
                    imax[d] = end_bnd;
                    break;
                  default:
                    assert(0);
                  }

                  imin[d] = std::max(tmin[d], imin[d]);
                  imax[d] = std::min(tmax[d], imax[d]);
                }

#ifdef CCTK_DEBUG
                bool isempty = false;
                for (int d = 0; d < dim; ++d)
                  isempty |= imin[d] >= imax[d];
                if (!isempty) {
                  vect<int, dim> all_imin, all_imax;
                  box_all<CI, CJ, CK>(group_nghostzones, all_imin, all_imax);
                  vect<int, dim> int_imin, int_imax;
                  box_int<CI, CJ, CK>(group_nghostzones, int_imin, int_imax);
                  for (int d = 0; d < dim; ++d) {
                    assert(all_imin[d] <= imin[d]);
                    assert(imax[d] <= all_imax[d]);
                  }
                  bool overlaps = true;
                  for (int d = 0; d < dim; ++d)
                    overlaps &=
                        !(imax[d] <= int_imin[d] || imin[d] >= int_imax[d]);
                  assert(!overlaps);
                }
#endif

                loop_box<CI, CJ, CK>(f, imin, imax, inormal);
              }
            } // if rank
          }
        }
      }
    } // for rank
  }

  template <int CI, int CJ, int CK, where_t where, typename F>
  inline CCTK_ATTRIBUTE_ALWAYS_INLINE
      std::enable_if_t<(where == where_t::everywhere), void>
      loop(const std::array<int, dim> &group_nghostzones, const F &f) const {
    loop_all<CI, CJ, CK>(group_nghostzones, f);
  }
  template <int CI, int CJ, int CK, where_t where, typename F>
  inline CCTK_ATTRIBUTE_ALWAYS_INLINE
      std::enable_if_t<(where == where_t::interior), void>
      loop(const std::array<int, dim> &group_nghostzones, const F &f) const {
    loop_int<CI, CJ, CK>(group_nghostzones, f);
  }
  template <int CI, int CJ, int CK, where_t where, typename F>
  inline CCTK_ATTRIBUTE_ALWAYS_INLINE
      std::enable_if_t<(where == where_t::boundary), void>
      loop(const std::array<int, dim> &group_nghostzones, const F &f) const {
    loop_bnd<CI, CJ, CK>(group_nghostzones, f);
  }
  template <int CI, int CJ, int CK, where_t where, typename F>
  inline CCTK_ATTRIBUTE_ALWAYS_INLINE
      std::enable_if_t<(where == where_t::ghosts_inclusive), void>
      loop(const std::array<int, dim> &group_nghostzones, const F &f) const {
    loop_ghosts_inclusive<CI, CJ, CK>(group_nghostzones, f);
  }
  template <int CI, int CJ, int CK, where_t where, typename F>
  inline CCTK_ATTRIBUTE_ALWAYS_INLINE
      std::enable_if_t<(where == where_t::ghosts), void>
      loop(const std::array<int, dim> &group_nghostzones, const F &f) const {
    loop_ghosts<CI, CJ, CK>(group_nghostzones, f);
  }

  template <int CI, int CJ, int CK, typename F>
  inline CCTK_ATTRIBUTE_ALWAYS_INLINE void
  loop(where_t where, const std::array<int, dim> &group_nghostzones,
       const F &f) const {
    switch (where) {
    case where_t::everywhere:
      return noinline([&] {
        return loop<CI, CJ, CK, where_t::everywhere>(group_nghostzones, f);
      });
    case where_t::interior:
      return noinline([&] {
        return loop<CI, CJ, CK, where_t::interior>(group_nghostzones, f);
      });
    case where_t::boundary:
      return noinline([&] {
        return loop<CI, CJ, CK, where_t::boundary>(group_nghostzones, f);
      });
    case where_t::ghosts_inclusive:
      return noinline([&] {
        return loop<CI, CJ, CK, where_t::ghosts_inclusive>(group_nghostzones,
                                                           f);
      });
    case where_t::ghosts:
      return noinline([&] {
        return loop<CI, CJ, CK, where_t::ghosts>(group_nghostzones, f);
      });
    default:
      assert(0);
    }
  }

  template <int CI, int CJ, int CK, typename F>
  inline CCTK_ATTRIBUTE_ALWAYS_INLINE void loop(where_t where,
                                                const F &f) const {
    loop<CI, CJ, CK>(where, nghostzones, f);
  }

  template <int CI, int CJ, int CK, where_t where, typename F>
  inline CCTK_ATTRIBUTE_ALWAYS_INLINE void loop(const F &f) const {
    loop<CI, CJ, CK, where>(nghostzones, f);
  }

  template <typename F>
  inline CCTK_ATTRIBUTE_ALWAYS_INLINE void
  loop_idx(where_t where, const std::array<int, dim> &indextype,
           const std::array<int, dim> &group_nghostzones, const F &f) const {
    switch (indextype[0] + 2 * indextype[1] + 4 * indextype[2]) {
    case 0b000:
      return noinline(
          [&] { return loop<0, 0, 0>(where, group_nghostzones, f); });
    case 0b001:
      return noinline(
          [&] { return loop<1, 0, 0>(where, group_nghostzones, f); });
    case 0b010:
      return noinline(
          [&] { return loop<0, 1, 0>(where, group_nghostzones, f); });
    case 0b011:
      return noinline(
          [&] { return loop<1, 1, 0>(where, group_nghostzones, f); });
    case 0b100:
      return noinline(
          [&] { return loop<0, 0, 1>(where, group_nghostzones, f); });
    case 0b101:
      return noinline(
          [&] { return loop<1, 0, 1>(where, group_nghostzones, f); });
    case 0b110:
      return noinline(
          [&] { return loop<0, 1, 1>(where, group_nghostzones, f); });
    case 0b111:
      return noinline(
          [&] { return loop<1, 1, 1>(where, group_nghostzones, f); });
    default:
      assert(0);
    }
  }

  template <typename F>
  inline CCTK_ATTRIBUTE_ALWAYS_INLINE void
  loop_idx(where_t where, const std::array<int, dim> &indextype,
           const F &f) const {
    loop_idx(where, indextype, nghostzones, f);
  }
};

template <typename F>
inline CCTK_ATTRIBUTE_ALWAYS_INLINE void
loop_idx(const cGH *cctkGH, where_t where,
         const std::array<int, dim> &indextype,
         const std::array<int, dim> &nghostzones, const F &f) {
  GridDescBase(cctkGH).loop_idx(where, indextype, nghostzones, f);
}

template <typename F>
inline CCTK_ATTRIBUTE_ALWAYS_INLINE void
loop_idx(const cGH *cctkGH, where_t where,
         const std::array<int, dim> &indextype, const F &f) {
  GridDescBase(cctkGH).loop_idx(where, indextype, f);
}

template <int CI, int CJ, int CK, where_t where, typename F>
inline CCTK_ATTRIBUTE_ALWAYS_INLINE void loop(const cGH *cctkGH, const F &f) {
  GridDescBase(cctkGH).loop<CI, CJ, CK, where>(f);
}

template <int CI, int CJ, int CK, typename F>
inline CCTK_ATTRIBUTE_ALWAYS_INLINE void loop(const cGH *cctkGH, where_t where,
                                              const F &f) {
  GridDescBase(cctkGH).loop<CI, CJ, CK>(where, f);
}

// Keep these for convenience
template <int CI, int CJ, int CK, typename F>
inline CCTK_ATTRIBUTE_ALWAYS_INLINE void loop_all(const cGH *cctkGH,
                                                  const F &f) {
  loop<CI, CJ, CK, where_t::everywhere>(cctkGH, f);
}

template <int CI, int CJ, int CK, typename F>
inline CCTK_ATTRIBUTE_ALWAYS_INLINE void loop_int(const cGH *cctkGH,
                                                  const F &f) {
  loop<CI, CJ, CK, where_t::interior>(cctkGH, f);
}

template <int CI, int CJ, int CK, typename F>
inline CCTK_ATTRIBUTE_ALWAYS_INLINE void loop_bnd(const cGH *cctkGH,
                                                  const F &f) {
  loop<CI, CJ, CK, where_t::boundary>(cctkGH, f);
}

template <int CI, int CJ, int CK, typename F>
inline CCTK_ATTRIBUTE_ALWAYS_INLINE void
loop_ghosts_inclusive(const cGH *cctkGH, const F &f) {
  loop<CI, CJ, CK, where_t::ghosts_inclusive>(cctkGH, f);
}

template <int CI, int CJ, int CK, typename F>
inline CCTK_ATTRIBUTE_ALWAYS_INLINE void loop_ghosts(const cGH *cctkGH,
                                                     const F &f) {
  loop<CI, CJ, CK, where_t::ghosts>(cctkGH, f);
}

////////////////////////////////////////////////////////////////////////////////

template <typename T, int CI, int CJ, int CK> struct GF3D {
  static_assert(CI == 0 || CI == 1, "");
  static_assert(CJ == 0 || CJ == 1, "");
  static_assert(CK == 0 || CK == 1, "");
  typedef T value_type;
  static constexpr int di = 1;
  const int dj, dk, np;
  const int ni, nj, nk;
  T *restrict const ptr;
  static constexpr CCTK_ATTRIBUTE_ALWAYS_INLINE std::array<int, dim>
  indextype() {
    return {CI, CJ, CK};
  }
  GF3D() = delete;
  GF3D(const GF3D &) = default;
  GF3D(GF3D &&) = default;
  GF3D &operator=(const GF3D &) = default;
  GF3D &operator=(GF3D &&) = default;
  GF3D(const cGH *restrict cctkGH, T *restrict ptr)
      : dj(di * (cctkGH->cctk_ash[0] - CI)),
        dk(dj * (cctkGH->cctk_ash[1] - CJ)),
        np(dk * (cctkGH->cctk_ash[2] - CK)), ni(cctkGH->cctk_lsh[0] - CI),
        nj(cctkGH->cctk_lsh[1] - CJ), nk(cctkGH->cctk_lsh[2] - CK), ptr(ptr) {}
  inline CCTK_ATTRIBUTE_ALWAYS_INLINE int offset(int i, int j, int k) const {
    // These index checks prevent vectorization. We thus only enable
    // them in debug mode.
#ifdef CCTK_DEBUG
    assert(i >= 0 && i < ni);
    assert(j >= 0 && j < nj);
    assert(k >= 0 && k < nk);
#endif
    return i * di + j * dj + k * dk;
  }
  inline int offset(const vect<int, dim> &I) const {
    return offset(I[0], I[1], I[2]);
  }
  inline T &restrict operator()(int i, int j, int k) const {
    return ptr[offset(i, j, k)];
  }
  inline T &restrict operator()(const vect<int, dim> &I) const {
    return ptr[offset(I)];
  }
};

////////////////////////////////////////////////////////////////////////////////

template <typename T> struct GF3D1 {
  typedef T value_type;
  T *restrict ptr;
#ifdef CCTK_DEBUG
  std::array<int, dim> imin, imax;
  std::array<int, dim> ash;
#endif
  static constexpr int di = 1;
  int dj, dk, np;
  int off;
  GF3D1() = delete;
  GF3D1(const GF3D1 &) = default;
  GF3D1(GF3D1 &&) = default;
  GF3D1 &operator=(const GF3D1 &) = default;
  GF3D1 &operator=(GF3D1 &&) = default;
  GF3D1(T *restrict ptr, const std::array<int, dim> &imin,
        const std::array<int, dim> &imax, const std::array<int, dim> &ash)
      : ptr(ptr),
#ifdef CCTK_DEBUG
        imin(imin), imax(imax), ash(ash),
#endif
        dj(di * ash[0]), dk(dj * ash[1]), np(dk * ash[2]),
        off(imin[0] * di + imin[1] * dj + imin[2] * dk) {
  }
  GF3D1(const cGH *restrict cctkGH, const std::array<int, dim> &indextype,
        const std::array<int, dim> &nghostzones, T *restrict ptr) {
    for (int d = 0; d < dim; ++d)
      assert(indextype[d] == 0 || indextype[d] == 1);
    for (int d = 0; d < dim; ++d) {
      assert(nghostzones[d] >= 0);
      assert(nghostzones[d] <= cctkGH->cctk_nghostzones[d]);
    }
    std::array<int, dim> imin, imax;
    for (int d = 0; d < dim; ++d) {
      imin[d] = cctkGH->cctk_nghostzones[d] - nghostzones[d];
      imax[d] = cctkGH->cctk_lsh[d] - indextype[d] -
                (cctkGH->cctk_nghostzones[d] - nghostzones[d]);
    }
    std::array<int, dim> ash;
    for (int d = 0; d < dim; ++d)
      ash[d] = cctkGH->cctk_ash[d] - indextype[d] -
               2 * (cctkGH->cctk_nghostzones[d] - nghostzones[d]);
    *this = GF3D1(ptr, imin, imax, ash);
  }
  int offset(int i, int j, int k) const {
    // These index checks prevent vectorization. We thus only enable
    // them in debug mode.
#ifdef CCTK_DEBUG
    assert(i >= imin[0] && i < imax[0]);
    assert(j >= imin[1] && j < imax[1]);
    assert(k >= imin[2] && k < imax[2]);
#endif
    return i * di + j * dj + k * dk - off;
  }
  int offset(const vect<int, dim> &I) const { return offset(I[0], I[1], I[2]); }
  T &restrict operator()(int i, int j, int k) const {
    return ptr[offset(i, j, k)];
  }
  T &restrict operator()(const vect<int, dim> &I) const {
    return ptr[offset(I)];
  }
  T &restrict operator()(const PointDesc &p) const { return (*this)(p.I); }
};

////////////////////////////////////////////////////////////////////////////////

struct GF3D2layout {
#ifdef CCTK_DEBUG
  vect<int, dim> imin, imax;
  vect<int, dim> ash;
#endif
  static constexpr int di = 1;
  int dj, dk, np;
  int off;
  GF3D2layout() = delete;
  GF3D2layout(const GF3D2layout &) = default;
  GF3D2layout(GF3D2layout &&) = default;
  GF3D2layout &operator=(const GF3D2layout &) = default;
  GF3D2layout &operator=(GF3D2layout &&) = default;
  GF3D2layout(const vect<int, dim> &imin, const vect<int, dim> &imax,
              const vect<int, dim> &ash)
      :
#ifdef CCTK_DEBUG
        imin(imin), imax(imax), ash(ash),
#endif
        dj(di * ash[0]), dk(dj * ash[1]), np(dk * ash[2]),
        off(imin[0] * di + imin[1] * dj + imin[2] * dk) {
  }
  GF3D2layout(const vect<int, dim> &imin, const vect<int, dim> &imax)
      : GF3D2layout(imin, imax, imax - imin) {
#ifdef CCTK_DEBUG
    assert(linear(imin[0], imin[1], imin[2]) == 0);
    assert(linear(imax[0] - 1, imax[1] - 1, imax[2] - 1) == np - 1);
#endif
  }
  GF3D2layout(const cGH *restrict cctkGH, const vect<int, dim> &indextype,
              const vect<int, dim> &nghostzones) {
    for (int d = 0; d < dim; ++d)
      assert(indextype[d] == 0 || indextype[d] == 1);
    for (int d = 0; d < dim; ++d) {
      assert(nghostzones[d] >= 0);
      assert(nghostzones[d] <= cctkGH->cctk_nghostzones[d]);
    }
    vect<int, dim> imin, imax;
    for (int d = 0; d < dim; ++d) {
      imin[d] = cctkGH->cctk_nghostzones[d] - nghostzones[d];
      imax[d] = cctkGH->cctk_lsh[d] - indextype[d] -
                (cctkGH->cctk_nghostzones[d] - nghostzones[d]);
    }
    vect<int, dim> ash;
    for (int d = 0; d < dim; ++d)
      ash[d] = cctkGH->cctk_ash[d] - indextype[d] -
               2 * (cctkGH->cctk_nghostzones[d] - nghostzones[d]);
    *this = GF3D2layout(imin, imax, ash);
  }
  GF3D2layout(const cGH *restrict cctkGH, const vect<int, dim> &indextype)
      : GF3D2layout(cctkGH, indextype,
                    {cctkGH->cctk_nghostzones[0], cctkGH->cctk_nghostzones[1],
                     cctkGH->cctk_nghostzones[2]}) {}
  bool operator==(const GF3D2layout &other) const {
    bool iseq = true;
#ifdef CCTK_DEBUG
    iseq &= std::equal_to<vect<int, dim> >()(imin, other.imin) &&
            std::equal_to<vect<int, dim> >()(imax, other.imax);
    iseq &= std::equal_to<vect<int, dim> >()(ash, other.ash);
#endif
    iseq &= dj == other.dj && dk == other.dk && np == other.np;
    iseq &= off == other.off;
    return iseq;
  }
  CCTK_DEVICE CCTK_HOST int linear(int i, int j, int k) const {
    // These index checks prevent vectorization. We thus only enable
    // them in debug mode.
#ifdef CCTK_DEBUG
    assert(i >= imin[0] && i < imax[0]);
    assert(j >= imin[1] && j < imax[1]);
    assert(k >= imin[2] && k < imax[2]);
#endif
    return i * di + j * dj + k * dk - off;
  }
  CCTK_DEVICE CCTK_HOST int linear(const vect<int, dim> &I) const {
    return linear(I[0], I[1], I[2]);
  }
  CCTK_DEVICE CCTK_HOST int delta(int i, int j, int k) const {
    return i * di + j * dj + k * dk;
  }
  CCTK_DEVICE CCTK_HOST int delta(const vect<int, dim> &I) const {
    return delta(I[0], I[1], I[2]);
  }
};

struct GF3D2index {
#ifdef CCTK_DEBUG
  GF3D2layout layout;
#endif
  int m_linear;
  GF3D2index() = delete;
  GF3D2index(const GF3D2index &) = default;
  GF3D2index(GF3D2index &&) = default;
  GF3D2index &operator=(const GF3D2index &) = default;
  GF3D2index &operator=(GF3D2index &&) = default;
  CCTK_DEVICE CCTK_HOST GF3D2index(const GF3D2layout &layout,
                                   const vect<int, dim> &I)
      :
#ifdef CCTK_DEBUG
        layout(layout),
#endif
        m_linear(layout.linear(I)) {
  }
  CCTK_DEVICE CCTK_HOST int linear() const { return m_linear; }
};

template <typename T> struct GF3D2 {
  typedef T value_type;
  // TODO: disallow inf, nan
  // Note: Keep `ptr` as first member, this improves performance a bit (?)
  T *restrict ptr;
  GF3D2layout layout;
  GF3D2() = delete;
  GF3D2(const GF3D2 &) = default;
  GF3D2(GF3D2 &&) = default;
  GF3D2 &operator=(const GF3D2 &) = default;
  GF3D2 &operator=(GF3D2 &&) = default;
  CCTK_DEVICE CCTK_HOST GF3D2(const GF3D2layout &layout, T *restrict ptr)
      : ptr(ptr), layout(layout) {}
  CCTK_DEVICE CCTK_HOST GF3D2index index(const vect<int, dim> &I) const {
    return GF3D2index(layout, I);
  }
  CCTK_DEVICE CCTK_HOST int linear(const vect<int, dim> &I) const {
    return index(I).linear();
  }
  CCTK_DEVICE CCTK_HOST int linear(int i, int j, int k) const {
    return index(vect<int, dim>{i, j, k}).linear();
  }
  CCTK_DEVICE CCTK_HOST int delta(const vect<int, dim> &I) const {
    return layout.delta(I);
  }
  CCTK_DEVICE CCTK_HOST int delta(int i, int j, int k) const {
    return layout.delta(i, j, k);
  }
  CCTK_DEVICE CCTK_HOST T &restrict operator()(const GF3D2index &index) const {
#ifdef CCTK_DEBUG
    assert(index.layout == this->layout);
#endif
    return ptr[index.linear()];
  }
  CCTK_DEVICE CCTK_HOST T &restrict operator()(const vect<int, dim> &I) const {
    return ptr[linear(I)];
  }
  CCTK_DEVICE CCTK_HOST T &restrict operator()(int i, int j, int k) const {
    return ptr[linear(i, j, k)];
  }
  CCTK_DEVICE CCTK_HOST void store(const GF3D2index &index,
                                   const T &value) const {
    ptr[index.linear()] = value;
  }
  CCTK_DEVICE CCTK_HOST void store(const vect<int, dim> &I,
                                   const T &value) const {
    ptr[linear(I)] = value;
  }
  CCTK_DEVICE CCTK_HOST Arith::simd<std::remove_cv_t<T> >
  operator()(const Arith::simdl<std::remove_cv_t<T> > &mask,
             const GF3D2index &index) const {
    return Arith::maskz_loadu(mask, &(*this)(index));
  }
  CCTK_DEVICE CCTK_HOST Arith::simd<std::remove_cv_t<T> >
  operator()(const Arith::simdl<std::remove_cv_t<T> > &mask,
             const GF3D2index &index,
             const Arith::simd<std::remove_cv_t<T> > &other) const {
    return Arith::masko_loadu(mask, &(*this)(index), other);
  }
  template <typename U,
            std::enable_if_t<std::is_convertible_v<T, U> > * = nullptr>
  CCTK_DEVICE CCTK_HOST Arith::simd<std::remove_cv_t<T> >
  operator()(const Arith::simdl<std::remove_cv_t<T> > &mask,
             const GF3D2index &index, const U &other) const {
    return Arith::masko_loadu(mask, &(*this)(index), other);
  }
  CCTK_DEVICE CCTK_HOST Arith::simd<std::remove_cv_t<T> >
  operator()(const Arith::simdl<std::remove_cv_t<T> > &mask,
             const vect<int, dim> &I) const {
    return Arith::maskz_loadu(mask, &(*this)(I));
  }
  CCTK_DEVICE CCTK_HOST Arith::simd<std::remove_cv_t<T> >
  operator()(const Arith::simdl<std::remove_cv_t<T> > &mask,
             const vect<int, dim> &I,
             const Arith::simd<std::remove_cv_t<T> > &other) const {
    return Arith::masko_loadu(mask, &(*this)(I), other);
  }
  template <typename U,
            std::enable_if_t<std::is_convertible_v<T, U> > * = nullptr>
  CCTK_DEVICE CCTK_HOST Arith::simd<std::remove_cv_t<T> >
  operator()(const Arith::simdl<std::remove_cv_t<T> > &mask,
             const vect<int, dim> &I, const U &other) const {
    return Arith::masko_loadu(mask, &(*this)(I), other);
  }
  CCTK_DEVICE CCTK_HOST void store(const Arith::simdl<T> &mask,
                                   const vect<int, dim> &I,
                                   const Arith::simd<T> &value) const {
    mask_storeu(mask, &ptr[linear(I)], value);
  }
  CCTK_DEVICE CCTK_HOST void store(const Arith::simdl<T> &mask,
                                   const GF3D2index &index,
                                   const Arith::simd<T> &value) const {
    mask_storeu(mask, &ptr[index.linear()], value);
  }
};

////////////////////////////////////////////////////////////////////////////////

template <int NI, int NJ, int NK, int OFF = 0> struct GF3D3layout {
  static_assert(NI >= 0, "");
  static_assert(NJ >= 0, "");
  static_assert(NK >= 0, "");

  static constexpr int ni = NI;
  static constexpr int nj = NJ;
  static constexpr int nk = NK;

  static constexpr int off = OFF;

  static constexpr int di = 1;
  static constexpr int dj = NI * di;
  static constexpr int dk = NJ * dj;
  static constexpr int np = NK * dk;

  constexpr int linear(int i, int j, int k) const {
    return i * di + j * dj + k * dk - OFF;
  }
  constexpr int linear(const vect<int, dim> &I) const {
    return (*this)(I[0], I[1], I[2]);
  }
};

template <int imin0, int imin1, int imin2, int imax0, int imax1, int imax2>
struct makeGF3D3layout {
private:
  static constexpr int ni = imax0 - imin0;
  static constexpr int nj = imax1 - imin1;
  static constexpr int nk = imax2 - imin2;
  static constexpr int di = GF3D3layout<ni, nj, nk>::di;
  static constexpr int dj = GF3D3layout<ni, nj, nk>::dj;
  static constexpr int dk = GF3D3layout<ni, nj, nk>::dk;
  static constexpr int off = imin0 * di + imin1 * dj + imin2 * dk;

public:
  typedef GF3D3layout<ni, nj, nk, off> type;
};
template <int imin0, int imin1, int imin2, int imax0, int imax1, int imax2>
using makeGF3D3layout_t =
    typename makeGF3D3layout<imin0, imin1, imin2, imax0, imax1, imax2>::type;

template <typename T, int NI, int NJ, int NK, int OFF = 0>
struct GF3D3 : GF3D3layout<NI, NJ, NK, OFF> {
  using GF3D3layout<NI, NJ, NK, OFF>::np;
  using GF3D3layout<NI, NJ, NK, OFF>::linear;

  std::array<T, np> arr;

  constexpr T &restrict operator()(int i, int j, int k) {
    return arr[linear(i, j, k)];
  }
  constexpr const T &restrict operator()(int i, int j, int k) const {
    return arr[linear(i, j, k)];
  }
  constexpr T &restrict operator()(const vect<int, dim> &I) {
    return (*this)(I[0], I[1], I[2]);
  }
  constexpr const T &restrict operator()(const vect<int, dim> &I) const {
    return (*this)(I[0], I[1], I[2]);
  }
};

template <typename T, int NI, int NJ, int NK, int OFF = 0>
struct GF3D3ptr : GF3D3layout<NI, NJ, NK, OFF> {
  using GF3D3layout<NI, NJ, NK, OFF>::np;
  using GF3D3layout<NI, NJ, NK, OFF>::linear;

  T *restrict ptr;

  GF3D3ptr() = delete;
  GF3D3ptr(T *restrict ptr) : ptr(ptr) {}

  constexpr T &restrict operator()(int i, int j, int k) const {
    return ptr[linear(i, j, k)];
  }
  constexpr T &restrict operator()(const vect<int, dim> &I) const {
    return (*this)(I[0], I[1], I[2]);
  }
};

////////////////////////////////////////////////////////////////////////////////

struct GF3D5layout {
#ifdef CCTK_DEBUG
  vect<int, dim> imin, imax;
  vect<int, dim> ash;
#endif
  static constexpr int di = 1;
  int dj, dk, np;
  int off;
  GF3D5layout() = delete;
  GF3D5layout(const GF3D5layout &) = default;
  GF3D5layout(GF3D5layout &&) = default;
  GF3D5layout &operator=(const GF3D5layout &) = default;
  GF3D5layout &operator=(GF3D5layout &&) = default;
  CCTK_DEVICE CCTK_HOST GF3D5layout(const vect<int, dim> &imin,
                                    const vect<int, dim> &imax,
                                    const vect<int, dim> &ash)
      :
#ifdef CCTK_DEBUG
        imin(imin), imax(imax), ash(ash),
#endif
        dj(di * ash[0]), dk(dj * ash[1]), np(dk * ash[2]),
        off(imin[0] * di + imin[1] * dj + imin[2] * dk) {
  }
  CCTK_DEVICE CCTK_HOST GF3D5layout(const vect<int, dim> &imin,
                                    const vect<int, dim> &imax)
      : GF3D5layout(imin, imax, imax - imin) {
#ifdef CCTK_DEBUG
    assert(linear(imin[0], imin[1], imin[2]) == 0);
    assert(linear(imax[0] - 1, imax[1] - 1, imax[2] - 1) == np - 1);
#endif
  }
  CCTK_DEVICE CCTK_HOST GF3D5layout(const cGH *restrict cctkGH,
                                    const vect<int, dim> &indextype,
                                    const vect<int, dim> &nghostzones) {
    for (int d = 0; d < dim; ++d)
      assert(indextype[d] == 0 || indextype[d] == 1);
    for (int d = 0; d < dim; ++d) {
      assert(nghostzones[d] >= 0);
      assert(nghostzones[d] <= cctkGH->cctk_nghostzones[d]);
    }
    vect<int, dim> imin, imax;
    for (int d = 0; d < dim; ++d) {
      imin[d] = cctkGH->cctk_nghostzones[d] - nghostzones[d];
      imax[d] = cctkGH->cctk_lsh[d] - indextype[d] -
                (cctkGH->cctk_nghostzones[d] - nghostzones[d]);
    }
    vect<int, dim> ash;
    for (int d = 0; d < dim; ++d)
      ash[d] = cctkGH->cctk_ash[d] - indextype[d] -
               2 * (cctkGH->cctk_nghostzones[d] - nghostzones[d]);
    *this = GF3D5layout(imin, imax, ash);
  }
  CCTK_DEVICE CCTK_HOST GF3D5layout(const cGH *restrict cctkGH,
                                    const vect<int, dim> &indextype)
      : GF3D5layout(cctkGH, indextype,
                    {cctkGH->cctk_nghostzones[0], cctkGH->cctk_nghostzones[1],
                     cctkGH->cctk_nghostzones[2]}) {}
  CCTK_DEVICE CCTK_HOST bool operator==(const GF3D5layout &other) const {
    bool iseq = true;
#ifdef CCTK_DEBUG
    iseq &= std::equal_to<vect<int, dim> >()(imin, other.imin) &&
            std::equal_to<vect<int, dim> >()(imax, other.imax);
    iseq &= std::equal_to<vect<int, dim> >()(ash, other.ash);
#endif
    iseq &= dj == other.dj && dk == other.dk && np == other.np;
    iseq &= off == other.off;
    return iseq;
  }
  CCTK_DEVICE CCTK_HOST int linear(int i, int j, int k) const {
    // These index checks prevent vectorization. We thus only enable
    // them in debug mode.
#ifdef CCTK_DEBUG
    assert(i >= imin[0] && i < imax[0]);
    assert(j >= imin[1] && j < imax[1]);
    assert(k >= imin[2] && k < imax[2]);
#endif
    return i * di + j * dj + k * dk - off;
  }
  CCTK_DEVICE CCTK_HOST int linear(const vect<int, dim> &I) const {
    return linear(I[0], I[1], I[2]);
  }
  CCTK_DEVICE CCTK_HOST int delta(int i, int j, int k) const {
    return i * di + j * dj + k * dk;
  }
  CCTK_DEVICE CCTK_HOST int delta(const vect<int, dim> &I) const {
    return delta(I[0], I[1], I[2]);
  }
};

struct GF3D5index {
#ifdef CCTK_DEBUG
  GF3D5layout layout;
#endif
  int m_linear;
  GF3D5index() = delete;
  GF3D5index(const GF3D5index &) = default;
  GF3D5index(GF3D5index &&) = default;
  GF3D5index &operator=(const GF3D5index &) = default;
  GF3D5index &operator=(GF3D5index &&) = default;
  CCTK_DEVICE CCTK_HOST GF3D5index(const GF3D5layout &layout,
                                   const vect<int, dim> &I)
      :
#ifdef CCTK_DEBUG
        layout(layout),
#endif
        m_linear(layout.linear(I)) {
  }
  CCTK_DEVICE CCTK_HOST int linear() const { return m_linear; }
};

template <typename T> struct GF3D5 {
  typedef T value_type;
#ifdef CCTK_DEBUG
  GF3D5layout layout;
#endif
  T *restrict ptr;
  GF3D5() = delete;
  GF3D5(const GF3D5 &) = default;
  GF3D5(GF3D5 &&) = default;
  GF3D5 &operator=(const GF3D5 &) = default;
  GF3D5 &operator=(GF3D5 &&) = default;
  CCTK_DEVICE CCTK_HOST GF3D5(const GF3D5layout &layout, T *restrict ptr)
      :
#ifdef CCTK_DEBUG
        layout(layout),
#endif
        ptr(ptr) {
#ifdef CCTK_DEBUG
    assert(&(*this)(layout, layout.imin) == ptr);
#endif
  }
  CCTK_DEVICE CCTK_HOST constexpr T &restrict
  operator()(const GF3D5index &index) const {
#ifdef CCTK_DEBUG
    assert(index.layout == this->layout);
#endif
    return ptr[index.linear()];
  }
  CCTK_DEVICE CCTK_HOST constexpr T &restrict
  operator()(const GF3D5layout &layout, const vect<int, dim> &I) const {
    return (*this)(GF3D5index(layout, I));
  }
  CCTK_DEVICE CCTK_HOST constexpr T &restrict
  operator()(const GF3D5layout &layout, int i, int j, int k) const {
    return (*this)(GF3D5index(layout, vect<int, dim>{i, j, k}));
  }
  CCTK_DEVICE CCTK_HOST void store(const GF3D5index &index,
                                   const T &value) const {
    operator()(index) = value;
  }
  CCTK_DEVICE CCTK_HOST void store(const GF3D5layout &layout,
                                   const vect<int, dim> &I,
                                   const T &value) const {
    operator()(layout, I) = value;
  }
  // CCTK_DEVICE CCTK_HOST Arith::simd<std::remove_cv_t<T> >
  // operator()(const Arith::simdl<std::remove_cv_t<T> > &mask,
  //            const GF3D5index &index) const {
  //   return Arith::maskz_loadu(mask, &(*this)(index));
  // }
  // CCTK_DEVICE CCTK_HOST Arith::simd<std::remove_cv_t<T> >
  // operator()(const Arith::simdl<std::remove_cv_t<T> > &mask,
  //            const GF3D5layout &layout, const vect<int, dim> &I) const {
  //   return (*this)(mask, GF3D5index(layout, I));
  // }
  struct simd_reference {
    using element_type = std::remove_cv_t<T>;
    using value_type = Arith::simd<element_type>;
    T *ptr;
    Arith::simdl<element_type> mask;
    template <typename U>
    CCTK_DEVICE CCTK_HOST simd_reference(T *const ptr,
                                         const Arith::simdl<U> mask)
        : ptr(ptr), mask(mask) {}
    simd_reference() = delete;
    simd_reference(const simd_reference &) = default;
    simd_reference(simd_reference &&) = default;
    CCTK_DEVICE CCTK_HOST operator value_type() const {
      return Arith::maskz_loadu(mask, ptr);
    }
    template <typename U>
    CCTK_DEVICE CCTK_HOST simd_reference operator=(const Arith::simd<U> value) {
      mask_storeu(mask, ptr, value);
      return *this;
    }
    template <typename U>
    CCTK_DEVICE CCTK_HOST simd_reference operator=(const U value) {
      mask_storeu(mask, ptr, value_type(value));
      return *this;
    }
  };
  CCTK_DEVICE CCTK_HOST simd_reference
  operator()(const Arith::simdl<std::remove_cv_t<T> > &mask,
             const GF3D5index &index) const {
    return simd_reference(&(*this)(index), mask);
  }
  CCTK_DEVICE CCTK_HOST simd_reference
  operator()(const Arith::simdl<std::remove_cv_t<T> > &mask,
             const GF3D5layout &layout, const vect<int, dim> &I) const {
    return (*this)(mask, GF3D5index(layout, I));
  }
  CCTK_DEVICE CCTK_HOST Arith::simd<std::remove_cv_t<T> >
  operator()(const Arith::simdl<std::remove_cv_t<T> > &mask,
             const GF3D5index &index,
             const Arith::simd<std::remove_cv_t<T> > &other) const {
    return Arith::masko_loadu(mask, &(*this)(index), other);
  }
  template <typename U,
            std::enable_if_t<std::is_convertible_v<T, U> > * = nullptr>
  CCTK_DEVICE CCTK_HOST Arith::simd<std::remove_cv_t<T> >
  operator()(const Arith::simdl<std::remove_cv_t<T> > &mask,
             const GF3D5index &index, const U &other) const {
    return Arith::masko_loadu(mask, &(*this)(index), other);
  }
  CCTK_DEVICE CCTK_HOST Arith::simd<std::remove_cv_t<T> >
  operator()(const Arith::simdl<std::remove_cv_t<T> > &mask,
             const GF3D5layout &layout, const vect<int, dim> &I,
             const Arith::simd<std::remove_cv_t<T> > &other) const {
    return (*this)(mask, GF3D5index(layout, I), other);
  }
  template <typename U,
            std::enable_if_t<std::is_convertible_v<T, U> > * = nullptr>
  CCTK_DEVICE CCTK_HOST Arith::simd<std::remove_cv_t<T> >
  operator()(const Arith::simdl<std::remove_cv_t<T> > &mask,
             const GF3D5layout &layout, const vect<int, dim> &I,
             const U &other) const {
    return (*this)(mask, GF3D5index(layout, I), other);
  }
  CCTK_DEVICE CCTK_HOST void store(const Arith::simdl<T> &mask,
                                   const GF3D5index &index,
                                   const Arith::simd<T> &value) const {
    mask_storeu(mask, &(*this)(index), value);
  }
  CCTK_DEVICE CCTK_HOST void store(const Arith::simdl<T> &mask,
                                   const GF3D5layout &layout,
                                   const vect<int, dim> &I,
                                   const Arith::simd<T> &value) const {
    store(mask, GF3D5index(layout, I), value);
  }
};

template <typename T> struct is_GF3D5 : std::false_type {};
template <typename T> struct is_GF3D5<GF3D5<T> > : std::true_type {};
template <typename T> inline constexpr bool is_GF3D5_v = is_GF3D5<T>::value;

template <typename T> struct GF3D5vector {
  static_assert((std::is_same_v<T, amrex::Real>), "");
  typedef T value_type;
  GF3D5layout layout;

  amrex::FArrayBox fab;

private:
  static amrex::FArrayBox make_fab(const GF3D5layout &layout, const int nvars) {
    const amrex::Box box(amrex::IntVect(0, 0, 0),
                         amrex::IntVect(layout.np - 1, 0, 0));
    amrex::FArrayBox fab;
#pragma omp critical(GF3D5vector_GF3D5vector)
    fab = amrex::FArrayBox(box, nvars, amrex::The_Async_Arena());
    return fab;
  }

public:
  GF3D5vector(const GF3D5layout &layout, const int nvars)
      : layout(layout), fab(make_fab(layout, nvars)) {}
  size_t size() const { return fab.nComp(); }
  GF3D5<T> operator()(const int n) const {
    assert(n >= 0 && n < int(size()));
    return GF3D5<T>(layout, (T *)fab.dataPtr(n) + layout.off);
  }
};

} // namespace Loop

// Macros for declaring variables using DECLARE_CCTK_ARGUMENTSX_func_name
#define CCTK_CENTERING_GRID const Loop::GridDescBase cctk_grid(cctkGH)
#define grid cctk_grid

#ifdef CARPETX_GF3D5

#define CCTK_CENTERING_LAYOUT(L, V)                                            \
  constexpr std::array<int, Loop::dim> L##_centered V;                         \
  const Loop::GF3D5layout L##gf_layout(cctkGH, L##_centered)
#define CCTK_CENTERING_GF(C, L, N)                                             \
  const Loop::GF3D5<C CCTK_REAL> N(L##gf_layout, cctk_ptr_##N)

#else

#define CCTK_CENTERING_LAYOUT(L, V)                                            \
  constexpr std::array<int, Loop::dim> L##_centered V;                         \
  const Loop::GF3D2layout L##gf_layout(cctkGH, L##_centered)
#define CCTK_CENTERING_GF(C, L, N)                                             \
  const Loop::GF3D2<C CCTK_REAL> N(L##gf_layout, cctk_ptr_##N)

#endif

#endif // #ifndef LOOP_HXX
