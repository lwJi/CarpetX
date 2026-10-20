#ifndef CARPETX_LOOP_LOOP_DEVICE_HXX
#define CARPETX_LOOP_LOOP_DEVICE_HXX

#include "loop.hxx"

#include <cctk.h>

#include <AMReX.H>
#include <AMReX_Box.H>
#include <AMReX_IndexType.H>
#include <AMReX_IntVect.H>
#ifdef AMREX_USE_GPU
#include <AMReX_Gpu.H>
#endif

#include <cassert>

#ifndef HAVE_CAPABILITY_Loop
#error                                                                         \
    "Using #include <loop_device.hxx> requires the capability 'Loop' in the calling thorn. Add this to the thorn's 'configuration.ccl' file."
#endif

namespace Loop {

struct GridDescBaseDevice : GridDescBase {

protected:
  GridDescBaseDevice() : GridDescBase() {}

public:
  GridDescBaseDevice(const GridDescBaseDevice &) = default;
  GridDescBaseDevice(GridDescBaseDevice &&) = default;
  GridDescBaseDevice &operator=(const GridDescBaseDevice &) = default;
  GridDescBaseDevice &operator=(GridDescBaseDevice &&) = default;

  GridDescBaseDevice(const cGH *cctkGH) : GridDescBase(cctkGH) {}

  template <int CI, int CJ, int CK, int VS = 1, int N = 1,
            int NT = AMREX_GPU_MAX_THREADS, typename F>
  void loop_box_device(const vect<int, dim> &restrict bnd_min,
                       const vect<int, dim> &restrict bnd_max,
                       const vect<int, dim> &restrict loop_min,
                       const vect<int, dim> &restrict loop_max,
                       const F &f) const {
#ifndef AMREX_USE_GPU

    return this->template loop_box<CI, CJ, CK, VS, N>(bnd_min, bnd_max,
                                                      loop_min, loop_max, f);

#else

    static_assert(CI == 0 || CI == 1);
    static_assert(CJ == 0 || CJ == 1);
    static_assert(CK == 0 || CK == 1);
    static_assert(VS > 0);
    static_assert(N >= 0);
    static_assert(NT > 0);

    static_assert(VS == 1, "Only vector size 1 is supported on GPUs");

    if (N == 0 || any(loop_max <= loop_min))
      return;

    const auto bnd_min1 = bnd_min;
    const auto bnd_max1 = bnd_max;
    const auto loop_min1 = loop_min;
    const auto loop_max1 = loop_max;

    const amrex::Box box(
        amrex::IntVect(loop_min[0], loop_min[1], loop_min[2]),
        amrex::IntVect(loop_max[0] - 1, loop_max[1] - 1, loop_max[2] - 1),
        amrex::IntVect(CI ? amrex::IndexType::CELL : amrex::IndexType::NODE,
                       CJ ? amrex::IndexType::CELL : amrex::IndexType::NODE,
                       CK ? amrex::IndexType::CELL : amrex::IndexType::NODE));

    amrex::ParallelFor<NT>(
        box, [=, *this] CCTK_DEVICE(const int i, const int j,
                                    const int k) CCTK_ATTRIBUTE_ALWAYS_INLINE {
          const vect<int, dim> I = {i, j, k};
          const vect<int, dim> NI =
              vect<int, dim>(I > bnd_max1 - 1) - vect<int, dim>(I < bnd_min1);
          const vect<int, dim> I0 =
              if_else(NI == 0, 0, if_else(NI < 0, bnd_min1, bnd_max1 - 1));
          const vect<int, dim> BI =
              vect<int, dim>(I == bnd_max1 - 1) - vect<int, dim>(I == bnd_min1);
          for (int iter = 0; iter < N; ++iter) {
            const PointDesc p =
                point_desc({CI, CJ, CK}, I, iter, NI, I0, BI, bnd_min1,
                           bnd_max1, loop_min1, loop_max1);
            f(p);
          }
        });

    static const bool gpu_sync_after_every_kernel = []() {
      int type;
      const void *const ptr =
          CCTK_ParameterGet("gpu_sync_after_every_kernel", "CarpetX", &type);
      assert(ptr);
      assert(type == PARAMETER_BOOLEAN);
      return *static_cast<const CCTK_INT *>(ptr);
    }();
    if (gpu_sync_after_every_kernel) {
      amrex::Gpu::synchronize();
      AMREX_GPU_ERROR_CHECK();
    }
#endif
  }

  template <int CI, int CJ, int CK, int VS = 1, int N = 1,
            int NT = AMREX_GPU_MAX_THREADS, typename F>
  inline CCTK_KERNEL void
  loop_all_device(const vect<int, dim> &group_nghostzones, const F &f) const {
    vect<int, dim> bnd_min, bnd_max;
    boundary_box<CI, CJ, CK>(group_nghostzones, bnd_min, bnd_max);
    vect<int, dim> imin, imax;
    box_all<CI, CJ, CK>(group_nghostzones, imin, imax);
    loop_box_device<CI, CJ, CK, VS, N, NT>(bnd_min, bnd_max, imin, imax, f);
  }

  template <int CI, int CJ, int CK, int VS = 1, int N = 1,
            int NT = AMREX_GPU_MAX_THREADS, typename F>
  inline CCTK_KERNEL void
  loop_int_device(const vect<int, dim> &group_nghostzones, const F &f) const {
    vect<int, dim> bnd_min, bnd_max;
    boundary_box<CI, CJ, CK>(group_nghostzones, bnd_min, bnd_max);
    vect<int, dim> imin, imax;
    box_int<CI, CJ, CK>(group_nghostzones, imin, imax);
    loop_box_device<CI, CJ, CK, VS, N, NT>(bnd_min, bnd_max, imin, imax, f);
  }

  template <int CI, int CJ, int CK, int VS = 1, int N = 1,
            int NT = AMREX_GPU_MAX_THREADS, typename F>
  inline CCTK_KERNEL void
  loop_there_device(const vect<int, dim> &group_nghostzones,
                    const vect<vect<vect<bool, dim>, dim>, dim> &there,
                    const F &f) const {
    constexpr vect<int, dim> offset{CI, CJ, CK};

    vect<int, dim> bnd_min, bnd_max;
    boundary_box<CI, CJ, CK>(group_nghostzones, bnd_min, bnd_max);
    vect<int, dim> all_min, all_max, int_min, int_max;
    domain_boxes<CI, CJ, CK>(group_nghostzones, all_min, all_max, int_min,
                             int_max);

    for (int rank = dim; rank >= 0; --rank) {

      for (int nk = -1; nk <= +1; ++nk) {
        for (int nj = -1; nj <= +1; ++nj) {
          for (int ni = -1; ni <= +1; ++ni) {
            if ((ni == 0) + (nj == 0) + (nk == 0) == rank) {

              if (there[ni + 1][nj + 1][nk + 1]) {

                const vect<int, dim> inormal{ni, nj, nk};

                vect<int, dim> imin, imax;
                for (int d = 0; d < dim; ++d) {
                  switch (inormal[d]) {
                  case -1:
                    imin[d] = all_min[d];
                    imax[d] = int_min[d];
                    break;
                  case 0:
                    imin[d] = int_min[d];
                    imax[d] = int_max[d];
                    break;
                  case +1:
                    imin[d] = int_max[d];
                    imax[d] = all_max[d];
                    break;
                  default:
                    assert(0);
                  }

                  using std::max, std::min;
                  imin[d] = max(tmin[d], imin[d]);
                  imax[d] = min(tmax[d], imax[d]);
                }

                loop_box_device<CI, CJ, CK, VS, N, NT>(bnd_min, bnd_max, imin,
                                                       imax, f);
              }
            }
          }
        }
      }
    }
  }

  template <int CI, int CJ, int CK, int VS = 1, int N = 1,
            int NT = AMREX_GPU_MAX_THREADS, typename F>
  inline CCTK_KERNEL void
  loop_bnd_device(const vect<int, dim> &group_nghostzones, const F &f) const {
    vect<int, dim> bnd_min, bnd_max;
    boundary_box<CI, CJ, CK>(group_nghostzones, bnd_min, bnd_max);
    vect<int, dim> all_min, all_max, int_min, int_max;
    domain_boxes<CI, CJ, CK>(group_nghostzones, all_min, all_max, int_min,
                             int_max);

    for (int rank = dim - 1; rank >= 0; --rank) {

      for (int nk = -1; nk <= +1; ++nk) {
        for (int nj = -1; nj <= +1; ++nj) {
          for (int ni = -1; ni <= +1; ++ni) {
            if ((ni == 0) + (nj == 0) + (nk == 0) == rank) {

              if ((ni != 0 && bbox[ni < 0 ? 0 : 1][0]) ||
                  (nj != 0 && bbox[nj < 0 ? 0 : 1][1]) ||
                  (nk != 0 && bbox[nk < 0 ? 0 : 1][2])) {

                const vect<int, dim> inormal{ni, nj, nk};

                vect<int, dim> imin, imax;
                for (int d = 0; d < dim; ++d) {
                  switch (inormal[d]) {
                  case -1:
                    imin[d] = all_min[d];
                    imax[d] = int_min[d];
                    break;
                  case 0:
                    imin[d] = int_min[d];
                    imax[d] = int_max[d];
                    break;
                  case +1:
                    imin[d] = int_max[d];
                    imax[d] = all_max[d];
                    break;
                  default:
                    assert(0);
                  }

                  using std::min, std::max;
                  imin[d] = max(tmin[d], imin[d]);
                  imax[d] = min(tmax[d], imax[d]);
                }

                loop_box_device<CI, CJ, CK, VS, N, NT>(bnd_min, bnd_max, imin,
                                                       imax, f);
              }
            }
          }
        }
      }
    }
  }

  template <int CI, int CJ, int CK, int VS = 1, int N = 1,
            int NT = AMREX_GPU_MAX_THREADS, typename F>
  inline CCTK_KERNEL void
  loop_outermost_int_device(const vect<int, dim> &group_nghostzones,
                            const vect<vect<bool, dim>, 2> &is_sym_bnd,
                            const F &f) const {
    vect<int, dim> bnd_min, bnd_max;
    boundary_box<CI, CJ, CK>(group_nghostzones, bnd_min, bnd_max);

    vect<int, dim> all_min, all_max, int_min, int_max;
    domain_boxes<CI, CJ, CK>(group_nghostzones, all_min, all_max, int_min,
                             int_max);

    for (int d = 0; d < dim; ++d) {
      assert(group_nghostzones[d] == nghostzones[d]);
    }

    all_min += nghostzones;
    int_min += nghostzones;
    bnd_min += nghostzones;
    all_max -= nghostzones;
    int_max -= nghostzones;
    bnd_max -= nghostzones;

    for (int rank = dim - 1; rank >= 0; --rank) {

      for (int nk = -1; nk <= +1; ++nk) {
        for (int nj = -1; nj <= +1; ++nj) {
          for (int ni = -1; ni <= +1; ++ni) {
            if ((ni == 0) + (nj == 0) + (nk == 0) == rank) {
              if ((ni != 0 && bbox[ni < 0 ? 0 : 1][0] &&
                   !is_sym_bnd[ni < 0 ? 0 : 1][0]) ||
                  (nj != 0 && bbox[nj < 0 ? 0 : 1][1] &&
                   !is_sym_bnd[nj < 0 ? 0 : 1][1]) ||
                  (nk != 0 && bbox[nk < 0 ? 0 : 1][2] &&
                   !is_sym_bnd[nk < 0 ? 0 : 1][2])) {

                const vect<int, dim> inormal{ni, nj, nk};

                vect<int, dim> imin, imax;
                for (int d = 0; d < dim; ++d) {
                  switch (inormal[d]) {
                  case -1:
                    imin[d] = all_min[d];
                    imax[d] = int_min[d];
                    break;
                  case 0:
                    imin[d] = int_min[d];
                    imax[d] = int_max[d];
                    break;
                  case +1:
                    imin[d] = int_max[d];
                    imax[d] = all_max[d];
                    break;
                  default:
                    assert(0);
                  }

                  using std::min, std::max;
                  imin[d] = max(tmin[d], imin[d]);
                  imax[d] = min(tmax[d], imax[d]);
                }

                loop_box_device<CI, CJ, CK, VS, N, NT>(bnd_min, bnd_max, imin,
                                                       imax, f);
              }
            }
          }
        }
      }
    }
  }

  template <int CI, int CJ, int CK, int VS = 1, int N = 1,
            int NT = AMREX_GPU_MAX_THREADS, typename F>
  inline CCTK_KERNEL void
  loop_ghosts_device(const vect<int, dim> &group_nghostzones,
                     const F &f) const {
    vect<int, dim> bnd_min, bnd_max;
    boundary_box<CI, CJ, CK>(group_nghostzones, bnd_min, bnd_max);
    vect<int, dim> all_min, all_max, int_min, int_max;
    domain_boxes<CI, CJ, CK>(group_nghostzones, all_min, all_max, int_min,
                             int_max);

    for (int rank = dim - 1; rank >= 0; --rank) {

      for (int nk = -1; nk <= +1; ++nk) {
        for (int nj = -1; nj <= +1; ++nj) {
          for (int ni = -1; ni <= +1; ++ni) {
            if ((ni == 0) + (nj == 0) + (nk == 0) == rank) {

              if ((ni == 0 || !bbox[ni < 0 ? 0 : 1][0]) &&
                  (nj == 0 || !bbox[nj < 0 ? 0 : 1][1]) &&
                  (nk == 0 || !bbox[nk < 0 ? 0 : 1][2])) {

                const vect<int, dim> inormal{ni, nj, nk};

                vect<int, dim> imin, imax;
                for (int d = 0; d < dim; ++d) {
                  switch (inormal[d]) {
                  case -1:
                    imin[d] = all_min[d];
                    imax[d] = int_min[d];
                    break;
                  case 0:
                    imin[d] = int_min[d];
                    imax[d] = int_max[d];
                    break;
                  case +1:
                    imin[d] = int_max[d];
                    imax[d] = all_max[d];
                    break;
                  default:
                    assert(0);
                  }

                  using std::min, std::max;
                  imin[d] = max(tmin[d], imin[d]);
                  imax[d] = min(tmax[d], imax[d]);
                }

                loop_box_device<CI, CJ, CK, VS, N, NT>(bnd_min, bnd_max, imin,
                                                       imax, f);
              }
            }
          }
        }
      }
    }
  }

  template <int CI, int CJ, int CK, where_t where, typename F>
  inline CCTK_KERNEL std::enable_if_t<(where == where_t::everywhere), void>
  loop_device(const vect<int, dim> &group_nghostzones, const F &f) const {
    loop_all_device<CI, CJ, CK>(group_nghostzones, f);
  }
  template <int CI, int CJ, int CK, where_t where, typename F>
  inline CCTK_KERNEL std::enable_if_t<(where == where_t::interior), void>
  loop_device(const vect<int, dim> &group_nghostzones, const F &f) const {
    loop_int_device<CI, CJ, CK>(group_nghostzones, f);
  }
  template <int CI, int CJ, int CK, where_t where, typename F>
  inline CCTK_KERNEL std::enable_if_t<(where == where_t::boundary), void>
  loop_device(const vect<int, dim> &group_nghostzones, const F &f) const {
    loop_bnd_device<CI, CJ, CK>(group_nghostzones, f);
  }
  template <int CI, int CJ, int CK, where_t where, typename F>
  inline CCTK_KERNEL std::enable_if_t<(where == where_t::ghosts), void>
  loop_device(const vect<int, dim> &group_nghostzones, const F &f) const {
    loop_ghosts_device<CI, CJ, CK>(group_nghostzones, f);
  }

  template <typename F>
  inline CCTK_KERNEL void
  loop_there_device_idx(const vect<int, dim> &indextype,
                        const vect<vect<vect<bool, dim>, dim>, dim> &there,
                        const vect<int, dim> &group_nghostzones,
                        const F &f) const {
    switch (indextype[0] + 2 * indextype[1] + 4 * indextype[2]) {
    case 0b000:
      return noinline([&] {
        return loop_there_device<0, 0, 0>(group_nghostzones, there, f);
      });
    case 0b001:
      return noinline([&] {
        return loop_there_device<1, 0, 0>(group_nghostzones, there, f);
      });
    case 0b010:
      return noinline([&] {
        return loop_there_device<0, 1, 0>(group_nghostzones, there, f);
      });
    case 0b011:
      return noinline([&] {
        return loop_there_device<1, 1, 0>(group_nghostzones, there, f);
      });
    case 0b100:
      return noinline([&] {
        return loop_there_device<0, 0, 1>(group_nghostzones, there, f);
      });
    case 0b101:
      return noinline([&] {
        return loop_there_device<1, 0, 1>(group_nghostzones, there, f);
      });
    case 0b110:
      return noinline([&] {
        return loop_there_device<0, 1, 1>(group_nghostzones, there, f);
      });
    case 0b111:
      return noinline([&] {
        return loop_there_device<1, 1, 1>(group_nghostzones, there, f);
      });
    default:
      assert(0);
    }
  }

  template <where_t where, typename F>
  inline CCTK_KERNEL void
  loop_device_idx(const vect<int, dim> &indextype,
                  const vect<int, dim> &group_nghostzones, const F &f) const {
    switch (indextype[0] + 2 * indextype[1] + 4 * indextype[2]) {
    case 0b000:
      return noinline(
          [&] { return loop_device<0, 0, 0, where>(group_nghostzones, f); });
    case 0b001:
      return noinline(
          [&] { return loop_device<1, 0, 0, where>(group_nghostzones, f); });
    case 0b010:
      return noinline(
          [&] { return loop_device<0, 1, 0, where>(group_nghostzones, f); });
    case 0b011:
      return noinline(
          [&] { return loop_device<1, 1, 0, where>(group_nghostzones, f); });
    case 0b100:
      return noinline(
          [&] { return loop_device<0, 0, 1, where>(group_nghostzones, f); });
    case 0b101:
      return noinline(
          [&] { return loop_device<1, 0, 1, where>(group_nghostzones, f); });
    case 0b110:
      return noinline(
          [&] { return loop_device<0, 1, 1, where>(group_nghostzones, f); });
    case 0b111:
      return noinline(
          [&] { return loop_device<1, 1, 1, where>(group_nghostzones, f); });
    default:
      assert(0);
    }
  }
};

template <where_t where, typename F>
inline CCTK_KERNEL void
loop_device_idx(const cGH *cctkGH, const vect<int, dim> &indextype,
                const vect<int, dim> &nghostzones, const F &f) {
  GridDescBaseDevice(cctkGH).loop_device_idx<where>(indextype, nghostzones, f);
}

} // namespace Loop

#undef CCTK_CENTERING_GRID
#define CCTK_CENTERING_GRID const Loop::GridDescBaseDevice cctk_grid(cctkGH)

#endif
