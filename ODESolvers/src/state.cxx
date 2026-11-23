#include "state.hxx"

#include "rk_tableau.hxx"

#include <carpetx.hxx>

#include <cctk.h>

#include <div.hxx>

#include <AMReX_MultiFab.H>
#include <AMReX_Version.H>

#if defined _OPENMP || defined __HIPCC__
#include <omp.h>
#else
static inline int omp_get_max_threads() { return 1; }
#endif

#include <algorithm>
#include <array>
#include <cassert>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <functional>
#include <span>
#include <utility>
#include <vector>

namespace ODESolvers {
namespace {
constexpr CarpetX::tl_t current_timelevel{0};

template <std::size_t N>
void lincomb(
    const std::vector<amrex::MultiFab *> &dstmfabs, const CCTK_REAL scale,
    const std::array<CCTK_REAL, N> &factors,
    const std::vector<std::array<const amrex::MultiFab *, N> > &srcmfabs) {
  const std::size_t size = dstmfabs.size();
  assert(srcmfabs.size() == size);
  for (std::size_t m = 0; m < size; ++m) {
    const auto ncomp = dstmfabs.at(m)->nComp();
    const auto ngrowvect = dstmfabs.at(m)->nGrowVect();
    for (std::size_t n = 0; n < N; ++n) {
      assert(srcmfabs.at(m)[n]->nComp() == ncomp);
      assert(srcmfabs.at(m)[n]->nGrowVect() == ngrowvect);
    }
  }

  using std::isfinite;
  assert(isfinite(scale));
  const bool read_dst = scale != 0;
  for (std::size_t n = 0; n < N; ++n)
    assert(isfinite(factors[n]));

#ifndef AMREX_USE_GPU
  std::vector<std::function<void()> > tasks;
#endif

  for (std::size_t m = 0; m < size; ++m) {
    amrex::MultiFab &dstmfab = *dstmfabs.at(m);
    const std::ptrdiff_t ncomps = dstmfab.nComp();
    const auto mfitinfo = amrex::MFItInfo().DisableDeviceSync();
    for (amrex::MFIter mfi(dstmfab, mfitinfo); mfi.isValid(); ++mfi) {
      const amrex::Array4<CCTK_REAL> dstvar = dstmfab.array(mfi);
      std::array<amrex::Array4<const CCTK_REAL>, N> srcvars;
      for (std::size_t n = 0; n < N; ++n)
        srcvars[n] = srcmfabs.at(m)[n]->const_array(mfi);
#if AMREX_RELEASE_NUMBER >= 260200
      for (std::size_t n = 0; n < N; ++n) {
        assert(srcvars[n].template get_stride<1>() == dstvar.get_stride<1>());
        assert(srcvars[n].template get_stride<2>() == dstvar.get_stride<2>());
        assert(srcvars[n].template get_stride<3>() == dstvar.get_stride<3>());
      }
      const std::ptrdiff_t nstride = dstvar.get_stride<3>();
#else
      for (std::size_t n = 0; n < N; ++n) {
        assert(srcvars[n].jstride == dstvar.jstride);
        assert(srcvars[n].kstride == dstvar.kstride);
        assert(srcvars[n].nstride == dstvar.nstride);
      }
      const std::ptrdiff_t nstride = dstvar.nstride;
#endif
      const std::ptrdiff_t npoints = nstride * ncomps;

      CCTK_REAL *restrict const dstptr = dstvar.dataPtr();
      std::array<const CCTK_REAL *restrict, N> srcptrs;
      for (std::size_t n = 0; n < N; ++n)
        srcptrs[n] = srcvars[n].dataPtr();

#ifndef AMREX_USE_GPU
      const std::ptrdiff_t ntiles = omp_get_max_threads();
      const std::ptrdiff_t tile_size = Arith::align_ceil(
          Arith::div_ceil(npoints, ntiles), std::ptrdiff_t(64));

      for (std::ptrdiff_t imin = 0; imin < npoints; imin += tile_size) {
        using std::min;
        const std::ptrdiff_t imax = min(npoints, imin + tile_size);

        if (!read_dst && N == 1 && factors[0] == 1) {
          auto task = [=]() {
            std::memcpy(&dstptr[imin], &srcptrs[0][imin],
                        (imax - imin) * sizeof *dstptr);
          };
          tasks.push_back(std::move(task));

        } else if (!read_dst && N >= 1 && factors[0] == 1) {
          auto task = [=]() {
#pragma omp simd
            for (std::ptrdiff_t i = imin; i < imax; ++i) {
              CCTK_REAL accum = srcptrs[0][i];
              for (std::size_t n = 1; n < N; ++n)
                accum += factors[n] * srcptrs[n][i];
              dstptr[i] = accum;
            }
          };
          tasks.push_back(std::move(task));

        } else if (!read_dst) {
          auto task = [=]() {
#pragma omp simd
            for (std::ptrdiff_t i = imin; i < imax; ++i) {
              CCTK_REAL accum = 0;
              for (std::size_t n = 0; n < N; ++n)
                accum += factors[n] * srcptrs[n][i];
              dstptr[i] = accum;
            }
          };
          tasks.push_back(std::move(task));

        } else if (scale == 1) {
          auto task = [=]() {
#pragma omp simd
            for (std::ptrdiff_t i = imin; i < imax; ++i) {
              CCTK_REAL accum = dstptr[i];
              for (std::size_t n = 0; n < N; ++n)
                accum += factors[n] * srcptrs[n][i];
              dstptr[i] = accum;
            }
          };
          tasks.push_back(std::move(task));

        } else {
          auto task = [=]() {
#pragma omp simd
            for (std::ptrdiff_t i = imin; i < imax; ++i) {
              CCTK_REAL accum = scale * dstptr[i];
              for (std::size_t n = 0; n < N; ++n)
                accum += factors[n] * srcptrs[n][i];
              dstptr[i] = accum;
            }
          };
          tasks.push_back(std::move(task));
        }
      }

#else
      const CCTK_REAL scale1 = scale;
      assert(npoints < INT_MAX);
      const amrex::Box box(
          amrex::IntVect(0, 0, 0), amrex::IntVect(npoints - 1, 0, 0),
          amrex::IntVect(amrex::IndexType::CELL, amrex::IndexType::CELL,
                         amrex::IndexType::CELL));

      if (!read_dst) {
        amrex::launch(
            box,
            [=] CCTK_DEVICE(const amrex::Box &box)
                __attribute__((__always_inline__, __flatten__)) {
                  const int i = box.smallEnd()[0];
                  CCTK_REAL accum = 0;
                  const CCTK_REAL *restrict const factors_ptr = factors.data();
                  const CCTK_REAL *restrict const *restrict const srcptrs_ptr =
                      srcptrs.data();
                  for (std::size_t n = 0; n < N; ++n)
                    accum += factors_ptr[n] * srcptrs_ptr[n][i];
                  dstptr[i] = accum;
                });

      } else {
        amrex::launch(
            box,
            [=] CCTK_DEVICE(const amrex::Box &box)
                __attribute__((__always_inline__, __flatten__)) {
                  const int i = box.smallEnd()[0];
                  CCTK_REAL accum = scale1 * dstptr[i];
                  const CCTK_REAL *restrict const factors_ptr = factors.data();
                  const CCTK_REAL *restrict const *restrict const srcptrs_ptr =
                      srcptrs.data();
                  for (std::size_t n = 0; n < N; ++n)
                    accum += factors_ptr[n] * srcptrs_ptr[n][i];
                  dstptr[i] = accum;
                });
      }

#endif
    }
  }

#ifndef AMREX_USE_GPU
#pragma omp parallel for schedule(dynamic)
  for (std::size_t i = 0; i < tasks.size(); ++i)
    tasks[i]();
#else
  amrex::Gpu::synchronize();
  AMREX_GPU_ERROR_CHECK();
#endif
}

template <std::size_t N>
void combine_exactly(const std::vector<amrex::MultiFab *> &dstmfabs,
                     const CCTK_REAL scale,
                     const std::vector<const term_t *> &terms) {
  assert(terms.size() == N);
  std::array<CCTK_REAL, N> factors;
  std::vector<std::array<const amrex::MultiFab *, N> > srcmfabs(
      dstmfabs.size());
  for (std::size_t n = 0; n < N; ++n) {
    factors[n] = terms[n]->factor;
    assert(terms[n]->mfabs.size() == dstmfabs.size());
    for (std::size_t m = 0; m < dstmfabs.size(); ++m)
      srcmfabs[m][n] = terms[n]->mfabs[m];
  }
  lincomb(dstmfabs, scale, factors, srcmfabs);
}

constexpr std::size_t max_terms = max_stages + 1;

template <std::size_t... Ns>
bool combine_up_to(std::index_sequence<Ns...>,
                   const std::vector<amrex::MultiFab *> &dstmfabs,
                   const CCTK_REAL scale,
                   const std::vector<const term_t *> &terms) {
  return ((terms.size() == Ns &&
           (combine_exactly<Ns>(dstmfabs, scale, terms), true)) ||
          ...);
}

void combine(const std::vector<amrex::MultiFab *> &dstmfabs,
             const CCTK_REAL scale, const std::span<const term_t> terms) {
  std::vector<const term_t *> nonzero;
  nonzero.reserve(terms.size());
  for (const term_t &term : terms)
    if (term.factor != 0)
      nonzero.push_back(&term);
  if (!combine_up_to(std::make_index_sequence<max_terms + 1>(), dstmfabs, scale,
                     nonzero))
    CCTK_VERROR("ODESolvers: a linear combination of %d terms is not "
                "supported; a stage of an RK step reads at most %d, the old "
                "state and %d stage right-hand sides",
                int(nonzero.size()), int(max_terms), max_stages);
}

} // namespace

mfabs_t grid_state_t::mfabs() const {
  mfabs_t mfabs;
  mfabs.reserve(groups.size());
  for (const CarpetX::group_t &group : groups)
    mfabs.push_back(&CarpetX::data(group, current_timelevel));
  return mfabs;
}

mfabs_t scratch_state_t::mfabs() const {
  mfabs_t mfabs;
  mfabs.reserve(scratch_.size());
  for (const CarpetX::scratch_t &scratch : scratch_)
    mfabs.push_back(&scratch.mfab());
  return mfabs;
}

std::vector<amrex::MultiFab *> scratch_state_t::destinations() {
  std::vector<amrex::MultiFab *> dstmfabs;
  dstmfabs.reserve(scratch_.size());
  for (CarpetX::scratch_t &scratch : scratch_)
    dstmfabs.push_back(&scratch.mfab());
  return dstmfabs;
}

scratch_state_t scratch_state_t::scaled(const grid_state_t &src,
                                        const CCTK_REAL factor) {
  scratch_state_t result;
  result.scratch_.reserve(src.groups.size());
  for (const CarpetX::group_t &group : src.groups)
    result.scratch_.push_back(CarpetX::scratch(group));
  const term_t terms[] = {{factor, src.mfabs()}};
  combine(result.destinations(), 0, terms);
  return result;
}

void scratch_state_t::add(const CCTK_REAL factor, const grid_state_t &src) {
  const term_t terms[] = {{factor, src.mfabs()}};
  combine(destinations(), 1, terms);
}

void update(const grid_state_t &dst, const CCTK_REAL scale,
            const std::span<const term_t> terms,
            const CarpetX::valid_t leaves_valid) {
  std::vector<CarpetX::write_t> writes;
  std::vector<amrex::MultiFab *> dstmfabs;
  writes.reserve(dst.groups.size());
  dstmfabs.reserve(dst.groups.size());
  for (const CarpetX::group_t &group : dst.groups) {
    writes.push_back(CarpetX::write(group, current_timelevel, leaves_valid,
                                    "ODESolvers::update"));
    dstmfabs.push_back(&writes.back().mfab());
  }
  combine(dstmfabs, scale, terms);
}

} // namespace ODESolvers
