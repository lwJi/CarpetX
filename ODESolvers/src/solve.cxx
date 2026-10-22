#include "solve.hxx"

namespace ODESolvers {
using namespace std;

////////////////////////////////////////////////////////////////////////////////

void statecomp_t::require_valid(const CarpetX::valid_t where,
                                const char *const why) const {
  for (const entry_t &entry : entries)
    if (!entry.tmp)
      CarpetX::require_valid(entry.group, timelevel, where, why);
}

statecomp_t statecomp_t::copy() const {
  statecomp_t result;
  result.timelevel = timelevel;
  result.entries.reserve(entries.size());
  for (const entry_t &entry : entries)
    result.entries.push_back(
        entry_t{entry.group, CarpetX::scratch(entry.group)});
  lincomb(result, 0, make_array(CCTK_REAL(1)), make_array(this),
          CarpetX::valid_t());
  return result;
}

namespace {

const amrex::MultiFab &source(const statecomp_t &src, const std::size_t m) {
  const statecomp_t::entry_t &entry = src.entries.at(m);
  if (entry.tmp)
    return entry.tmp->mfab();
  return CarpetX::data(entry.group, src.timelevel);
}

} // namespace

template <std::size_t N>
void statecomp_t::lincomb(statecomp_t &dst, const CCTK_REAL scale,
                          const std::array<CCTK_REAL, N> &factors,
                          const std::array<const statecomp_t *, N> &srcs,
                          const CarpetX::valid_t where) {
  const std::size_t size = dst.entries.size();
  for (std::size_t n = 0; n < N; ++n)
    assert(srcs[n]->entries.size() == size);

  std::vector<CarpetX::write_t> writes;
  std::vector<amrex::MultiFab *> dstmfabs;
  std::vector<std::array<const amrex::MultiFab *, N> > srcmfabs;
  dstmfabs.reserve(size);
  srcmfabs.reserve(size);
  for (std::size_t m = 0; m < size; ++m) {
    entry_t &entry = dst.entries.at(m);
    if (entry.tmp) {
      dstmfabs.push_back(&entry.tmp->mfab());
    } else {
      writes.push_back(CarpetX::write(entry.group, dst.timelevel, where,
                                      "ODESolvers::statecomp_t::lincomb"));
      dstmfabs.push_back(&writes.back().mfab());
    }
    std::array<const amrex::MultiFab *, N> sources;
    for (std::size_t n = 0; n < N; ++n)
      sources[n] = &source(*srcs[n], m);
    srcmfabs.push_back(sources);
  }
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

  // TODO: Poison ghosts/boundaries

  for (std::size_t m = 0; m < size; ++m) {
    amrex::MultiFab &dstmfab = *dstmfabs.at(m);
    const std::ptrdiff_t ncomps = dstmfab.nComp();
    const auto mfitinfo = amrex::MFItInfo().DisableDeviceSync();
    for (amrex::MFIter mfi(dstmfab, mfitinfo); mfi.isValid(); ++mfi) {
      const amrex::Array4<CCTK_REAL> dstvar = dstmfab.array(mfi);
      std::array<amrex::Array4<const CCTK_REAL>, N> srcvars;
      for (std::size_t n = 0; n < N; ++n)
        srcvars[n] = srcmfabs.at(m)[n]->const_array(mfi);
      // Array4's public stride members were replaced by get_stride() in
      // AMReX 26.02
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
      // CPU

      const std::ptrdiff_t ntiles = omp_get_max_threads();
      const std::ptrdiff_t tile_size = Arith::align_ceil(
          Arith::div_ceil(npoints, ntiles), std::ptrdiff_t(64));

      for (std::ptrdiff_t imin = 0; imin < npoints; imin += tile_size) {
        using std::min;
        const std::ptrdiff_t imax = min(npoints, imin + tile_size);

        if (!read_dst && N == 1 && factors[0] == 1) {
          // Copy

          auto task = [=]() {
            std::memcpy(&dstptr[imin], &srcptrs[0][imin],
                        (imax - imin) * sizeof *dstptr);
          };
          tasks.push_back(std::move(task));

        } else if (!read_dst && N >= 1 && factors[0] == 1) {
          // Write without scaling

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
          // Write

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
          // Update without scaling

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
          // Update

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
      } // for imin

#else
      // GPU

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
  // run all tasks
#pragma omp parallel for schedule(dynamic)
  for (std::size_t i = 0; i < tasks.size(); ++i)
    tasks[i]();
#else
  // wait for all tasks
  amrex::Gpu::synchronize();
  AMREX_GPU_ERROR_CHECK();
#endif
}

namespace detail {
template <std::size_t N>
void call_lincomb(statecomp_t &dst, const CCTK_REAL scale,
                  const std::vector<CCTK_REAL> &factors,
                  const std::vector<const statecomp_t *> &srcs,
                  const std::vector<std::size_t> &indices,
                  const CarpetX::valid_t where) {
  assert(indices.size() == N);
  std::array<CCTK_REAL, N> factors1;
  std::array<const statecomp_t *, N> srcs1;
  for (std::size_t n = 0; n < N; ++n) {
    factors1[n] = factors.at(indices[n]);
    srcs1[n] = srcs.at(indices[n]);
  }
  statecomp_t::lincomb(dst, scale, factors1, srcs1, where);
}
} // namespace detail

void statecomp_t::lincomb(statecomp_t &dst, const CCTK_REAL scale,
                          const std::vector<CCTK_REAL> &factors,
                          const std::vector<const statecomp_t *> &srcs,
                          const CarpetX::valid_t where) {
  const std::size_t N = factors.size();
  assert(srcs.size() == N);

  std::size_t NNZ = 0;
  for (std::size_t n = 0; n < N; ++n)
    NNZ += factors[n] != 0;
  std::vector<std::size_t> indices;
  indices.reserve(NNZ);
  for (std::size_t n = 0; n < N; ++n)
    if (factors[n] != 0)
      indices.push_back(n);
  assert(indices.size() == NNZ);

  switch (NNZ) {
  case 0:
    return detail::call_lincomb<0>(dst, scale, factors, srcs, indices, where);
  case 1:
    return detail::call_lincomb<1>(dst, scale, factors, srcs, indices, where);
  case 2:
    return detail::call_lincomb<2>(dst, scale, factors, srcs, indices, where);
  case 3:
    return detail::call_lincomb<3>(dst, scale, factors, srcs, indices, where);
  case 4:
    return detail::call_lincomb<4>(dst, scale, factors, srcs, indices, where);
  case 5:
    return detail::call_lincomb<5>(dst, scale, factors, srcs, indices, where);
  case 6:
    return detail::call_lincomb<6>(dst, scale, factors, srcs, indices, where);
  case 7:
    return detail::call_lincomb<7>(dst, scale, factors, srcs, indices, where);
  case 8:
    return detail::call_lincomb<8>(dst, scale, factors, srcs, indices, where);
  case 9:
    return detail::call_lincomb<9>(dst, scale, factors, srcs, indices, where);
  case 10:
    return detail::call_lincomb<10>(dst, scale, factors, srcs, indices, where);
  case 11:
    return detail::call_lincomb<11>(dst, scale, factors, srcs, indices, where);
  case 12:
    return detail::call_lincomb<12>(dst, scale, factors, srcs, indices, where);
  case 13:
    return detail::call_lincomb<13>(dst, scale, factors, srcs, indices, where);
  case 14:
    return detail::call_lincomb<14>(dst, scale, factors, srcs, indices, where);
  case 15:
    return detail::call_lincomb<15>(dst, scale, factors, srcs, indices, where);
  case 16:
    return detail::call_lincomb<16>(dst, scale, factors, srcs, indices, where);
  default:
    CCTK_VERROR("Unsupported vector length: %d", (int)NNZ);
  }
}

template void statecomp_t::lincomb<1>(statecomp_t &dst, CCTK_REAL scale,
                                      const array<CCTK_REAL, 1> &factors,
                                      const array<const statecomp_t *, 1> &srcs,
                                      const CarpetX::valid_t where);
template void statecomp_t::lincomb<2>(statecomp_t &dst, CCTK_REAL scale,
                                      const array<CCTK_REAL, 2> &factors,
                                      const array<const statecomp_t *, 2> &srcs,
                                      const CarpetX::valid_t where);
template void statecomp_t::lincomb<3>(statecomp_t &dst, CCTK_REAL scale,
                                      const array<CCTK_REAL, 3> &factors,
                                      const array<const statecomp_t *, 3> &srcs,
                                      const CarpetX::valid_t where);
template void statecomp_t::lincomb<4>(statecomp_t &dst, CCTK_REAL scale,
                                      const array<CCTK_REAL, 4> &factors,
                                      const array<const statecomp_t *, 4> &srcs,
                                      const CarpetX::valid_t where);
template void statecomp_t::lincomb<5>(statecomp_t &dst, CCTK_REAL scale,
                                      const array<CCTK_REAL, 5> &factors,
                                      const array<const statecomp_t *, 5> &srcs,
                                      const CarpetX::valid_t where);
template void statecomp_t::lincomb<6>(statecomp_t &dst, CCTK_REAL scale,
                                      const array<CCTK_REAL, 6> &factors,
                                      const array<const statecomp_t *, 6> &srcs,
                                      const CarpetX::valid_t where);
template void statecomp_t::lincomb<7>(statecomp_t &dst, CCTK_REAL scale,
                                      const array<CCTK_REAL, 7> &factors,
                                      const array<const statecomp_t *, 7> &srcs,
                                      const CarpetX::valid_t where);
template void statecomp_t::lincomb<8>(statecomp_t &dst, CCTK_REAL scale,
                                      const array<CCTK_REAL, 8> &factors,
                                      const array<const statecomp_t *, 8> &srcs,
                                      const CarpetX::valid_t where);
template void statecomp_t::lincomb<9>(statecomp_t &dst, CCTK_REAL scale,
                                      const array<CCTK_REAL, 9> &factors,
                                      const array<const statecomp_t *, 9> &srcs,
                                      const CarpetX::valid_t where);
template void statecomp_t::lincomb<10>(
    statecomp_t &dst, CCTK_REAL scale, const array<CCTK_REAL, 10> &factors,
    const array<const statecomp_t *, 10> &srcs, const CarpetX::valid_t where);
template void statecomp_t::lincomb<11>(
    statecomp_t &dst, CCTK_REAL scale, const array<CCTK_REAL, 11> &factors,
    const array<const statecomp_t *, 11> &srcs, const CarpetX::valid_t where);
template void statecomp_t::lincomb<12>(
    statecomp_t &dst, CCTK_REAL scale, const array<CCTK_REAL, 12> &factors,
    const array<const statecomp_t *, 12> &srcs, const CarpetX::valid_t where);
template void statecomp_t::lincomb<13>(
    statecomp_t &dst, CCTK_REAL scale, const array<CCTK_REAL, 13> &factors,
    const array<const statecomp_t *, 13> &srcs, const CarpetX::valid_t where);
template void statecomp_t::lincomb<14>(
    statecomp_t &dst, CCTK_REAL scale, const array<CCTK_REAL, 14> &factors,
    const array<const statecomp_t *, 14> &srcs, const CarpetX::valid_t where);
template void statecomp_t::lincomb<15>(
    statecomp_t &dst, CCTK_REAL scale, const array<CCTK_REAL, 15> &factors,
    const array<const statecomp_t *, 15> &srcs, const CarpetX::valid_t where);
template void statecomp_t::lincomb<16>(
    statecomp_t &dst, CCTK_REAL scale, const array<CCTK_REAL, 16> &factors,
    const array<const statecomp_t *, 16> &srcs, const CarpetX::valid_t where);

} // namespace ODESolvers
