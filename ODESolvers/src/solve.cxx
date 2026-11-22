#include "solve.hxx"

namespace ODESolvers {
using namespace std;

////////////////////////////////////////////////////////////////////////////////

statecomp_t statecomp_t::copy() const { return scaled(1); }

statecomp_t statecomp_t::scaled(const CCTK_REAL factor) const {
  statecomp_t result;
  result.timelevel = timelevel;
  result.entries.reserve(entries.size());
  for (const entry_t &entry : entries)
    result.entries.push_back(
        entry_t{entry.group, CarpetX::scratch(entry.group)});
  lincomb(result, 0, make_array(factor), make_array(this));
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
                          const std::array<const statecomp_t *, N> &srcs) {
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
      writes.push_back(CarpetX::write(entry.group, dst.timelevel,
                                      CarpetX::make_valid_int(),
                                      "ODESolvers::statecomp_t::lincomb"));
      dstmfabs.push_back(&writes.back().mfab());
    }
    std::array<const amrex::MultiFab *, N> sources;
    for (std::size_t n = 0; n < N; ++n)
      sources[n] = &source(*srcs[n], m);
    srcmfabs.push_back(sources);
  }
  for (std::size_t m = 0; m < size; ++m) {
    for (std::size_t n = 0; n < N; ++n) {
      assert(srcmfabs.at(m)[n]->nComp() == dstmfabs.at(m)->nComp());
      assert(srcmfabs.at(m)[n]->boxArray() == dstmfabs.at(m)->boxArray());
      assert(srcmfabs.at(m)[n]->DistributionMap() ==
             dstmfabs.at(m)->DistributionMap());
    }
  }

  using std::isfinite;
  assert(isfinite(scale));
  for (std::size_t n = 0; n < N; ++n)
    assert(isfinite(factors[n]));

  const int untiled = 1024000;
  const amrex::IntVect zslabs(untiled, untiled,
                              amrex::FabArrayBase::mfiter_tile_size[2]);
  const auto mfitinfo = amrex::MFItInfo()
                            .SetDynamic(true)
                            .EnableTiling(zslabs)
                            .DisableDeviceSync();
  for (std::size_t m = 0; m < size; ++m) {
    amrex::MultiFab &dstmfab = *dstmfabs.at(m);
    const std::array<const amrex::MultiFab *, N> &sources = srcmfabs.at(m);
    const int ncomp = dstmfab.nComp();
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
    for (amrex::MFIter mfi(dstmfab, mfitinfo); mfi.isValid(); ++mfi) {
      const amrex::Box box = mfi.tilebox();
      const amrex::Array4<CCTK_REAL> dstvar = dstmfab.array(mfi);
      std::array<amrex::Array4<const CCTK_REAL>, N> srcvars;
      for (std::size_t s = 0; s < N; ++s)
        srcvars[s] = sources[s]->const_array(mfi);

      if (scale == 0)
        amrex::ParallelFor(box, ncomp,
                           [=] CCTK_DEVICE(const int i, const int j,
                                           const int k, const int n) {
                             CCTK_REAL accum = 0;
                             for (std::size_t s = 0; s < N; ++s)
                               accum += factors[s] * srcvars[s](i, j, k, n);
                             dstvar(i, j, k, n) = accum;
                           });
      else
        amrex::ParallelFor(box, ncomp,
                           [=] CCTK_DEVICE(const int i, const int j,
                                           const int k, const int n) {
                             CCTK_REAL accum = scale * dstvar(i, j, k, n);
                             for (std::size_t s = 0; s < N; ++s)
                               accum += factors[s] * srcvars[s](i, j, k, n);
                             dstvar(i, j, k, n) = accum;
                           });
    }
  }
  CarpetX::synchronize();
}

namespace detail {
template <std::size_t N>
void call_lincomb(statecomp_t &dst, const CCTK_REAL scale,
                  const std::vector<CCTK_REAL> &factors,
                  const std::vector<const statecomp_t *> &srcs,
                  const std::vector<std::size_t> &indices) {
  assert(indices.size() == N);
  std::array<CCTK_REAL, N> factors1;
  std::array<const statecomp_t *, N> srcs1;
  for (std::size_t n = 0; n < N; ++n) {
    factors1[n] = factors.at(indices[n]);
    srcs1[n] = srcs.at(indices[n]);
  }
  statecomp_t::lincomb(dst, scale, factors1, srcs1);
}
} // namespace detail

void statecomp_t::lincomb(statecomp_t &dst, const CCTK_REAL scale,
                          const std::vector<CCTK_REAL> &factors,
                          const std::vector<const statecomp_t *> &srcs) {
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
    return detail::call_lincomb<0>(dst, scale, factors, srcs, indices);
  case 1:
    return detail::call_lincomb<1>(dst, scale, factors, srcs, indices);
  case 2:
    return detail::call_lincomb<2>(dst, scale, factors, srcs, indices);
  case 3:
    return detail::call_lincomb<3>(dst, scale, factors, srcs, indices);
  case 4:
    return detail::call_lincomb<4>(dst, scale, factors, srcs, indices);
  case 5:
    return detail::call_lincomb<5>(dst, scale, factors, srcs, indices);
  case 6:
    return detail::call_lincomb<6>(dst, scale, factors, srcs, indices);
  case 7:
    return detail::call_lincomb<7>(dst, scale, factors, srcs, indices);
  case 8:
    return detail::call_lincomb<8>(dst, scale, factors, srcs, indices);
  case 9:
    return detail::call_lincomb<9>(dst, scale, factors, srcs, indices);
  case 10:
    return detail::call_lincomb<10>(dst, scale, factors, srcs, indices);
  case 11:
    return detail::call_lincomb<11>(dst, scale, factors, srcs, indices);
  case 12:
    return detail::call_lincomb<12>(dst, scale, factors, srcs, indices);
  case 13:
    return detail::call_lincomb<13>(dst, scale, factors, srcs, indices);
  case 14:
    return detail::call_lincomb<14>(dst, scale, factors, srcs, indices);
  case 15:
    return detail::call_lincomb<15>(dst, scale, factors, srcs, indices);
  case 16:
    return detail::call_lincomb<16>(dst, scale, factors, srcs, indices);
  default:
    CCTK_VERROR("Unsupported vector length: %d", (int)NNZ);
  }
}

template void
statecomp_t::lincomb<1>(statecomp_t &dst, CCTK_REAL scale,
                        const array<CCTK_REAL, 1> &factors,
                        const array<const statecomp_t *, 1> &srcs);
template void
statecomp_t::lincomb<2>(statecomp_t &dst, CCTK_REAL scale,
                        const array<CCTK_REAL, 2> &factors,
                        const array<const statecomp_t *, 2> &srcs);
template void
statecomp_t::lincomb<3>(statecomp_t &dst, CCTK_REAL scale,
                        const array<CCTK_REAL, 3> &factors,
                        const array<const statecomp_t *, 3> &srcs);
template void
statecomp_t::lincomb<4>(statecomp_t &dst, CCTK_REAL scale,
                        const array<CCTK_REAL, 4> &factors,
                        const array<const statecomp_t *, 4> &srcs);
template void
statecomp_t::lincomb<5>(statecomp_t &dst, CCTK_REAL scale,
                        const array<CCTK_REAL, 5> &factors,
                        const array<const statecomp_t *, 5> &srcs);
template void
statecomp_t::lincomb<6>(statecomp_t &dst, CCTK_REAL scale,
                        const array<CCTK_REAL, 6> &factors,
                        const array<const statecomp_t *, 6> &srcs);
template void
statecomp_t::lincomb<7>(statecomp_t &dst, CCTK_REAL scale,
                        const array<CCTK_REAL, 7> &factors,
                        const array<const statecomp_t *, 7> &srcs);
template void
statecomp_t::lincomb<8>(statecomp_t &dst, CCTK_REAL scale,
                        const array<CCTK_REAL, 8> &factors,
                        const array<const statecomp_t *, 8> &srcs);
template void
statecomp_t::lincomb<9>(statecomp_t &dst, CCTK_REAL scale,
                        const array<CCTK_REAL, 9> &factors,
                        const array<const statecomp_t *, 9> &srcs);
template void
statecomp_t::lincomb<10>(statecomp_t &dst, CCTK_REAL scale,
                         const array<CCTK_REAL, 10> &factors,
                         const array<const statecomp_t *, 10> &srcs);
template void
statecomp_t::lincomb<11>(statecomp_t &dst, CCTK_REAL scale,
                         const array<CCTK_REAL, 11> &factors,
                         const array<const statecomp_t *, 11> &srcs);
template void
statecomp_t::lincomb<12>(statecomp_t &dst, CCTK_REAL scale,
                         const array<CCTK_REAL, 12> &factors,
                         const array<const statecomp_t *, 12> &srcs);
template void
statecomp_t::lincomb<13>(statecomp_t &dst, CCTK_REAL scale,
                         const array<CCTK_REAL, 13> &factors,
                         const array<const statecomp_t *, 13> &srcs);
template void
statecomp_t::lincomb<14>(statecomp_t &dst, CCTK_REAL scale,
                         const array<CCTK_REAL, 14> &factors,
                         const array<const statecomp_t *, 14> &srcs);
template void
statecomp_t::lincomb<15>(statecomp_t &dst, CCTK_REAL scale,
                         const array<CCTK_REAL, 15> &factors,
                         const array<const statecomp_t *, 15> &srcs);
template void
statecomp_t::lincomb<16>(statecomp_t &dst, CCTK_REAL scale,
                         const array<CCTK_REAL, 16> &factors,
                         const array<const statecomp_t *, 16> &srcs);

} // namespace ODESolvers
