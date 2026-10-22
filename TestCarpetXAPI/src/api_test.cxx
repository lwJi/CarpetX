#include <carpetx.hxx>
#include <loop.hxx>

#include <cctk.h>
#include <cctk_Arguments.h>
#include <cctk_Parameters.h>

#include <AMReX_Array4.H>
#include <AMReX_Geometry.H>
#include <AMReX_MFIter.H>
#include <AMReX_MultiFab.H>

#include <cassert>
#include <cmath>
#include <vector>

namespace TestCarpetXAPI {
using namespace Loop;

constexpr int tl = 0;

template <typename T>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE T written_function(const T x,
                                                            const T y,
                                                            const T z) {
  using std::acos, std::cos;
  const T pi = acos(-T(1));
  return cos(pi * x) * cos(pi * y) * cos(pi * z);
}

int state_group() {
  const int gi = CCTK_GroupIndex("TestCarpetXAPI::state");
  assert(gi >= 0);
  return gi;
}

void require_flags(const CarpetX::group_t group,
                   const CarpetX::valid_t expected, const char *const when) {
  const CarpetX::valid_t found = CarpetX::valid(group, tl, 0);
  if (found != expected)
    CCTK_VERROR("TestCarpetXAPI::state on patch %d, refinement level %d is "
                "%s %s; expected %s",
                group.level.patch, group.level.level,
                std::string(found).c_str(), when,
                std::string(expected).c_str());
}

void fill_interior(const CarpetX::level_t level, amrex::MultiFab &mfab) {
  const amrex::Geometry &geom = CarpetX::geometry(level);
  const CCTK_REAL x0 = geom.ProbLo(0), y0 = geom.ProbLo(1), z0 = geom.ProbLo(2);
  const CCTK_REAL dx = geom.CellSize(0), dy = geom.CellSize(1),
                  dz = geom.CellSize(2);
  for (amrex::MFIter mfi(mfab); mfi.isValid(); ++mfi) {
    const amrex::Array4<CCTK_REAL> arr = mfab.array(mfi);
    amrex::ParallelFor(
        mfi.validbox(),
        [=] AMREX_GPU_DEVICE(const int i, const int j, const int k) {
          arr(i, j, k, 0) = written_function(x0 + (i + CCTK_REAL(0.5)) * dx,
                                             y0 + (j + CCTK_REAL(0.5)) * dy,
                                             z0 + (k + CCTK_REAL(0.5)) * dz);
        });
  }
  CarpetX::synchronize();
}

extern "C" void TestCarpetXAPI_Fill(CCTK_ARGUMENTS) {
  DECLARE_CCTK_PARAMETERS;

  const int gi = state_group();
  const std::vector<int> groups{gi};
  const CarpetX::window_t window = CarpetX::active_window();

  window.loop_serially([&](const CarpetX::level_t level) {
    const CarpetX::group_t group{level, gi};
    if (CarpetX::numvars(group) != 1)
      CCTK_VERROR("CarpetX::numvars reports %d variables for "
                  "TestCarpetXAPI::state; expected 1",
                  CarpetX::numvars(group));
    const auto indextype = CarpetX::indextype(group);
    if (indextype != std::array<int, dim>{1, 1, 1})
      CCTK_VERROR("CarpetX::indextype reports [%d,%d,%d] for "
                  "TestCarpetXAPI::state; expected [1,1,1]",
                  indextype[0], indextype[1], indextype[2]);
    const auto nghostzones = CarpetX::nghostzones(group);
    const amrex::IntVect ng = CarpetX::data(group, tl).nGrowVect();
    if (nghostzones != std::array<int, dim>{ng[0], ng[1], ng[2]})
      CCTK_VERROR("CarpetX::nghostzones reports [%d,%d,%d] for "
                  "TestCarpetXAPI::state, but CarpetX::data has [%d,%d,%d] "
                  "ghost zones",
                  nghostzones[0], nghostzones[1], nghostzones[2], ng[0], ng[1],
                  ng[2]);
    require_flags(group, CarpetX::valid_t(), "before anything wrote it");
  });

  if (CCTK_EQUALS(mode, "require-invalid")) {
    window.loop_serially([&](const CarpetX::level_t level) {
      CarpetX::require_valid(CarpetX::group_t{level, gi}, tl,
                             CarpetX::make_valid_ghosts(),
                             "TestCarpetXAPI_Fill before any sync");
    });
    CCTK_ERROR("CarpetX::require_valid accepted ghosts that nothing has "
               "filled");
  }

  std::vector<CarpetX::write_t> writes;
  window.loop_serially([&](const CarpetX::level_t level) {
    writes.push_back(CarpetX::write(CarpetX::group_t{level, gi}, tl,
                                    CarpetX::make_valid_int(),
                                    "TestCarpetXAPI_Fill"));
    fill_interior(level, writes.back().mfab());
  });

  if (CCTK_EQUALS(mode, "sync-during-write")) {
    CarpetX::sync(window, groups);
    CCTK_ERROR("CarpetX::sync ran while a write scope was open");
  }

  writes.clear();

  window.loop_serially([&](const CarpetX::level_t level) {
    require_flags(CarpetX::group_t{level, gi}, CarpetX::make_valid_int(),
                  "after the write scope closed");
  });

  CarpetX::sync(window, groups);

  window.loop_serially([&](const CarpetX::level_t level) {
    const CarpetX::group_t group{level, gi};
    require_flags(group, CarpetX::make_valid_all(), "after CarpetX::sync");
    const amrex::MultiFab &mfab = CarpetX::data(group, tl);
    if (mfab.nComp() != 1)
      CCTK_VERROR("CarpetX::data returned a MultiFab with %d components for "
                  "TestCarpetXAPI::state; expected 1",
                  mfab.nComp());
    CarpetX::require_valid(group, tl, CarpetX::make_valid_all(),
                           "TestCarpetXAPI_Fill after CarpetX::sync");
  });

  const CarpetX::reduction<CCTK_REAL, dim> red = CarpetX::reduce(gi, 0, tl);
  if (!(red.min >= -1 && red.max <= 1 && red.vol > 0))
    CCTK_VERROR("CarpetX::reduce reports min %g, max %g, vol %g for "
                "TestCarpetXAPI::state; expected a product of cosines on a "
                "domain of positive volume",
                double(red.min), double(red.max), double(red.vol));
}

extern "C" void TestCarpetXAPI_Check(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTSX_TestCarpetXAPI_Check;
  DECLARE_CCTK_PARAMETERS;

  int nbad = 0;
  vect<CCTK_REAL, dim> bad_X{0, 0, 0};
  CCTK_REAL bad_value = 0, bad_expected = 0;
  grid.loop_all<1, 1, 1>(grid.nghostzones, [&](const PointDesc &p) {
    const CCTK_REAL expected = written_function(p.X[0], p.X[1], p.X[2]);
    using std::fabs;
    if (!(fabs(state(p.I) - expected) <= tolerance)) {
      if (nbad == 0) {
        bad_X = p.X;
        bad_value = state(p.I);
        bad_expected = expected;
      }
      ++nbad;
    }
  });

  if (nbad != 0) {
#pragma omp critical(TestCarpetXAPI_Check)
    CCTK_VERROR("TestCarpetXAPI::state on level %d holds %.17g at [%g,%g,%g], "
                "expected %.17g; %d mismatches on this tile",
                cctk_level, double(bad_value), double(bad_X[0]),
                double(bad_X[1]), double(bad_X[2]), double(bad_expected), nbad);
  }
}

} // namespace TestCarpetXAPI
