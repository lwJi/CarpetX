#include <carpetx.hxx>
#include <carpetx_rk.hxx>
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

int group(const char *const name) {
  const int gi = CCTK_GroupIndex(name);
  if (gi < 0)
    CCTK_VERROR("Group %s does not exist", name);
  return gi;
}

CarpetX::rk_config_t integrator(const char *const state, const char *const rhs,
                                const std::vector<CCTK_REAL> &b,
                                const CarpetX::rk_dense_output_t dense_output) {
  CarpetX::rk_config_t config;
  config.dense_output = dense_output;
  config.b = b;
  config.integrated.push_back(CarpetX::rk_pair_t{group(state), group(rhs)});
  return config;
}

const std::vector<CCTK_REAL> rk4_weights{1.0 / 6, 1.0 / 3, 1.0 / 3, 1.0 / 6};

CarpetX::rk_config_t valid_integrator() {
  return integrator("TestCarpetXAPI::state", "TestCarpetXAPI::state_rhs",
                    rk4_weights, CarpetX::rk_dense_output_t::rk4);
}

bool rk_step_mode(const char *const mode) {
  return CCTK_EQUALS(mode, "rhs-repeated") ||
         CCTK_EQUALS(mode, "step-closed-early");
}

extern "C" int TestCarpetXAPI_Startup() {
  DECLARE_CCTK_PARAMETERS;

  if (CCTK_EQUALS(mode, "window-outside-traversal")) {
    const CarpetX::window_t window = CarpetX::active_window();
    CCTK_VERROR("CarpetX::active_window reported levels [%d,%d) at STARTUP, "
                "where the driver has not opened a traversal",
                window.min_level, window.max_level);
  }
  return 0;
}

extern "C" void TestCarpetXAPI_Register(CCTK_ARGUMENTS) {
  DECLARE_CCTK_PARAMETERS;

  if (rk_step_mode(mode))
    CarpetX::register_integrator(valid_integrator());
  if (CCTK_EQUALS(mode, "register-twice")) {
    CarpetX::register_integrator(valid_integrator());
    CarpetX::register_integrator(valid_integrator());
    CCTK_ERROR("CarpetX::register_integrator accepted a second registration");
  }
  if (CCTK_EQUALS(mode, "dense-output-stage-mismatch")) {
    CarpetX::register_integrator(
        integrator("TestCarpetXAPI::state", "TestCarpetXAPI::state_rhs",
                   {0.5, 0.5}, CarpetX::rk_dense_output_t::rk4));
    CCTK_ERROR("CarpetX::register_integrator accepted 2 RK stages with the "
               "RK4 dense output");
  }
  if (CCTK_EQUALS(mode, "weights-sum-mismatch")) {
    CarpetX::register_integrator(
        integrator("TestCarpetXAPI::state", "TestCarpetXAPI::state_rhs",
                   {0.5, 0.5, 0.5}, CarpetX::rk_dense_output_t::none));
    CCTK_ERROR("CarpetX::register_integrator accepted RK stage weights that "
               "sum to 1.5");
  }
  if (CCTK_EQUALS(mode, "reflux-without-weights")) {
    CarpetX::register_integrator(
        integrator("TestCarpetXAPI::conserved", "TestCarpetXAPI::conserved_rhs",
                   {}, CarpetX::rk_dense_output_t::none));
    CCTK_ERROR("CarpetX::register_integrator accepted an integrator without "
               "weights for a flux-tagged integrated group under "
               "CarpetX::do_reflux");
  }
  if (CCTK_EQUALS(mode, "subcycling-without-dense-output")) {
    CarpetX::register_integrator(
        integrator("TestCarpetXAPI::state", "TestCarpetXAPI::state_rhs",
                   rk4_weights, CarpetX::rk_dense_output_t::none));
    CCTK_ERROR("CarpetX::register_integrator accepted an integrator without "
               "a dense output under subcycling");
  }
  if (CCTK_EQUALS(mode, "integrated-not-checkpointed")) {
    CarpetX::register_integrator(
        integrator("TestCarpetXAPI::unsaved", "TestCarpetXAPI::unsaved_rhs",
                   rk4_weights, CarpetX::rk_dense_output_t::rk4));
    CCTK_ERROR("CarpetX::register_integrator accepted an integrated group "
               "tagged checkpoint=\"no\"");
  }
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

  if (CCTK_EQUALS(mode, "register-after-use")) {
    CarpetX::register_integrator(valid_integrator());
    CCTK_ERROR("CarpetX::register_integrator accepted a registration at "
               "CCTK_INITIAL, after the driver read its integrator record");
  }
  if (CCTK_EQUALS(mode, "unregistered-rk-call")) {
    const CarpetX::rk_step_t step =
        CarpetX::open_rk_step(CarpetX::all_levels());
    CCTK_ERROR("CarpetX::open_rk_step ran without a registered integrator");
  }

  const int gi = state_group();
  const std::vector<int> groups{gi};
  const CarpetX::window_t window = CarpetX::active_window();

  if (CCTK_EQUALS(mode, "level-missing")) {
    const CarpetX::write_t write =
        CarpetX::write(CarpetX::group_t{CarpetX::level_t{0, 99}, gi}, tl,
                       CarpetX::make_valid_int(), "TestCarpetXAPI_Fill");
    CCTK_ERROR("CarpetX::write accepted a handle to refinement level 99, "
               "which does not exist");
  }
  if (CCTK_EQUALS(mode, "timelevel-missing")) {
    const CarpetX::write_t write =
        CarpetX::write(CarpetX::group_t{CarpetX::level_t{0, 0}, gi}, 7,
                       CarpetX::make_valid_int(), "TestCarpetXAPI_Fill");
    CCTK_ERROR("CarpetX::write accepted time level 7 of state, which has no "
               "such time level");
  }

  if (CCTK_EQUALS(mode, "prolongate-unwritten-parent") &&
      window.max_level > 1) {
    CarpetX::prolongate_ghosts(window, {group("TestCarpetXAPI::state_rhs")});
    CCTK_ERROR("CarpetX::prolongate_ghosts filled the ghosts of state_rhs on "
               "level 1 from a parent level that nothing has written");
  }
  if (CCTK_EQUALS(mode, "sync-invalid-coarse") && window.max_level > 1) {
    std::vector<CarpetX::write_t> writes;
    window.loop_serially([&](const CarpetX::level_t level) {
      writes.push_back(CarpetX::write(CarpetX::group_t{level, gi}, tl,
                                      CarpetX::make_valid_int(),
                                      "TestCarpetXAPI_Fill"));
      fill_interior(level, writes.back().mfab());
    });
    writes.clear();
    CarpetX::invalidate(
        CarpetX::window_t{0, 1, window.min_patch, window.max_patch}, groups, tl,
        "TestCarpetXAPI_Fill");
    CarpetX::sync(window, groups);
    CCTK_ERROR("CarpetX::sync read the interior of state on level 0 after it "
               "had been invalidated");
  }

  if (window.min_level != 0 || window.max_level != 1)
    return;

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
  if (CCTK_EQUALS(mode, "write-unfilled")) {
    window.loop_serially([&](const CarpetX::level_t level) {
      CarpetX::write_t write =
          CarpetX::write(CarpetX::group_t{level, gi}, tl,
                         CarpetX::make_valid_all(), "TestCarpetXAPI_Fill");
      fill_interior(level, write.mfab());
    });
    CCTK_ERROR("CarpetX::write_t closed a scope that promised every region "
               "of state after only the interior was filled");
  }

  std::vector<CarpetX::write_t> writes;
  window.loop_serially([&](const CarpetX::level_t level) {
    writes.push_back(CarpetX::write(CarpetX::group_t{level, gi}, tl,
                                    CarpetX::make_valid_int(),
                                    "TestCarpetXAPI_Fill"));
    fill_interior(level, writes.back().mfab());
    if (rk_step_mode(mode)) {
      writes.push_back(CarpetX::write(
          CarpetX::group_t{level, group("TestCarpetXAPI::state_rhs")}, tl,
          CarpetX::make_valid_int(), "TestCarpetXAPI_Fill"));
      fill_interior(level, writes.back().mfab());
    }
  });

  if (CCTK_EQUALS(mode, "write-twice")) {
    window.loop_serially([&](const CarpetX::level_t level) {
      writes.push_back(CarpetX::write(CarpetX::group_t{level, gi}, tl,
                                      CarpetX::make_valid_int(),
                                      "TestCarpetXAPI_Fill, second scope"));
    });
    CCTK_ERROR("CarpetX::write opened a second scope on a group and time "
               "level that already had one open");
  }
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

  if (CCTK_EQUALS(mode, "rhs-repeated")) {
    CarpetX::rk_step_t step = CarpetX::open_rk_step(window);
    step.rhs_computed();
    step.rhs_computed();
    CCTK_ERROR("CarpetX::rk_step_t::rhs_computed ran twice without "
               "state_updated in between");
  }
  if (CCTK_EQUALS(mode, "step-closed-early")) {
    {
      CarpetX::rk_step_t step = CarpetX::open_rk_step(window);
      step.rhs_computed();
      step.state_updated();
    }
    CCTK_ERROR("CarpetX::rk_step_t closed after 1 of 4 stages");
  }
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
