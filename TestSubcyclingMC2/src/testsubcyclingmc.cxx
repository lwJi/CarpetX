#include <loop_device.hxx>
#include <subcycling.hxx>

#include <sum.hxx>
#include <vect.hxx>

#include <cctk.h>
#include <cctk_Arguments.h>
#include <cctk_Parameters.h>

#include <array>
#include <cassert>
#include <cmath>
#include <limits>

namespace TestSubcyclingMC2 {
using namespace Arith;

constexpr int dim = 3;

// u(t,x,y,z) =
//   A cos(2 pi omega t) sin(2 pi kx x) sin(2 pi ky y) sin(2 pi kz z)
template <typename T>
constexpr void standing_wave(const T A, const T kx, const T ky, const T kz,
                             const T t, const T x, const T y, const T z, T &u,
                             T &rho) {
  using std::acos, std::cos, std::pow, std::sin, std::sqrt;

  const T pi = acos(-T(1));
  const T omega = sqrt(pow(kx, 2) + pow(ky, 2) + pow(kz, 2));

  u = A * cos(2 * pi * omega * t) * cos(2 * pi * kx * x) *
      cos(2 * pi * ky * y) * cos(2 * pi * kz * z);
  rho = A * (-2 * pi * omega) * sin(2 * pi * omega * t) * cos(2 * pi * kx * x) *
        cos(2 * pi * ky * y) * cos(2 * pi * kz * z);
}

// u(t,r) = (f(t-r) - f(t+r)) / r
// f(v) = A exp(-1/2 (r/W)^2)
template <typename T>
constexpr void gaussian(const T A, const T W, const T t, const T x, const T y,
                        const T z, T &u, T &rho) {
  using std::exp, std::pow, std::sqrt;

  const T r = sqrt(pow(x, 2) + pow(y, 2) + pow(z, 2));
  const auto f = [&](const T v) {
    return A * exp(-pow(v, 2) / (2 * pow(W, 2)));
  };

  if (r < sqrt(std::numeric_limits<T>::epsilon())) {
    // L'Hôpital
    u = 2 / pow(W, 2) * f(t) * t;
    rho = -2 / pow(W, 4) * f(t) * (pow(t, 2) - pow(W, 2));
  } else {
    u = (f(t - r) - f(t + r)) / r;
    rho = -(f(t - r) * (t - r) - f(t + r) * (t + r)) / (pow(W, 2) * r);
  }
}

extern "C" void TestSubcyclingMC2_Initial(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTSX_TestSubcyclingMC2_Initial;
  DECLARE_CCTK_PARAMETERS;

  if (CCTK_EQUALS(initial_condition, "standing wave")) {
    grid.loop_int_device<0, 0, 0>(
        grid.nghostzones,
        [=] CCTK_DEVICE(const Loop::PointDesc &p) CCTK_ATTRIBUTE_ALWAYS_INLINE {
          standing_wave(amplitude, standing_wave_kx, standing_wave_ky,
                        standing_wave_kz, cctk_time, p.x, p.y, p.z, u(p.I),
                        rho(p.I));
        });

  } else if (CCTK_EQUALS(initial_condition, "Gaussian")) {
    grid.loop_int_device<0, 0, 0>(
        grid.nghostzones,
        [=] CCTK_DEVICE(const Loop::PointDesc &p) CCTK_ATTRIBUTE_ALWAYS_INLINE {
          gaussian(amplitude, gaussian_width, cctk_time, p.x, p.y, p.z, u(p.I),
                   rho(p.I));
        });

  } else {
    CCTK_ERROR("Unknown initial condition");
  }

  grid.loop_int_device<0, 0, 0>(grid.nghostzones,
                                [=] CCTK_DEVICE(const Loop::PointDesc &p)
                                    CCTK_ATTRIBUTE_ALWAYS_INLINE {
                                      u_k1(p.I) = 0.0;
                                      u_k2(p.I) = 0.0;
                                      u_k3(p.I) = 0.0;
                                      u_k4(p.I) = 0.0;
                                      rho_k1(p.I) = 0.0;
                                      rho_k2(p.I) = 0.0;
                                      rho_k3(p.I) = 0.0;
                                      rho_k4(p.I) = 0.0;
                                    });
}

/**
 * \brief Calculate RHSs from Us
 *          rhs = RHS(u),
 *
 * \param vlr       RHSs of state vector Us
 * \param vlu       state vector Us
 */
template <int D, typename tVarOut>
CCTK_DEVICE CCTK_HOST CCTK_ATTRIBUTE_ALWAYS_INLINE inline void
CalcRhs(const Loop::GridDescBaseDevice &grid, const array<tVarOut, D> &vlr,
        const array<tVarOut, D> &vlu) {
  tVarOut &u_rhs = vlr[0];
  tVarOut &rho_rhs = vlr[1];
  tVarOut &u = vlu[0];
  tVarOut &rho = vlu[1];

  grid.loop_int_device<0, 0, 0>(
      grid.nghostzones,
      [=] CCTK_DEVICE(const Loop::PointDesc &p) CCTK_ATTRIBUTE_ALWAYS_INLINE {
        using std::pow;
        Arith::vect<CCTK_REAL, dim> ddu;
        for (int d = 0; d < dim; ++d) {
          // ddu[d] = (u(p.I - p.DI[d]) - 2 * u(p.I) + u(p.I + p.DI[d])) /
          //          pow(p.DX[d], 2);
          ddu[d] = (-(u(p.I - 2 * p.DI[d]) + u(p.I + 2 * p.DI[d])) +
                    16 * (u(p.I - p.DI[d]) + u(p.I + p.DI[d])) - 30 * u(p.I)) /
                   (12 * pow(p.DX[d], 2));
        }
        u_rhs(p.I) = rho(p.I);
        rho_rhs(p.I) = ddu[0] + ddu[1] + ddu[2];
      });
}

/**
 * \brief Update state vector Us from RHSs and old Us
 *          u   += rhs * dt.
 *
 * \param vlu       state vector Us
 * \param vlr       RHS of Us
 * \param dt        time factor of each RK substep
 */
template <int D, typename tVarOut>
CCTK_DEVICE CCTK_HOST CCTK_ATTRIBUTE_ALWAYS_INLINE inline void
UpdateU(const Loop::GridDescBaseDevice &grid, const array<tVarOut, D> &vlu,
        const array<tVarOut, D> &vlr, const CCTK_REAL dt) {
  for (size_t v = 0; v < D; ++v) {
    grid.loop_int_device<0, 0, 0>(
        grid.nghostzones,
        [=] CCTK_DEVICE(const Loop::PointDesc &p)
            CCTK_ATTRIBUTE_ALWAYS_INLINE { vlu[v](p.I) += vlr[v](p.I) * dt; });
  }
}

/**
 * \brief Calculate Ks from Ys, Update state vector Us from Ks and old Us
 *          rhs = RHS(w),
 *          u   += rhs * dt.
 *
 * \param vlr       RK Ks
 * \param vlw       RK substage Ys
 * \param vlu       state vector Us
 * \param dt        time factor of each RK substep
 */
template <int D, typename tVarOut>
CCTK_DEVICE CCTK_HOST CCTK_ATTRIBUTE_ALWAYS_INLINE inline void
CalcRhsAndUpdateU(const Loop::GridDescBaseDevice &grid,
                  const array<tVarOut, D> &vlr, const array<tVarOut, D> &vlw,
                  const array<tVarOut, D> &vlu, const CCTK_REAL dt) {
  CalcRhs<D>(grid, vlr, vlw);
  UpdateU<D>(grid, vlu, vlr, dt);
}

/**
 * \brief Calculate Ys from U0 and rhs
 *          w = u_p + rhs * dt
 *
 * \param vlw       RK substage Ys
 * \param vlp       state vector U0
 * \param vlr       RK Ks
 * \param dt        time factor of each RK substage
 */
template <int D, typename tVarOut, typename tVarIn>
CCTK_DEVICE CCTK_HOST CCTK_ATTRIBUTE_ALWAYS_INLINE inline void
CalcYs(const Loop::GridDescBaseDevice &grid, const array<tVarOut, D> &vlw,
       const array<tVarIn, D> &vlp, const array<tVarOut, D> &vlr,
       const CCTK_REAL dt) {
  for (size_t v = 0; v < D; ++v) {
    grid.loop_int_device<0, 0, 0>(
        grid.nghostzones,
        [=] CCTK_DEVICE(const Loop::PointDesc &p) CCTK_ATTRIBUTE_ALWAYS_INLINE {
          vlw[v](p.I) = vlp[v](p.I) + vlr[v](p.I) * dt;
        });
  }
}

extern "C" void TestSubcyclingMC2_SetP(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTSX_TestSubcyclingMC2_SetP;

  CCTK_VINFO("Updating grid function at iteration %d level %d time %g",
             cctk_iteration, cctk_level, cctk_time);
  grid.loop_int_device<0, 0, 0>(grid.nghostzones,
                                [=] CCTK_DEVICE(const Loop::PointDesc &p)
                                    CCTK_ATTRIBUTE_ALWAYS_INLINE {
                                      u_p(p.I) = u(p.I);
                                      rho_p(p.I) = rho(p.I);
                                    });
}

extern "C" void TestSubcyclingMC2_CalcK1(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTSX_TestSubcyclingMC2_CalcK1;
  DECLARE_CCTK_PARAMETERS;
  const CCTK_REAL dt = CCTK_DELTA_TIME;
  const array<const Loop::GF3D2<CCTK_REAL>, 2> k1{u_k1, rho_k1};
  const array<const Loop::GF3D2<CCTK_REAL>, 2> vlu{u, rho};
  const array<const Loop::GF3D2<CCTK_REAL>, 2> vlw{u_w, rho_w};
  const array<const Loop::GF3D2<const CCTK_REAL>, 2> vlp{u_p, rho_p};
  constexpr size_t nvars = vlu.size();

  if (use_subcycling_wip) {
    vector<int> u_groups, p_groups;
    array<vector<int>, 4> ks_groups;
    u_groups.push_back(CCTK_GroupIndex("TestSubcyclingMC2::ustate"));
    p_groups.push_back(CCTK_GroupIndex("TestSubcyclingMC2::pstate"));
    ks_groups[0].push_back(CCTK_GroupIndex("TestSubcyclingMC2::k1"));
    ks_groups[1].push_back(CCTK_GroupIndex("TestSubcyclingMC2::k2"));
    ks_groups[2].push_back(CCTK_GroupIndex("TestSubcyclingMC2::k3"));
    ks_groups[3].push_back(CCTK_GroupIndex("TestSubcyclingMC2::k4"));
    const CCTK_REAL xsi = (cctk_iteration % 2) ? 0.0 : 0.5;
    Subcycling::CalcYfFromKcs<4>(CCTK_PASS_CTOC, u_groups, p_groups, ks_groups,
                                 dt * 2, xsi, 1);
  }
  CalcRhsAndUpdateU<nvars>(grid, k1, vlu, vlu, dt / CCTK_REAL(6.));
  CalcYs<nvars>(grid, vlw, vlp, k1, dt * CCTK_REAL(0.5));
}

extern "C" void TestSubcyclingMC2_CalcK2(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTSX_TestSubcyclingMC2_CalcK2;
  DECLARE_CCTK_PARAMETERS;
  const CCTK_REAL dt = CCTK_DELTA_TIME;
  const array<const Loop::GF3D2<CCTK_REAL>, 2> k2{u_k2, rho_k2};
  const array<const Loop::GF3D2<CCTK_REAL>, 2> vlu{u, rho};
  const array<const Loop::GF3D2<CCTK_REAL>, 2> vlw{u_w, rho_w};
  const array<const Loop::GF3D2<const CCTK_REAL>, 2> vlp{u_p, rho_p};
  constexpr size_t nvars = vlu.size();

  if (use_subcycling_wip) {
    vector<int> w_groups, p_groups;
    array<vector<int>, 4> ks_groups;
    w_groups.push_back(CCTK_GroupIndex("TestSubcyclingMC2::wstate"));
    p_groups.push_back(CCTK_GroupIndex("TestSubcyclingMC2::pstate"));
    ks_groups[0].push_back(CCTK_GroupIndex("TestSubcyclingMC2::k1"));
    ks_groups[1].push_back(CCTK_GroupIndex("TestSubcyclingMC2::k2"));
    ks_groups[2].push_back(CCTK_GroupIndex("TestSubcyclingMC2::k3"));
    ks_groups[3].push_back(CCTK_GroupIndex("TestSubcyclingMC2::k4"));
    const CCTK_REAL xsi = (cctk_iteration % 2) ? 0.0 : 0.5;
    Subcycling::CalcYfFromKcs<4>(CCTK_PASS_CTOC, w_groups, p_groups, ks_groups,
                                 dt * 2, xsi, 2);
  }
  CalcRhsAndUpdateU<nvars>(grid, k2, vlw, vlu, dt / CCTK_REAL(3.));
  CalcYs<nvars>(grid, vlw, vlp, k2, dt * CCTK_REAL(0.5));
}

extern "C" void TestSubcyclingMC2_CalcK3(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTSX_TestSubcyclingMC2_CalcK3;
  DECLARE_CCTK_PARAMETERS;
  const CCTK_REAL dt = CCTK_DELTA_TIME;
  const array<const Loop::GF3D2<CCTK_REAL>, 2> k3{u_k3, rho_k3};
  const array<const Loop::GF3D2<CCTK_REAL>, 2> vlu{u, rho};
  const array<const Loop::GF3D2<CCTK_REAL>, 2> vlw{u_w, rho_w};
  const array<const Loop::GF3D2<const CCTK_REAL>, 2> vlp{u_p, rho_p};
  constexpr size_t nvars = vlu.size();

  if (use_subcycling_wip) {
    vector<int> w_groups, p_groups;
    array<vector<int>, 4> ks_groups;
    w_groups.push_back(CCTK_GroupIndex("TestSubcyclingMC2::wstate"));
    p_groups.push_back(CCTK_GroupIndex("TestSubcyclingMC2::pstate"));
    ks_groups[0].push_back(CCTK_GroupIndex("TestSubcyclingMC2::k1"));
    ks_groups[1].push_back(CCTK_GroupIndex("TestSubcyclingMC2::k2"));
    ks_groups[2].push_back(CCTK_GroupIndex("TestSubcyclingMC2::k3"));
    ks_groups[3].push_back(CCTK_GroupIndex("TestSubcyclingMC2::k4"));
    const CCTK_REAL xsi = (cctk_iteration % 2) ? 0.0 : 0.5;
    Subcycling::CalcYfFromKcs<4>(CCTK_PASS_CTOC, w_groups, p_groups, ks_groups,
                                 dt * 2, xsi, 3);
  }
  CalcRhsAndUpdateU<nvars>(grid, k3, vlw, vlu, dt / CCTK_REAL(3.));
  CalcYs<nvars>(grid, vlw, vlp, k3, dt);
}

extern "C" void TestSubcyclingMC2_CalcK4(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTSX_TestSubcyclingMC2_CalcK4;
  DECLARE_CCTK_PARAMETERS;
  const CCTK_REAL dt = CCTK_DELTA_TIME;
  const array<const Loop::GF3D2<CCTK_REAL>, 2> k4{u_k4, rho_k4};
  const array<const Loop::GF3D2<CCTK_REAL>, 2> vlu{u, rho};
  const array<const Loop::GF3D2<CCTK_REAL>, 2> vlw{u_w, rho_w};
  constexpr size_t nvars = vlu.size();

  if (use_subcycling_wip) {
    vector<int> w_groups, p_groups;
    array<vector<int>, 4> ks_groups;
    w_groups.push_back(CCTK_GroupIndex("TestSubcyclingMC2::wstate"));
    p_groups.push_back(CCTK_GroupIndex("TestSubcyclingMC2::pstate"));
    ks_groups[0].push_back(CCTK_GroupIndex("TestSubcyclingMC2::k1"));
    ks_groups[1].push_back(CCTK_GroupIndex("TestSubcyclingMC2::k2"));
    ks_groups[2].push_back(CCTK_GroupIndex("TestSubcyclingMC2::k3"));
    ks_groups[3].push_back(CCTK_GroupIndex("TestSubcyclingMC2::k4"));
    const CCTK_REAL xsi = (cctk_iteration % 2) ? 0.0 : 0.5;
    Subcycling::CalcYfFromKcs<4>(CCTK_PASS_CTOC, w_groups, p_groups, ks_groups,
                                 dt * 2, xsi, 4);
  }
  CalcRhsAndUpdateU<nvars>(grid, k4, vlw, vlu, dt / CCTK_REAL(6.));
}

extern "C" void TestSubcyclingMC2_Sync(CCTK_ARGUMENTS) {
  // do nothing
}

extern "C" void TestSubcyclingMC2_Error(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTSX_TestSubcyclingMC2_Error;
  DECLARE_CCTK_PARAMETERS;

  if (CCTK_EQUALS(initial_condition, "standing wave")) {

    grid.loop_int_device<0, 0, 0>(
        grid.nghostzones,
        [=] CCTK_DEVICE(const Loop::PointDesc &p) CCTK_ATTRIBUTE_ALWAYS_INLINE {
          CCTK_REAL u0, rho0;
          standing_wave(amplitude, standing_wave_kx, standing_wave_ky,
                        standing_wave_kz, cctk_time, p.x, p.y, p.z, u0, rho0);
          u_err(p.I) = u(p.I) - u0;
          rho_err(p.I) = rho(p.I) - rho0;
        });

  } else if (CCTK_EQUALS(initial_condition, "Gaussian")) {

    grid.loop_int_device<0, 0, 0>(
        grid.nghostzones,
        [=] CCTK_DEVICE(const Loop::PointDesc &p) CCTK_ATTRIBUTE_ALWAYS_INLINE {
          CCTK_REAL u0, rho0;
          gaussian(amplitude, gaussian_width, cctk_time, p.x, p.y, p.z, u0,
                   rho0);
          u_err(p.I) = u(p.I) - u0;
          rho_err(p.I) = rho(p.I) - rho0;
        });

  } else {
    CCTK_ERROR("Unknown initial condition");
  }
}

} // namespace TestSubcyclingMC2
