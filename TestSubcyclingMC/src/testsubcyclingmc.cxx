#include <loop_device.hxx>

#include <sum.hxx>
#include <vect.hxx>

#include <cctk.h>
#include <cctk_Arguments.h>
#include <cctk_Parameters.h>

#include <array>
#include <cassert>
#include <cmath>
#include <limits>

namespace TestSubcyclingMC {
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

extern "C" void TestSubcyclingMC_Initial(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTSX_TestSubcyclingMC_Initial;
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

/**
 * \brief Calculate Ys ghost points for fine grid using Ks on coarse grid
 *
 * \param Yf        RK substage Ys on the fine side to be interperated into
 *                  the ghost zones
 * \param kcs       RK ks on the coarset side
 * \param u0        u at t0
 * \param dtc       Time step size on coarse side
 * \param xsi       which substep on fine level during a coarse time
 *                  step.  For an AMR simulation with subcycling and a
 *                  refinement ratio of 2, the number is either 0 or 0.5,
 *                  denoting the first and second substep, respectively.
 * \param stage     RK stage number starting from 1
 */
template <typename tVarOut, typename tVarIn>
CCTK_DEVICE CCTK_HOST CCTK_ATTRIBUTE_ALWAYS_INLINE inline void
CalcYfFromKcs(const Loop::GridDescBaseDevice &grid, tVarOut &Yf, tVarIn &u0,
              const array<tVarOut, 4> &kcs, tVarIn &isrmbndry,
              const CCTK_REAL dtc, const CCTK_REAL xsi, const CCTK_INT stage) {
  assert(stage > 0 && stage <= 4);

  CCTK_REAL r = 0.5; // ratio between coarse and fine cell size (2 to 1 MR case)
  CCTK_REAL xsi2 = xsi * xsi;
  CCTK_REAL xsi3 = xsi2 * xsi;
  // coefficients for U
  CCTK_REAL b1 = xsi - CCTK_REAL(1.5) * xsi2 + CCTK_REAL(2. / 3.) * xsi3;
  CCTK_REAL b2 = xsi2 - CCTK_REAL(2. / 3.) * xsi3;
  CCTK_REAL b3 = b2;
  CCTK_REAL b4 = CCTK_REAL(-0.5) * xsi2 + CCTK_REAL(2. / 3.) * xsi3;
  // coefficients for Ut
  CCTK_REAL c1 = CCTK_REAL(1.) - CCTK_REAL(3.) * xsi + CCTK_REAL(2.) * xsi2;
  CCTK_REAL c2 = CCTK_REAL(2.) * xsi - CCTK_REAL(2.) * xsi2;
  CCTK_REAL c3 = c2;
  CCTK_REAL c4 = -xsi + CCTK_REAL(2.) * xsi2;
  // coefficients for Utt
  CCTK_REAL d1 = CCTK_REAL(-3.) + CCTK_REAL(4.) * xsi;
  CCTK_REAL d2 = CCTK_REAL(2.) - CCTK_REAL(4.) * xsi;
  CCTK_REAL d3 = d2;
  CCTK_REAL d4 = CCTK_REAL(-1.) + CCTK_REAL(4.) * xsi;
  // coefficients for Uttt
  constexpr CCTK_REAL e1 = CCTK_REAL(4.);
  constexpr CCTK_REAL e2 = CCTK_REAL(-4.);
  constexpr CCTK_REAL e3 = CCTK_REAL(-4.);
  constexpr CCTK_REAL e4 = CCTK_REAL(4.);

  if (stage == 1) {
    grid.loop_ghosts_device<0, 0, 0>(
        grid.nghostzones,
        [=] CCTK_DEVICE(const Loop::PointDesc &p) CCTK_ATTRIBUTE_ALWAYS_INLINE {
          if (isrmbndry(p.I)) {
            CCTK_REAL k1 = kcs[0](p.I);
            CCTK_REAL k2 = kcs[1](p.I);
            CCTK_REAL k3 = kcs[2](p.I);
            CCTK_REAL k4 = kcs[3](p.I);
            CCTK_REAL uu = b1 * k1 + b2 * k2 + b3 * k3 + b4 * k4;
            Yf(p.I) = u0(p.I) + dtc * uu;
          }
        });
  } else if (stage == 2) {
    grid.loop_ghosts_device<0, 0, 0>(
        grid.nghostzones,
        [=] CCTK_DEVICE(const Loop::PointDesc &p) CCTK_ATTRIBUTE_ALWAYS_INLINE {
          if (isrmbndry(p.I)) {
            CCTK_REAL k1 = kcs[0](p.I);
            CCTK_REAL k2 = kcs[1](p.I);
            CCTK_REAL k3 = kcs[2](p.I);
            CCTK_REAL k4 = kcs[3](p.I);
            CCTK_REAL uu = b1 * k1 + b2 * k2 + b3 * k3 + b4 * k4;
            CCTK_REAL ut = c1 * k1 + c2 * k2 + c3 * k3 + c4 * k4;
            Yf(p.I) = u0(p.I) + dtc * (uu + CCTK_REAL(0.5) * r * ut);
          }
        });
  } else if (stage == 3 || stage == 4) {
    CCTK_REAL r2 = r * r;
    CCTK_REAL r3 = r2 * r;
    CCTK_REAL at = (stage == 3) ? CCTK_REAL(0.5) * r : r;
    CCTK_REAL att = (stage == 3) ? CCTK_REAL(0.25) * r2 : CCTK_REAL(0.5) * r2;
    CCTK_REAL attt =
        (stage == 3) ? CCTK_REAL(0.0625) * r3 : CCTK_REAL(0.125) * r3;
    CCTK_REAL ak = (stage == 3) ? CCTK_REAL(-4.) : CCTK_REAL(4.);

    grid.loop_ghosts_device<0, 0, 0>(
        grid.nghostzones,
        [=] CCTK_DEVICE(const Loop::PointDesc &p) CCTK_ATTRIBUTE_ALWAYS_INLINE {
          if (isrmbndry(p.I)) {
            CCTK_REAL k1 = kcs[0](p.I);
            CCTK_REAL k2 = kcs[1](p.I);
            CCTK_REAL k3 = kcs[2](p.I);
            CCTK_REAL k4 = kcs[3](p.I);
            CCTK_REAL uu = b1 * k1 + b2 * k2 + b3 * k3 + b4 * k4;
            CCTK_REAL ut = c1 * k1 + c2 * k2 + c3 * k3 + c4 * k4;
            CCTK_REAL utt = d1 * k1 + d2 * k2 + d3 * k3 + d4 * k4;
            CCTK_REAL uttt = e1 * k1 + e2 * k2 + e3 * k3 + e4 * k4;
            Yf(p.I) = u0(p.I) + dtc * (uu + at * ut + att * utt +
                                       attt * (uttt + ak * (k3 - k2)));
          }
        });
  }
}

/* Varlist version */
template <int D, typename tVarOut, typename tVarIn>
CCTK_DEVICE CCTK_HOST CCTK_ATTRIBUTE_ALWAYS_INLINE inline void
CalcYfFromKcs(const Loop::GridDescBaseDevice &grid,
              const array<tVarOut, D> &Yfs, const array<tVarIn, D> &u0s,
              const array<const array<tVarOut, 4>, D> &kcss, tVarIn &isrmbndry,
              const CCTK_REAL dtc, const CCTK_REAL xsi, const CCTK_INT stage) {
  for (size_t v = 0; v < D; ++v) {
    CalcYfFromKcs<tVarOut, tVarIn>(grid, Yfs[v], u0s[v], kcss[v], isrmbndry,
                                   dtc, xsi, stage);
  }
}

extern "C" void TestSubcyclingMC_SetP(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTSX_TestSubcyclingMC_SetP;

  CCTK_VINFO("Updating grid function at iteration %d level %d time %g",
             cctk_iteration, cctk_level, cctk_time);
  grid.loop_int_device<0, 0, 0>(grid.nghostzones,
                                [=] CCTK_DEVICE(const Loop::PointDesc &p)
                                    CCTK_ATTRIBUTE_ALWAYS_INLINE {
                                      u_p(p.I) = u(p.I);
                                      rho_p(p.I) = rho(p.I);
                                    });
}

extern "C" void TestSubcyclingMC_CalcK1(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTSX_TestSubcyclingMC_CalcK1;
  DECLARE_CCTK_PARAMETERS;
  const CCTK_REAL dt = CCTK_DELTA_TIME;
  const array<const Loop::GF3D2<CCTK_REAL>, 2> k1{u_k1, rho_k1};
  const array<const Loop::GF3D2<CCTK_REAL>, 2> vlu{u, rho};
  const array<const Loop::GF3D2<CCTK_REAL>, 2> vlw{u_w, rho_w};
  const array<const Loop::GF3D2<const CCTK_REAL>, 2> vlp{u_p, rho_p};
  constexpr size_t nvars = vlu.size();

  if (use_subcycling_wip) {
    const array<const array<const Loop::GF3D2<CCTK_REAL>, 4>, 2> kcss{
        {{u_k1, u_k2, u_k3, u_k4}, {rho_k1, rho_k2, rho_k3, rho_k4}}};
    const CCTK_REAL xsi = (cctk_iteration % 2) ? 0.0 : 0.5;

    CalcYfFromKcs<nvars>(grid, vlu, vlp, kcss, isrmbndry, dt * 2, xsi, 1);
  }
  CalcRhsAndUpdateU<nvars>(grid, k1, vlu, vlu, dt / CCTK_REAL(6.));
  CalcYs<nvars>(grid, vlw, vlp, k1, dt * CCTK_REAL(0.5));
}

extern "C" void TestSubcyclingMC_CalcK2(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTSX_TestSubcyclingMC_CalcK2;
  DECLARE_CCTK_PARAMETERS;
  const CCTK_REAL dt = CCTK_DELTA_TIME;
  const array<const Loop::GF3D2<CCTK_REAL>, 2> k2{u_k2, rho_k2};
  const array<const Loop::GF3D2<CCTK_REAL>, 2> vlu{u, rho};
  const array<const Loop::GF3D2<CCTK_REAL>, 2> vlw{u_w, rho_w};
  const array<const Loop::GF3D2<const CCTK_REAL>, 2> vlp{u_p, rho_p};
  constexpr size_t nvars = vlu.size();

  if (use_subcycling_wip) {
    const array<const array<const Loop::GF3D2<CCTK_REAL>, 4>, 2> kcss{
        {{u_k1, u_k2, u_k3, u_k4}, {rho_k1, rho_k2, rho_k3, rho_k4}}};
    const CCTK_REAL xsi = (cctk_iteration % 2) ? 0.0 : 0.5;

    CalcYfFromKcs<nvars>(grid, vlw, vlp, kcss, isrmbndry, dt * 2, xsi, 2);
  }
  CalcRhsAndUpdateU<nvars>(grid, k2, vlw, vlu, dt / CCTK_REAL(3.));
  CalcYs<nvars>(grid, vlw, vlp, k2, dt * CCTK_REAL(0.5));
}

extern "C" void TestSubcyclingMC_CalcK3(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTSX_TestSubcyclingMC_CalcK3;
  DECLARE_CCTK_PARAMETERS;
  const CCTK_REAL dt = CCTK_DELTA_TIME;
  const array<const Loop::GF3D2<CCTK_REAL>, 2> k3{u_k3, rho_k3};
  const array<const Loop::GF3D2<CCTK_REAL>, 2> vlu{u, rho};
  const array<const Loop::GF3D2<CCTK_REAL>, 2> vlw{u_w, rho_w};
  const array<const Loop::GF3D2<const CCTK_REAL>, 2> vlp{u_p, rho_p};
  constexpr size_t nvars = vlu.size();

  if (use_subcycling_wip) {
    const array<const array<const Loop::GF3D2<CCTK_REAL>, 4>, 2> kcss{
        {{u_k1, u_k2, u_k3, u_k4}, {rho_k1, rho_k2, rho_k3, rho_k4}}};
    const CCTK_REAL xsi = (cctk_iteration % 2) ? 0.0 : 0.5;

    CalcYfFromKcs<nvars>(grid, vlw, vlp, kcss, isrmbndry, dt * 2, xsi, 3);
  }
  CalcRhsAndUpdateU<nvars>(grid, k3, vlw, vlu, dt / CCTK_REAL(3.));
  CalcYs<nvars>(grid, vlw, vlp, k3, dt);
}

extern "C" void TestSubcyclingMC_CalcK4(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTSX_TestSubcyclingMC_CalcK4;
  DECLARE_CCTK_PARAMETERS;
  const CCTK_REAL dt = CCTK_DELTA_TIME;
  const array<const Loop::GF3D2<CCTK_REAL>, 2> k4{u_k4, rho_k4};
  const array<const Loop::GF3D2<CCTK_REAL>, 2> vlu{u, rho};
  const array<const Loop::GF3D2<CCTK_REAL>, 2> vlw{u_w, rho_w};
  constexpr size_t nvars = vlu.size();

  if (use_subcycling_wip) {
    const array<const array<const Loop::GF3D2<CCTK_REAL>, 4>, 2> kcss{
        {{u_k1, u_k2, u_k3, u_k4}, {rho_k1, rho_k2, rho_k3, rho_k4}}};
    const array<const Loop::GF3D2<const CCTK_REAL>, 2> vlp{u_p, rho_p};
    const CCTK_REAL xsi = (cctk_iteration % 2) ? 0.0 : 0.5;

    CalcYfFromKcs<nvars>(grid, vlw, vlp, kcss, isrmbndry, dt * 2, xsi, 4);
  }
  CalcRhsAndUpdateU<nvars>(grid, k4, vlw, vlu, dt / CCTK_REAL(6.));
}

extern "C" void TestSubcyclingMC_Sync(CCTK_ARGUMENTS) {
  // do nothing
}

extern "C" void TestSubcyclingMC_Error(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTSX_TestSubcyclingMC_Error;
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

} // namespace TestSubcyclingMC
