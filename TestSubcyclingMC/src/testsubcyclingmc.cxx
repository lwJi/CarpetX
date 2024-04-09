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
}

void calcRHS(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTSX_TestSubcyclingMC_CalcY2;
  grid.loop_int_device<0, 0, 0>(
      grid.nghostzones,
      [=] CCTK_DEVICE(const Loop::PointDesc &p) CCTK_ATTRIBUTE_ALWAYS_INLINE {
        using std::pow;
        Arith::vect<CCTK_REAL, dim> ddu;
        for (int d = 0; d < dim; ++d) {
          ddu[d] = (u_w(p.I - p.DI[d]) - 2 * u_w(p.I) + u_w(p.I + p.DI[d])) /
                   pow(p.DX[d], 2);
        }
        u_rhs(p.I) = rho_w(p.I);
        rho_rhs(p.I) = ddu[0] + ddu[1] + ddu[2];
      });
}

void updateU(CCTK_ARGUMENTS, CCTK_REAL dt) {
  DECLARE_CCTK_ARGUMENTSX_TestSubcyclingMC_CalcY2;
  grid.loop_int_device<0, 0, 0>(grid.nghostzones,
                                [=] CCTK_DEVICE(const Loop::PointDesc &p)
                                    CCTK_ATTRIBUTE_ALWAYS_INLINE {
                                      u(p.I) += u_rhs(p.I) * dt;
                                      rho(p.I) += rho_rhs(p.I) * dt;
                                    });
}

void calcYs(CCTK_ARGUMENTS, CCTK_REAL dt) {
  DECLARE_CCTK_ARGUMENTSX_TestSubcyclingMC_CalcY2;
  grid.loop_int_device<0, 0, 0>(
      grid.nghostzones,
      [=] CCTK_DEVICE(const Loop::PointDesc &p) CCTK_ATTRIBUTE_ALWAYS_INLINE {
        u_w(p.I) = u_p(p.I) + u_rhs(p.I) * dt;
        rho_w(p.I) = rho_p(p.I) + rho_rhs(p.I) * dt;
      });
}

extern "C" void TestSubcyclingMC_CalcY1(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTSX_TestSubcyclingMC_CalcY1;
  grid.loop_int_device<0, 0, 0>(grid.nghostzones,
                                [=] CCTK_DEVICE(const Loop::PointDesc &p)
                                    CCTK_ATTRIBUTE_ALWAYS_INLINE {
                                      u_p(p.I) = u(p.I);
                                      rho_p(p.I) = rho(p.I);
                                      u_w(p.I) = u(p.I);
                                      rho_w(p.I) = rho(p.I);
                                    });
}

extern "C" void TestSubcyclingMC_CalcY2(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTSX_TestSubcyclingMC_CalcY2;
  CCTK_REAL dt = CCTK_DELTA_TIME;
  calcRHS(CCTK_PASS_CTOC); // k1
  updateU(CCTK_PASS_CTOC, dt / CCTK_REAL(6.));
  calcYs(CCTK_PASS_CTOC, dt * CCTK_REAL(0.5)); // Y2
}

extern "C" void TestSubcyclingMC_CalcY3(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTSX_TestSubcyclingMC_CalcY3;
  CCTK_REAL dt = CCTK_DELTA_TIME;
  calcRHS(CCTK_PASS_CTOC); // k2
  updateU(CCTK_PASS_CTOC, dt / CCTK_REAL(3.));
  calcYs(CCTK_PASS_CTOC, dt * CCTK_REAL(0.5)); // Y3
}

extern "C" void TestSubcyclingMC_CalcY4(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTSX_TestSubcyclingMC_CalcY4;
  CCTK_REAL dt = CCTK_DELTA_TIME;
  calcRHS(CCTK_PASS_CTOC); // k3
  updateU(CCTK_PASS_CTOC, dt / CCTK_REAL(3.));
  calcYs(CCTK_PASS_CTOC, dt); // Y4
}

extern "C" void TestSubcyclingMC_UpdateU(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTSX_TestSubcyclingMC_UpdateU;
  CCTK_REAL dt = CCTK_DELTA_TIME;
  calcRHS(CCTK_PASS_CTOC); // k4
  updateU(CCTK_PASS_CTOC, dt / CCTK_REAL(6.));
}

} // namespace TestSubcyclingMC