#include <loop_device.hxx>

#include <vect.hxx>

#include <cctk.h>
#include <cctk_Arguments.h>
#include <cctk_Parameters.h>

#include <cmath>
#include <limits>

namespace TestSubcyclingCC {
using namespace Arith;
using namespace Loop;

constexpr int dim = 3;

// u(t,r) = (f(t-r) - f(t+r)) / r
// f(v) = A exp(-1/2 (v/W)^2)
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

// Fourth-order centred second derivative along direction d (five-point
// stencil, reaches two cells to either side)
template <typename T>
CCTK_DEVICE CCTK_HOST CCTK_ATTRIBUTE_ALWAYS_INLINE inline T
deriv2(const GF3D2<const T> &gf, const PointDesc &p, const int d) {
  const auto DI = p.DI[d];
  return (-gf(p.I - 2 * DI) + 16 * gf(p.I - DI) - 30 * gf(p.I) +
          16 * gf(p.I + DI) - gf(p.I + 2 * DI)) /
         (12 * p.DX[d] * p.DX[d]);
}

// Kreiss-Oliger dissipation for a fourth-order scheme (sixth-order
// dissipation operator, seven-point stencil per direction, reaches three
// cells to either side). Same normalisation as Derivs::calc_diss<4>:
// (-1)^(diss_order/2+1) / 2^(diss_order) * sum_d D_+^3 D_-^3 u / dx_d,
// with diss_order = 6, i.e. +1/64.
template <typename T>
CCTK_DEVICE CCTK_HOST CCTK_ATTRIBUTE_ALWAYS_INLINE inline T
diss(const GF3D2<const T> &gf, const PointDesc &p) {
  T result = 0;
  for (int d = 0; d < dim; ++d) {
    const auto DI = p.DI[d];
    result += (gf(p.I - 3 * DI) + gf(p.I + 3 * DI) -
               6 * (gf(p.I - 2 * DI) + gf(p.I + 2 * DI)) +
               15 * (gf(p.I - DI) + gf(p.I + DI)) - 20 * gf(p.I)) /
              p.DX[d];
  }
  return result / 64;
}

extern "C" void TestSubcyclingCC_Initial(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTSX_TestSubcyclingCC_Initial;
  DECLARE_CCTK_PARAMETERS;

  grid.loop_int_device<1, 1, 1>(
      grid.nghostzones,
      [=] CCTK_DEVICE(const PointDesc &p) CCTK_ATTRIBUTE_ALWAYS_INLINE {
        gaussian(amplitude, gaussian_width, cctk_time, p.x, p.y, p.z, u(p.I),
                 rho(p.I));
      });
}

extern "C" void TestSubcyclingCC_RHS(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTSX_TestSubcyclingCC_RHS;
  DECLARE_CCTK_PARAMETERS;

  grid.loop_int_device<1, 1, 1>(
      grid.nghostzones,
      [=] CCTK_DEVICE(const PointDesc &p) CCTK_ATTRIBUTE_ALWAYS_INLINE {
        u_rhs(p.I) = rho(p.I) + epsdiss * diss(u, p);
        rho_rhs(p.I) = deriv2(u, p, 0) + deriv2(u, p, 1) + deriv2(u, p, 2) +
                       epsdiss * diss(rho, p);
      });
}

extern "C" void TestSubcyclingCC_Sync(CCTK_ARGUMENTS) {
  // do nothing
}

extern "C" void TestSubcyclingCC_Error(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTSX_TestSubcyclingCC_Error;
  DECLARE_CCTK_PARAMETERS;

  grid.loop_int_device<1, 1, 1>(
      grid.nghostzones,
      [=] CCTK_DEVICE(const PointDesc &p) CCTK_ATTRIBUTE_ALWAYS_INLINE {
        CCTK_REAL u0, rho0;
        gaussian(amplitude, gaussian_width, cctk_time, p.x, p.y, p.z, u0,
                 rho0);
        u_err(p.I) = u(p.I) - u0;
        rho_err(p.I) = rho(p.I) - rho0;
      });
}

} // namespace TestSubcyclingCC
