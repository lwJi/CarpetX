#include <cctk.h>
#include <cctk_Parameters.h>
#include <cctk_Arguments.h>

#include <algorithm>
#include <cmath>

namespace TestODESolvers {

extern "C" void TestODESolvers_Initial(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTS_TestODESolvers_Initial;
  DECLARE_CCTK_PARAMETERS;

  using std::pow;
  const CCTK_REAL u0 = pow(1 + cctk_time, order);

  const int imin = cctk_tile_min[0];
  const int jmin = cctk_tile_min[1];
  const int kmin = cctk_tile_min[2];

  using std::min;
  const int imax = min(cctk_lsh[0] - 1, cctk_tile_max[0]);
  const int jmax = min(cctk_lsh[1] - 1, cctk_tile_max[1]);
  const int kmax = min(cctk_lsh[2] - 1, cctk_tile_max[2]);

  const int di = 1;
  const int dj = di * (cctk_ash[0] - 1);
  const int dk = dj * (cctk_ash[1] - 1);

  for (int k = kmin; k < kmax; ++k) {
    for (int j = jmin; j < jmax; ++j) {
      for (int i = imin; i < imax; ++i) {
        int ind = i * di + j * dj + k * dk;
        state[ind] = u0;
        state2[ind] = u0;
      }
    }
  }
}

extern "C" void TestODESolvers_RHS(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTS_TestODESolvers_RHS;
  DECLARE_CCTK_PARAMETERS;

  static int last_it = -1;
  static CCTK_REAL original_time;
#pragma omp critical
  if (last_it != cctk_iteration) {
    last_it = cctk_iteration;
    original_time = cctk_time;
  }
  const CCTK_REAL cctk_time2 = original_time + 2 * (cctk_time - original_time);

  const int imin = cctk_tile_min[0];
  const int jmin = cctk_tile_min[1];
  const int kmin = cctk_tile_min[2];

  using std::min;
  const int imax = min(cctk_lsh[0] - 1, cctk_tile_max[0]);
  const int jmax = min(cctk_lsh[1] - 1, cctk_tile_max[1]);
  const int kmax = min(cctk_lsh[2] - 1, cctk_tile_max[2]);

  const int di = 1;
  const int dj = di * (cctk_ash[0] - 1);
  const int dk = dj * (cctk_ash[1] - 1);

  for (int k = kmin; k < kmax; ++k) {
    for (int j = jmin; j < jmax; ++j) {
      for (int i = imin; i < imax; ++i) {
        int ind = i * di + j * dj + k * dk;
        const CCTK_REAL u_inverse =
            pow(state[ind], (order - 1) / CCTK_REAL(order));
        rhs[ind] = 0.5 * order * (u_inverse + pow(1 + cctk_time, order - 1));

        if (cctk_iteration % 2) {
          const CCTK_REAL u_inverse2 =
              pow(state2[ind], (order - 1) / CCTK_REAL(order));
          rhs2[ind] =
              2 * 0.5 * order * (u_inverse2 + pow(1 + cctk_time2, order - 1));
        } else {
          rhs2[ind] = 0;
        }
      }
    }
  }
}

extern "C" void TestODESolvers_PostStep(CCTK_ARGUMENTS) {}

extern "C" void TestODESolvers_Error(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTS_TestODESolvers_Error;
  DECLARE_CCTK_PARAMETERS;

  using std::pow;
  const CCTK_REAL u0 = pow(1 + cctk_time, order);

  const int imin = cctk_tile_min[0];
  const int jmin = cctk_tile_min[1];
  const int kmin = cctk_tile_min[2];

  using std::min;
  const int imax = min(cctk_lsh[0] - 1, cctk_tile_max[0]);
  const int jmax = min(cctk_lsh[1] - 1, cctk_tile_max[1]);
  const int kmax = min(cctk_lsh[2] - 1, cctk_tile_max[2]);

  const int di = 1;
  const int dj = di * (cctk_ash[0] - 1);
  const int dk = dj * (cctk_ash[1] - 1);

  for (int k = kmin; k < kmax; ++k) {
    for (int j = jmin; j < jmax; ++j) {
      for (int i = imin; i < imax; ++i) {
        int ind = i * di + j * dj + k * dk;
        error[ind] = state[ind] - u0;
        error2[ind] = state2[ind] - u0;

        if (error[ind] == 0 || error2[ind] == 0) {
          corder[ind] = 0;
        } else {
          using std::abs, std::log;
          corder[ind] = log(abs(error2[ind] / error[ind])) / log(2.0);
        }
      }
    }
  }
}

} // namespace TestODESolvers
