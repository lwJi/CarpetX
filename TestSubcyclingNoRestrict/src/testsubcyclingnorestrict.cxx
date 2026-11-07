#include <loop_device.hxx>

#include <derivs.hxx>
#include <mat.hxx>
#include <vec.hxx>
#include <vect.hxx>

#include <cctk.h>
#include <cctk_Arguments.h>
#include <cctk_Parameters.h>
#include <util_Table.h>

#include <array>
#include <cmath>
#include <limits>

namespace TestSubcyclingNoRestrict {
using namespace Arith;
using namespace Loop;
using std::array;
using std::result_of_t;

constexpr int dim = 3;

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

extern "C" void TestSubcyclingNoRestrict_Initial(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTSX_TestSubcyclingNoRestrict_Initial;
  DECLARE_CCTK_PARAMETERS;

  // Both state vectors start from the same data
  grid.loop_int_device<0, 0, 0>(
      grid.nghostzones,
      [=] CCTK_DEVICE(const PointDesc &p) CCTK_ATTRIBUTE_ALWAYS_INLINE {
        CCTK_REAL u0, rho0;
        gaussian(amplitude, gaussian_width, cctk_time, p.x, p.y, p.z, u0, rho0);
        u(p.I) = u0;
        rho(p.I) = rho0;
        u_nr(p.I) = u0;
        rho_nr(p.I) = rho0;
      });
}

extern "C" void TestSubcyclingNoRestrict_Sync(CCTK_ARGUMENTS) {
  // do nothing
}

extern "C" void TestSubcyclingNoRestrict_Floor(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTSX_TestSubcyclingNoRestrict_Floor;
  DECLARE_CCTK_PARAMETERS;

  // Idempotent: max(max(u, c), c) == max(u, c)
  grid.loop_int_device<0, 0, 0>(grid.nghostzones,
                                [=] CCTK_DEVICE(const PointDesc &p)
                                    CCTK_ATTRIBUTE_ALWAYS_INLINE {
                                      using std::fmax;
                                      u_nr(p.I) = fmax(u_nr(p.I), u_floor);
                                    });
}

extern "C" void TestSubcyclingNoRestrict_RHS(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTSX_TestSubcyclingNoRestrict_RHS;
  DECLARE_CCTK_PARAMETERS;

  const vect<CCTK_REAL, dim> dx{
      CCTK_DELTA_SPACE(0),
      CCTK_DELTA_SPACE(1),
      CCTK_DELTA_SPACE(2),
  };

  const array<int, dim> indextype = {0, 0, 0};
  const array<int, dim> nghostzones = {cctk_nghostzones[0], cctk_nghostzones[1],
                                       cctk_nghostzones[2]};
  vect<int, dim> imin, imax;
  GridDescBase(cctkGH).box_int<0, 0, 0>(nghostzones, imin, imax);
  // suffix 5: without ghost zones
  const GF3D5layout layout5(imin, imax);

  // The scalar wave equation u_t = rho, rho_t = Laplacian u for one state
  // vector, with a fourth-order centred Laplacian
  const auto calc_rhs = [&](const GF3D2<const CCTK_REAL> &gf_u,
                            const GF3D2<const CCTK_REAL> &gf_rho,
                            const GF3D2<CCTK_REAL> &gf_u_rhs,
                            const GF3D2<CCTK_REAL> &gf_rho_rhs) {
    // Tile variables for derivatives and so on
    const int ntmps = 10;
    GF3D5vector<CCTK_REAL> tmps(layout5, ntmps);
    int itmp = 0;

    const auto make_gf = [&]() { return GF3D5<CCTK_REAL>(tmps(itmp++)); };
    const auto make_vec = [&](const auto &f) {
      return vec<result_of_t<decltype(f)()>, 3>([&](int) { return f(); });
    };
    const auto make_mat = [&](const auto &f) {
      return smat<result_of_t<decltype(f)()>, 3>([&](int, int) { return f(); });
    };
    const auto make_vec_gf = [&]() { return make_vec(make_gf); };
    const auto make_mat_gf = [&]() { return make_mat(make_gf); };

    const GF3D5<CCTK_REAL> tl_u(make_gf());
    const vec<GF3D5<CCTK_REAL>, 3> tl_du(make_vec_gf());
    const smat<GF3D5<CCTK_REAL>, 3> tl_ddu(make_mat_gf());

    constexpr int deriv_order = 4;
    Derivs::calc_derivs2<0, 0, 0>(tl_u, tl_du, tl_ddu, layout5, grid, gf_u, dx,
                                  deriv_order);

    if (itmp != ntmps)
      CCTK_VERROR("Wrong number of temporary variables: ntmps=%d itmp=%d",
                  ntmps, itmp);

    grid.loop_int_device<0, 0, 0>(
        grid.nghostzones,
        [=] CCTK_DEVICE(const PointDesc &p) CCTK_ATTRIBUTE_ALWAYS_INLINE {
          const GF3D5index index5(layout5, p.I);
          gf_u_rhs(p.I) = gf_rho(p.I);
          gf_rho_rhs(p.I) = tl_ddu(0, 0)(index5) + tl_ddu(1, 1)(index5) +
                            tl_ddu(2, 2)(index5);
        });
  };

  calc_rhs(u, rho, u_rhs, rho_rhs);
  calc_rhs(u_nr, rho_nr, u_nr_rhs, rho_nr_rhs);
}

extern "C" void TestSubcyclingNoRestrict_Error(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTSX_TestSubcyclingNoRestrict_Error;
  DECLARE_CCTK_PARAMETERS;

  // Pointwise, so computed everywhere instead of SYNCed (see schedule.ccl)
  grid.loop_all_device<0, 0, 0>(
      grid.nghostzones,
      [=] CCTK_DEVICE(const PointDesc &p) CCTK_ATTRIBUTE_ALWAYS_INLINE {
        CCTK_REAL u0, rho0;
        gaussian(amplitude, gaussian_width, cctk_time, p.x, p.y, p.z, u0, rho0);
        u_err(p.I) = u(p.I) - u0;
        rho_err(p.I) = rho(p.I) - rho0;
      });
}

extern "C" void TestSubcyclingNoRestrict_InterpolateOnRecover(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTS_TestSubcyclingNoRestrict_InterpolateOnRecover;

  constexpr int npoints = 2;
  const array<CCTK_REAL, npoints> x{0.0, 3.0};
  const array<CCTK_REAL, npoints> y{0.0, 3.0};
  const array<CCTK_REAL, npoints> z{0.0, 3.0};
  const void *const coords[dim] = {x.data(), y.data(), z.data()};

  const CCTK_INT varinds[1] = {CCTK_VarIndex("TestSubcyclingNoRestrict::u")};
  const CCTK_INT operand_indices[1] = {0};
  const CCTK_INT operation_codes[1] = {0};

  const int interp_handle = CCTK_InterpHandle("CarpetX");
  if (interp_handle < 0)
    CCTK_VERROR("Could not obtain the 'CarpetX' interpolator handle: %d",
                interp_handle);

  const int table = Util_TableCreate(UTIL_TABLE_FLAGS_DEFAULT);
  if (table < 0)
    CCTK_VERROR("Could not create the parameter table: %d", table);
  int ierr = Util_TableSetInt(table, 1, "order");
  if (ierr < 0)
    CCTK_VERROR("Could not set order in the parameter table: %d", ierr);
  ierr = Util_TableSetIntArray(table, 1, operand_indices, "operand_indices");
  if (ierr < 0)
    CCTK_VERROR("Could not set operand_indices in the parameter table: %d",
                ierr);
  ierr = Util_TableSetIntArray(table, 1, operation_codes, "operation_codes");
  if (ierr < 0)
    CCTK_VERROR("Could not set operation_codes in the parameter table: %d",
                ierr);

  array<CCTK_REAL, npoints> result;
  const CCTK_INT output_array_type_codes[1] = {CCTK_VARIABLE_REAL};
  void *const output_arrays[1] = {result.data()};

  ierr = CCTK_InterpGridArrays(cctkGH, dim, interp_handle, table, 0, npoints,
                               CCTK_VARIABLE_REAL, coords, 1, varinds, 1,
                               output_array_type_codes, output_arrays);
  if (ierr < 0)
    CCTK_VERROR("CCTK_InterpGridArrays failed: %d", ierr);

  Util_TableDestroy(table);
}

} // namespace TestSubcyclingNoRestrict
