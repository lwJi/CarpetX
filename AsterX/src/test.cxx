#include <fixmath.hxx>
#include <loop_device.hxx>

#include <cctk.h>
#include <cctk_Arguments.h>

#include "utils.hxx"

namespace AsterX {

using namespace Arith;

extern "C" void AsterX_Test(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTS;

  const smat<CCTK_REAL, 3> g{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  const CCTK_REAL detg = -1.0;
  const smat<CCTK_REAL, 3> invg{1.0, -3.0, 2.0, 3.0, -1.0, 0.0};
  const vec<CCTK_REAL, 3> v_up{0.07, 0.08, 0.09};
  const CCTK_REAL tiny = 1e-14;

  {
    assert(calc_det(g) == detg);
    CCTK_VINFO("Test calc_det succeeded");
  }

  {
    const smat<CCTK_REAL, 3> invg_test = calc_inv(g, detg);
    assert(invg_test(0, 0) == 1.0);
    assert(invg_test(0, 1) == -3.0);
    assert(invg_test(0, 2) == 2.0);
    assert(invg_test(1, 1) == 3.0);
    assert(invg_test(1, 2) == -1.0);
    assert(invg_test(2, 2) == 0.0);
    assert(invg_test.elts[0] == 1.0);
    assert(invg_test.elts[1] == -3.0);
    assert(invg_test.elts[2] == 2.0);
    assert(invg_test.elts[3] == 3.0);
    assert(invg_test.elts[4] == -1.0);
    assert(invg_test.elts[5] == 0.0);
    assert(invg_test == invg);
    CCTK_VINFO("Test calc_inv succeeded");
  }

  {
    // const vec<CCTK_REAL, 3> v_dn([&](int i) ARITH_INLINE {
    //   return sum<3>([&](int j) ARITH_INLINE { return g(i, j) * v_up(j); });
    // });
    const vec<CCTK_REAL, 3> v_dn = calc_contraction(g, v_up);
    assert(v_dn(0) - 0.5 < tiny);
    assert(v_dn(1) - 0.91 < tiny);
    assert(v_dn(2) - 1.15 < tiny);
    CCTK_VINFO("Test calc_contraction of smat and vec succeeded");

    // const CCTK_REAL v2(
    //     sum<3>([&](int i) ARITH_INLINE { return v_up(i) * v_dn(i); }));
    const CCTK_REAL v2 = calc_contraction(v_up, v_dn);
    assert(v2 - 0.2113 < tiny);
    CCTK_VINFO("Test calc_contraction of vec and vec succeeded");

    const CCTK_REAL wlorentz = calc_wlorentz(v_up, v_dn);
    assert(wlorentz - 1.0 / sqrt(1.0 - v2) < tiny);
    CCTK_VINFO("Test calc_wlorentz of vec and vec succeeded");
  }
}

} // namespace AsterX
