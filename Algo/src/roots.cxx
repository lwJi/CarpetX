#include "roots.hxx"

#include <mat.hxx>
#include <vec.hxx>

#include <cctk.h>
#include <cctk_Arguments.h>

#include <AMReX_Gpu.H>

#include <cassert>
#include <cmath>
#include <limits>
#include <tuple>

namespace Algo {

using Arith::mat, Arith::vec;

namespace {
template <typename T> ALGO_HOST ALGO_DEVICE vec<T, 2> gn(vec<T, 2> x) {
  return vec<T, 2>{x(0) * x(0) - 2, x(0) * x(1) - 2};
}
template <typename T> ALGO_HOST ALGO_DEVICE vec<T, 3> gn3(vec<T, 3> x) {
  return vec<T, 3>{x(0) * x(0) - 2, x(0) * x(1) - 2, x(1) * x(2) - 2};
}
} // namespace

extern "C" void Test_roots(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTSX_Test_roots;

  auto fn = [] ALGO_HOST ALGO_DEVICE(CCTK_REAL x) { return x * x - 2; };
  auto fnd = [] ALGO_HOST ALGO_DEVICE(CCTK_REAL x) {
    return std::make_tuple(x * x - 2, 2 * x);
  };
  auto fnd2 = [] ALGO_HOST ALGO_DEVICE(CCTK_REAL x) {
    return std::make_tuple(x * x - 2, 2 * x, 2);
  };

  {
    const int minbits = std::numeric_limits<CCTK_REAL>::digits - 4;
    const int maxiters = 100;
    int iters;
    bool failed;
    auto [lo, hi] = bisect(fn, 1.0, 2.0, minbits, maxiters, iters, failed);
    assert(!failed);
    assert(iters < maxiters);
    assert(hi >= lo && hi - lo <= std::scalbn(2.0, -minbits));
    assert(fn(lo) <= 0 && fn(hi) >= 0);
    CCTK_VINFO("Test_bisect succeeded in %d iterations", iters);
  }

  {
    const int minbits = std::numeric_limits<CCTK_REAL>::digits - 4;
    const int maxiters = 100;
    int iters;
    bool failed;
    auto [lo, hi] = bracket_and_solve_root(fn, 1.0, 2.0, true, minbits,
                                           maxiters, iters, failed);
    assert(!failed);
    assert(iters < maxiters);
    assert(hi >= lo && hi - lo <= std::scalbn(2.0, -minbits));
    assert(fn(lo) <= 0 && fn(hi) >= 0);
    CCTK_VINFO("Test_bracket_and_solve_root succeeded in %d iterations", iters);
  }

  {
    const int minbits =
        static_cast<int>(0.6 * std::numeric_limits<CCTK_REAL>::digits);
    const int maxiters = 100;
    int iters;
    bool failed;
    auto x =
        newton_raphson(fnd, 1.0, 0.0, 10.0, minbits, maxiters, iters, failed);
    assert(!failed);
    assert(iters < maxiters);
    CCTK_REAL delta = std::scalbn(CCTK_REAL(1), -minbits);
    assert(fn(x - delta) * fn(x + delta) < 0);
    CCTK_VINFO("Test_newton_raphson succeeded in %d iterations", iters);
  }

  {
    const int minbits =
        static_cast<int>(0.6 * std::numeric_limits<CCTK_REAL>::digits);
    const int maxiters = 100;
    int iters;
    bool failed;
    auto x = halley(fnd2, 1.0, 0.0, 10.0, minbits, maxiters, iters, failed);
    assert(!failed);
    assert(iters < maxiters);
    CCTK_REAL delta = std::scalbn(CCTK_REAL(1), -minbits);
    assert(fn(x - delta) * fn(x + delta) < 0);
    CCTK_VINFO("Test_halley succeeded in %d iterations", iters);
  }

  {
    const int minbits =
        static_cast<int>(0.6 * std::numeric_limits<CCTK_REAL>::digits);
    const int maxiters = 100;
    int iters;
    bool failed;
    auto x = schroder(fnd2, 1.0, 0.0, 10.0, minbits, maxiters, iters, failed);
    assert(!failed);
    assert(iters < maxiters);
    CCTK_REAL delta = std::scalbn(CCTK_REAL(1), -minbits);
    assert(fn(x - delta) * fn(x + delta) < 0);

    auto expfn = [](CCTK_REAL x) {
      using std::exp;
      return std::make_tuple(exp(x) - 3, exp(x), exp(x));
    };
    int iters_h, iters_s;
    bool failed_h, failed_s;
    auto xh =
        halley(expfn, 0.1, -5.0, 5.0, minbits, maxiters, iters_h, failed_h);
    auto xs =
        schroder(expfn, 0.1, -5.0, 5.0, minbits, maxiters, iters_s, failed_s);
    assert(!failed_h && !failed_s);
    using std::abs, std::log;
    assert(abs(xh - log(CCTK_REAL(3))) < 1.0e-9);
    assert(abs(xs - log(CCTK_REAL(3))) < 1.0e-9);
    assert(iters_h != iters_s);
    CCTK_VINFO("Test_schroder succeeded in %d iterations", iters);
  }

  {
    const int minbits =
        static_cast<int>(0.6 * std::numeric_limits<CCTK_REAL>::digits);
    const int maxiters = 100;
    int iters;
    bool failed;

    newton_raphson(fnd, 1.0, 10.0, 0.0, minbits, maxiters, iters, failed);
    assert(failed);

    auto noroot = [](CCTK_REAL x) { return std::make_tuple(x * x + 1, 2 * x); };
    newton_raphson(noroot, 1.0, -10.0, 10.0, minbits, maxiters, iters, failed);
    assert(failed);
    auto noroot2 = [](CCTK_REAL x) {
      return std::make_tuple(x * x + 1, 2 * x, 2);
    };
    halley(noroot2, 1.0, -10.0, 10.0, minbits, maxiters, iters, failed);
    assert(failed);
    schroder(noroot2, 1.0, -10.0, 10.0, minbits, maxiters, iters, failed);
    assert(failed);

    bisect(fn, 2.0, 3.0, minbits, maxiters, iters, failed);
    assert(failed);

    bisect(fn, 1.0, 2.0, minbits, 0, iters, failed);
    assert(failed && iters == 0);
    newton_raphson(fnd, 1.0, 0.0, 10.0, minbits, 0, iters, failed);
    assert(failed && iters == 0);

    CCTK_VINFO("Test_wrapper_failure_modes succeeded");
  }

  {
    const int minbits = std::numeric_limits<CCTK_REAL>::digits - 4;
    const int maxiters = 100;
    const CCTK_REAL root = std::sqrt(CCTK_REAL(2));
    int iters;
    bool failed;

    auto tiny = [] ALGO_HOST ALGO_DEVICE(CCTK_REAL x) {
      return 1.0e-200 * (x * x - 2);
    };
    auto [tlo, thi] = brent(tiny, 1.0, 2.0, minbits, maxiters, iters, failed);
    assert(!failed);
    assert(tlo <= root && root <= thi);

    auto [plo, phi] =
        brent(fn, 1.0, 2.0, std::numeric_limits<CCTK_REAL>::digits, maxiters,
              iters, failed);
    assert(!failed);
    assert(plo <= root && root <= phi);

    brent(fn, 2.0, 3.0, minbits, maxiters, iters, failed);
    assert(failed);

    auto lin = [] ALGO_HOST ALGO_DEVICE(CCTK_REAL x) { return x - 1; };
    auto [llo, lhi] = brent(lin, 1.0, 2.0, minbits, maxiters, iters, failed);
    assert(!failed);
    assert(llo == lhi && llo == 1);

    CCTK_VINFO("Test_brent_edge_cases succeeded");
  }

  {
    const int minbits =
        static_cast<int>(0.6 * std::numeric_limits<CCTK_REAL>::digits);
    const CCTK_REAL root = std::sqrt(CCTK_REAL(2));
    int iters;
    bool failed;

    auto g1d = [](vec<CCTK_REAL, 1> x)
        -> std::pair<vec<CCTK_REAL, 1>, mat<CCTK_REAL, 1> > {
      return {vec<CCTK_REAL, 1>{x(0) * x(0) - 2}, mat<CCTK_REAL, 1>{2 * x(0)}};
    };

    using std::abs;
    auto x1 =
        newton_raphson_nd(g1d, vec<CCTK_REAL, 1>{1.0}, vec<CCTK_REAL, 1>{0.0},
                          vec<CCTK_REAL, 1>{10.0}, minbits, 100, iters, failed);
    assert(!failed);
    assert(abs(x1(0) - root) < 1.0e-9);

    auto x2 =
        newton_raphson_nd(g1d, vec<CCTK_REAL, 1>{1.0}, vec<CCTK_REAL, 1>{0.0},
                          vec<CCTK_REAL, 1>{1.2}, minbits, 100, iters, failed);
    assert(failed);
    assert(x2(0) <= 1.2);

    auto g2d = [](vec<CCTK_REAL, 2> x)
        -> std::pair<vec<CCTK_REAL, 2>, mat<CCTK_REAL, 2> > {
      return {vec<CCTK_REAL, 2>{x(0) * x(0) - 2, x(0) * x(1) - 2},
              mat<CCTK_REAL, 2>{2 * x(0), 0, x(1), x(0)}};
    };
    auto x3 = newton_raphson_nd(
        g2d, vec<CCTK_REAL, 2>{1.0, 1.0}, vec<CCTK_REAL, 2>{0.0, 0.0},
        vec<CCTK_REAL, 2>{10.0, 10.0}, minbits, 4, iters, failed);
    assert(!failed);
    assert(sumabs(gn(x3)) < 1.0e-9);

    auto gsing = [](vec<CCTK_REAL, 2> x)
        -> std::pair<vec<CCTK_REAL, 2>, mat<CCTK_REAL, 2> > {
      return {vec<CCTK_REAL, 2>{1.0, 1.0}, mat<CCTK_REAL, 2>{0, 0, 0, 0}};
    };
    newton_raphson_nd(
        gsing, vec<CCTK_REAL, 2>{1.0, 1.0}, vec<CCTK_REAL, 2>{-10.0, -10.0},
        vec<CCTK_REAL, 2>{10.0, 10.0}, minbits, 100, iters, failed);
    assert(failed);
    assert(iters < 100);

    CCTK_VINFO("Test_newton_raphson_nd_edge_cases succeeded");
  }

  amrex::Gpu::Buffer<int> niters({0, 0});
  auto *pniters = niters.data();
  const amrex::Box box(amrex::IntVect(0), amrex::IntVect(0, 0, 0));

  {
    amrex::launch(box, [=] ALGO_DEVICE(
                           amrex::Box const &tbx) CCTK_ATTRIBUTE_ALWAYS_INLINE {
      const int minbits = std::numeric_limits<CCTK_REAL>::digits - 4;
      const int maxiters = 100;
      int iters;
      bool failed;
      auto [lo, hi] = brent(fn, 1.0, 2.0, minbits, maxiters, iters, failed);
      assert(!failed);
      assert(iters < maxiters);
      assert(hi >= lo && hi - lo <= std::scalbn(2.0, -minbits));
      assert(fn(lo) <= 0 && fn(hi) >= 0);
      *pniters = iters;
    });
    auto *hp = niters.copyToHost();
    CCTK_VINFO("Test_brent succeeded in %d iterations", *hp);
  }

  {
    amrex::launch(box, [=] ALGO_DEVICE(
                           amrex::Box const &tbx) CCTK_ATTRIBUTE_ALWAYS_INLINE {
      const int minbits =
          static_cast<int>(0.6 * std::numeric_limits<CCTK_REAL>::digits);
      const int maxiters = 100;
      int iters;
      bool failed;

      auto gnd_CCTK_REAL = [](vec<CCTK_REAL, 2> x)
          -> std::pair<vec<CCTK_REAL, 2>, mat<CCTK_REAL, 2> > {
        return {vec<CCTK_REAL, 2>{x(0) * x(0) - 2, x(0) * x(1) - 2},
                mat<CCTK_REAL, 2>{2 * x(0), 0, x(1), x(0)}};
      };

      auto x = newton_raphson_nd(gnd_CCTK_REAL, vec<CCTK_REAL, 2>{1.0, 1.0},
                                 vec<CCTK_REAL, 2>{0.0, 0.0},
                                 vec<CCTK_REAL, 2>{10.0, 10.0}, minbits,
                                 maxiters, iters, failed);
      assert(!failed);
      assert(iters <= 10);
      assert(sumabs(gn(x)) < 1.0e-9);
      pniters[0] = iters;

      auto gnd3_CCTK_REAL = [](vec<CCTK_REAL, 3> x)
          -> std::pair<vec<CCTK_REAL, 3>, mat<CCTK_REAL, 3> > {
        return {
            vec<CCTK_REAL, 3>{x(0) * x(0) - 2, x(0) * x(1) - 2,
                              x(1) * x(2) - 2},
            mat<CCTK_REAL, 3>{2 * x(0), 0, 0, x(1), x(0), 0, 0, x(2), x(1)}};
      };
      auto x3 = newton_raphson_nd(
          gnd3_CCTK_REAL, vec<CCTK_REAL, 3>{1.0, 1.0, 1.0},
          vec<CCTK_REAL, 3>{0.0, 0.0, 0.0}, vec<CCTK_REAL, 3>{10.0, 10.0, 10.0},
          minbits, maxiters, iters, failed);
      assert(!failed);
      assert(iters <= 10);
      assert(sumabs(gn3(x3)) < 1.0e-9);
      pniters[1] = iters;
    });
    auto *hp = niters.copyToHost();
    CCTK_VINFO("Test_newton_raphson_nd succeeded in %d iterations (2D), "
               "%d iterations (3D)",
               hp[0], hp[1]);
  }
}

} // namespace Algo
