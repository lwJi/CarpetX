#ifndef CARPETX_ODESOLVERS_SOLVE_HXX
#define CARPETX_ODESOLVERS_SOLVE_HXX

// TODO: Don't include files from other thorns; create a proper interface
#include "../../CarpetX/src/driver.hxx"
#include "../../CarpetX/src/schedule.hxx"

#include <cctk.h>
#include <cctk_Parameters.h>
#include <cctk_Arguments.h>
#include <util_Table.h>

#include <div.hxx>

#include <AMReX_MultiFab.H>

#if defined _OPENMP || defined __HIPCC__
#include <omp.h>
#else
static inline int omp_get_max_threads() { return 1; }
#endif

#include <algorithm>
#include <array>
#include <cassert>
#include <cctype>
#include <cmath>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace ODESolvers {
using namespace std;

////////////////////////////////////////////////////////////////////////////////

// Taken from <https://en.cppreference.com/w/cpp/experimental/make_array>
namespace details {
template <class> struct is_ref_wrapper : std::false_type {};
template <class T>
struct is_ref_wrapper<std::reference_wrapper<T> > : std::true_type {};

template <class T>
using not_ref_wrapper = std::negation<is_ref_wrapper<std::decay_t<T> > >;

template <class D, class...> struct return_type_helper {
  using type = D;
};
template <class... Types>
struct return_type_helper<void, Types...> : std::common_type<Types...> {
  static_assert(std::conjunction_v<not_ref_wrapper<Types>...>,
                "Types cannot contain reference_wrappers when D is void");
};

template <class D, class... Types>
using return_type = std::array<typename return_type_helper<D, Types...>::type,
                               sizeof...(Types)>;
} // namespace details

template <class D = void, class... Types>
constexpr details::return_type<D, Types...> make_array(Types &&...t) {
  return {std::forward<Types>(t)...};
}

////////////////////////////////////////////////////////////////////////////////

// A state vector component, with mfabs for each level, group, and variable
struct statecomp_t {

  statecomp_t() = default;

  statecomp_t(statecomp_t &&) = default;
  statecomp_t &operator=(statecomp_t &&) = default;

  // Don't allow copies because we might own stuff
  statecomp_t(const statecomp_t &) = delete;
  statecomp_t &operator=(const statecomp_t &) = delete;

  vector<GHExt::PatchData::LevelData::GroupData *> groupdatas;
  vector<amrex::MultiFab *> mfabs;

  static void init_tmp_mfabs();
  static void free_tmp_mfabs();

  void set_valid(const valid_t valid) const;
  template <size_t N>
  static void combine_valids(const statecomp_t &dst, const CCTK_REAL scale,
                             const array<CCTK_REAL, N> &factors,
                             const array<const statecomp_t *, N> &srcs);
  void check_valid(const valid_t required, const function<string()> &why) const;
  void check_valid(const valid_t required, const string &why) const {
    check_valid(required, [=]() { return why; });
  }

  statecomp_t copy(const valid_t where) const;

  template <size_t N>
  static void lincomb(const statecomp_t &dst, CCTK_REAL scale,
                      const array<CCTK_REAL, N> &factors,
                      const array<const statecomp_t *, N> &srcs,
                      const valid_t where);
  template <size_t N>
  static void lincomb(const statecomp_t &dst, CCTK_REAL scale,
                      const array<CCTK_REAL, N> &factors,
                      const array<statecomp_t *, N> &srcs,
                      const valid_t where) {
    array<const statecomp_t *, N> srcs1;
    for (size_t n = 0; n < N; ++n)
      srcs1[n] = srcs[n];
    lincomb(dst, scale, factors, srcs1, where);
  }

  static void lincomb(const statecomp_t &dst, CCTK_REAL scale,
                      const vector<CCTK_REAL> &factors,
                      const vector<const statecomp_t *> &srcs,
                      const valid_t where);
};

////////////////////////////////////////////////////////////////////////////////

int groupindex(const int other_gi, std::string gn);

int get_group_rhs(const int gi);

std::vector<int> get_group_dependents(const int gi);

void mark_invalid(const std::vector<int> &groups);

} // namespace ODESolvers

#endif // #ifndef CARPETX_ODESOLVERS_SOLVE_HXX
