#ifndef CARPETX_SUBCYCLING_UTILS_HXX
#define CARPETX_SUBCYCLING_UTILS_HXX

#include <cctk.h>
#include <util_Table.h>

namespace Subcycling {
using namespace std;

enum class centering_t { ccc = 0, vcc, cvc, ccv, cvv, vcv, vvc, vvv, ntypes };

constexpr array<array<int, Loop::dim>, static_cast<size_t>(centering_t::ntypes)>
    g_indextypes{
        array<int, Loop::dim>{1, 1, 1}, array<int, Loop::dim>{0, 1, 1},
        array<int, Loop::dim>{1, 0, 1}, array<int, Loop::dim>{1, 1, 0},
        array<int, Loop::dim>{1, 0, 0}, array<int, Loop::dim>{0, 1, 0},
        array<int, Loop::dim>{0, 0, 1}, array<int, Loop::dim>{0, 0, 0}};

/**
 * \brief return refinement boundary flag grid function indexes
 */
inline array<int, static_cast<size_t>(centering_t::ntypes)>
get_isrmbndry_idx() {
  return array<int, static_cast<size_t>(centering_t::ntypes)>{
      CCTK_VarIndex("Subcycling::isrmbndry_ccc"),
      CCTK_VarIndex("Subcycling::isrmbndry_vcc"),
      CCTK_VarIndex("Subcycling::isrmbndry_cvc"),
      CCTK_VarIndex("Subcycling::isrmbndry_ccv"),
      CCTK_VarIndex("Subcycling::isrmbndry_cvv"),
      CCTK_VarIndex("Subcycling::isrmbndry_vcv"),
      CCTK_VarIndex("Subcycling::isrmbndry_vvc"),
      CCTK_VarIndex("Subcycling::isrmbndry_vvv")};
}

} // namespace Subcycling

#endif // #ifndef CARPETX_SUBCYCLING_UTILS_HXX
