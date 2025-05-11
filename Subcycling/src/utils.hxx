#ifndef CARPETX_SUBCYCLING_UTILS_HXX
#define CARPETX_SUBCYCLING_UTILS_HXX

#include <cctk.h>
#include <util_Table.h>

#include <unordered_map>

namespace Subcycling {
using namespace std;

enum class centering_t { ccc = 0, vcc, cvc, ccv, cvv, vcv, vvc, vvv, ntypes };

constexpr array<array<int, Loop::dim>, static_cast<size_t>(centering_t::ntypes)>
    g_indextypes{
        array<int, Loop::dim>{1, 1, 1}, array<int, Loop::dim>{0, 1, 1},
        array<int, Loop::dim>{1, 0, 1}, array<int, Loop::dim>{1, 1, 0},
        array<int, Loop::dim>{1, 0, 0}, array<int, Loop::dim>{0, 1, 0},
        array<int, Loop::dim>{0, 0, 1}, array<int, Loop::dim>{0, 0, 0}};

constexpr std::array<const char *, static_cast<size_t>(centering_t::ntypes)>
    g_isrmbndry_strs = {
        "Subcycling::isrmbndry_ccc", "Subcycling::isrmbndry_vcc",
        "Subcycling::isrmbndry_cvc", "Subcycling::isrmbndry_ccv",
        "Subcycling::isrmbndry_cvv", "Subcycling::isrmbndry_vcv",
        "Subcycling::isrmbndry_vvc", "Subcycling::isrmbndry_vvv"};

/**
 * \brief return refinement boundary flag grid function indexes
 */

inline std::array<int, static_cast<size_t>(centering_t::ntypes)>
get_isrmbndry_idx() {
  std::array<int, static_cast<size_t>(centering_t::ntypes)> indices;
  for (size_t i = 0; i < g_isrmbndry_strs.size(); ++i) {
    indices[i] = CCTK_VarIndex(g_isrmbndry_strs[i]);
  }
  return indices;
}

} // namespace Subcycling

#endif // #ifndef CARPETX_SUBCYCLING_UTILS_HXX
