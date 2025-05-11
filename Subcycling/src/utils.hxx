#ifndef CARPETX_SUBCYCLING_UTILS_HXX
#define CARPETX_SUBCYCLING_UTILS_HXX

#include <cctk.h>
#include <util_Table.h>

#include <unordered_map>

namespace Subcycling {

enum class centering_t { ccc = 0, vcc, cvc, ccv, cvv, vcv, vvc, vvv, ntypes };

constexpr std::array<std::array<int, Loop::dim>,
                     static_cast<size_t>(centering_t::ntypes)>
    g_indextypes{std::array<int, Loop::dim>{1, 1, 1},
                 std::array<int, Loop::dim>{0, 1, 1},
                 std::array<int, Loop::dim>{1, 0, 1},
                 std::array<int, Loop::dim>{1, 1, 0},
                 std::array<int, Loop::dim>{1, 0, 0},
                 std::array<int, Loop::dim>{0, 1, 0},
                 std::array<int, Loop::dim>{0, 0, 1},
                 std::array<int, Loop::dim>{0, 0, 0}};

constexpr std::array<const char *, static_cast<size_t>(centering_t::ntypes)>
    g_isrmbndry_strs = {
        "Subcycling::isrmbndry_ccc", "Subcycling::isrmbndry_vcc",
        "Subcycling::isrmbndry_cvc", "Subcycling::isrmbndry_ccv",
        "Subcycling::isrmbndry_cvv", "Subcycling::isrmbndry_vcv",
        "Subcycling::isrmbndry_vvc", "Subcycling::isrmbndry_vvv"};

// Custom hash function for std::array<int, Loop::dim>
struct ArrayHash {
  template <typename T, std::size_t N>
  std::size_t operator()(const std::array<T, N> &arr) const {
    std::size_t hash = 0;
    for (const auto &elem : arr) {
      hash ^= std::hash<T>{}(elem) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
    }
    return hash;
  }
};

using IndexMap = std::unordered_map<std::array<int, Loop::dim>, int, ArrayHash>;

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

inline IndexMap construct_index_map() {
  IndexMap index_map;
  auto indices = get_isrmbndry_idx();
  for (size_t i = 0; i < g_indextypes.size(); ++i) {
    index_map[g_indextypes[i]] = indices[i];
  }
  return index_map;
}

} // namespace Subcycling

#endif // #ifndef CARPETX_SUBCYCLING_UTILS_HXX
