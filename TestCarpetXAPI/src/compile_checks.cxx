#include <carpetx.hxx>
#include <carpetx_rk.hxx>

#include <array>
#include <span>
#include <type_traits>
#include <vector>

namespace TestCarpetXAPI {
namespace {

using CarpetX::gi_t;
using CarpetX::window_t;

template <class Groups>
concept sync_takes = requires(window_t window, Groups groups) {
  CarpetX::sync(window, groups);
};

template <class Index>
concept adds_int = requires(Index index) { index + 1; };

static_assert(std::is_constructible_v<gi_t, int>);
static_assert(!std::is_convertible_v<int, gi_t>);
static_assert(!std::is_convertible_v<gi_t, int>);
static_assert(std::is_trivially_copyable_v<gi_t>);

static_assert(adds_int<int>);
static_assert(!adds_int<gi_t>);

static_assert(sync_takes<std::span<const gi_t> >);
static_assert(sync_takes<std::vector<gi_t> >);
static_assert(sync_takes<std::array<gi_t, 1> >);
static_assert(!sync_takes<std::vector<int> >);

static_assert(std::is_same_v<decltype(CarpetX::rk_pair_t::state), gi_t>);
static_assert(std::is_same_v<decltype(CarpetX::group_t::gi), gi_t>);

} // namespace
} // namespace TestCarpetXAPI
