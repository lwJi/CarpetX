#include <carpetx.hxx>
#include <carpetx_rk.hxx>

#include <array>
#include <span>
#include <type_traits>
#include <vector>

namespace TestCarpetXAPI {
namespace {

using CarpetX::gi_t;
using CarpetX::tl_t;
using CarpetX::window_t;

template <class Groups>
concept sync_takes = requires(window_t window, Groups groups) {
  CarpetX::sync(window, groups);
};

template <class... Args>
concept sync_ghosts_only_takes =
    requires(window_t window, std::span<const gi_t> groups, Args... args) {
  CarpetX::sync_ghosts_only(window, groups, args...);
};

template <class... Args>
concept prolongate_ghosts_takes =
    requires(window_t window, std::span<const gi_t> groups, Args... args) {
  CarpetX::prolongate_ghosts(window, groups, args...);
};

template <class... Args>
concept prolongate_aligned_takes =
    requires(window_t window, std::span<const gi_t> groups, Args... args) {
  CarpetX::prolongate_aligned(window, groups, args...);
};

template <class Timelevel>
concept invalidate_takes =
    requires(window_t window, std::span<const gi_t> groups, Timelevel tl,
             const char *why) {
  CarpetX::invalidate(window, groups, tl, why);
};

template <class Timelevel>
concept data_takes = requires(CarpetX::group_t group, Timelevel tl) {
  CarpetX::data(group, tl);
};

template <class Index> concept adds_int = requires(Index index) { index + 1; };

static_assert(std::is_constructible_v<gi_t, int>);
static_assert(!std::is_convertible_v<int, gi_t>);
static_assert(!std::is_convertible_v<gi_t, int>);
static_assert(std::is_trivially_copyable_v<gi_t>);

static_assert(std::is_constructible_v<tl_t, int>);
static_assert(!std::is_convertible_v<int, tl_t>);
static_assert(!std::is_convertible_v<tl_t, int>);
static_assert(std::is_trivially_copyable_v<tl_t>);
static_assert(sizeof(tl_t) == sizeof(int));

static_assert(!std::is_constructible_v<tl_t, gi_t>);
static_assert(!std::is_constructible_v<gi_t, tl_t>);

static_assert(adds_int<int>);
static_assert(!adds_int<gi_t>);
static_assert(!adds_int<tl_t>);
static_assert(tl_t{0}.next() == tl_t{1});

static_assert(sync_takes<std::span<const gi_t> >);
static_assert(sync_takes<std::vector<gi_t> >);
static_assert(sync_takes<std::array<gi_t, 1> >);
static_assert(!sync_takes<std::vector<int> >);

static_assert(sync_ghosts_only_takes<>);
static_assert(sync_ghosts_only_takes<tl_t>);
static_assert(!sync_ghosts_only_takes<int>);
static_assert(!sync_ghosts_only_takes<gi_t>);

static_assert(prolongate_ghosts_takes<>);
static_assert(prolongate_ghosts_takes<tl_t>);
static_assert(!prolongate_ghosts_takes<int>);
static_assert(!prolongate_ghosts_takes<gi_t>);

static_assert(prolongate_aligned_takes<>);
static_assert(prolongate_aligned_takes<tl_t>);
static_assert(!prolongate_aligned_takes<int>);
static_assert(!prolongate_aligned_takes<gi_t>);

static_assert(invalidate_takes<tl_t>);
static_assert(!invalidate_takes<int>);
static_assert(!invalidate_takes<gi_t>);

static_assert(data_takes<tl_t>);
static_assert(!data_takes<int>);
static_assert(!data_takes<gi_t>);

static_assert(std::is_same_v<decltype(CarpetX::rk_pair_t::state), gi_t>);
static_assert(std::is_same_v<decltype(CarpetX::group_t::gi), gi_t>);

} // namespace
} // namespace TestCarpetXAPI
