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
using CarpetX::var_t;
using CarpetX::vi_t;
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

template <class Variable>
concept valid_takes = requires(CarpetX::group_t group, tl_t tl, Variable vi) {
  CarpetX::valid(group, tl, vi);
};

template <class Variable>
concept reduce_takes = requires(gi_t gi, Variable vi, tl_t tl) {
  CarpetX::reduce(gi, vi, tl);
};

template <class Var> concept variable_takes = requires(Var var) {
  CarpetX::variable(var);
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

static_assert(std::is_constructible_v<vi_t, int>);
static_assert(!std::is_convertible_v<int, vi_t>);
static_assert(!std::is_convertible_v<vi_t, int>);
static_assert(std::is_trivially_copyable_v<vi_t>);
static_assert(sizeof(vi_t) == sizeof(int));

static_assert(std::is_constructible_v<var_t, int>);
static_assert(!std::is_convertible_v<int, var_t>);
static_assert(!std::is_convertible_v<var_t, int>);
static_assert(std::is_trivially_copyable_v<var_t>);

static_assert(!std::is_constructible_v<var_t, vi_t>);
static_assert(!std::is_constructible_v<vi_t, var_t>);
static_assert(!std::is_constructible_v<vi_t, gi_t>);
static_assert(!std::is_constructible_v<gi_t, vi_t>);
static_assert(!std::is_constructible_v<vi_t, tl_t>);
static_assert(!std::is_constructible_v<tl_t, vi_t>);
static_assert(!std::is_constructible_v<var_t, gi_t>);

static_assert(adds_int<int>);
static_assert(!adds_int<gi_t>);
static_assert(!adds_int<tl_t>);
static_assert(!adds_int<vi_t>);
static_assert(!adds_int<var_t>);
static_assert(tl_t{0}.next() == tl_t{1});

static_assert(valid_takes<vi_t>);
static_assert(!valid_takes<gi_t>);
static_assert(!valid_takes<var_t>);
static_assert(!valid_takes<tl_t>);
static_assert(!valid_takes<int>);

static_assert(reduce_takes<vi_t>);
static_assert(!reduce_takes<var_t>);
static_assert(!reduce_takes<tl_t>);
static_assert(!reduce_takes<int>);

static_assert(variable_takes<var_t>);
static_assert(!variable_takes<vi_t>);
static_assert(!variable_takes<gi_t>);
static_assert(!variable_takes<int>);

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
static_assert(std::is_same_v<decltype(CarpetX::variable_t::gi), gi_t>);
static_assert(std::is_same_v<decltype(CarpetX::variable_t::vi), vi_t>);

} // namespace
} // namespace TestCarpetXAPI
