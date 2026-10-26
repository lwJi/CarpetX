#include <carpetx.hxx>
#include <carpetx_rk.hxx>

#include <array>
#include <optional>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

namespace TestCarpetXAPI {
namespace {

using CarpetX::gi_t;
using CarpetX::level_t;
using CarpetX::patch_t;
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

template <class Level> concept subtracts_int = requires(Level level) {
  level - 1;
};

template <class Level> concept reads_index = requires(Level level) {
  level.index();
};

template <class Level> concept reads_level_field = requires(Level level) {
  level.level;
};

template <class Patch> concept num_levels_takes = requires(Patch patch) {
  CarpetX::num_levels(patch);
};

template <class Window> concept reads_min_level = requires(Window window) {
  window.min_level;
};

template <class Window> concept braces_from_ints = requires(int i) {
  Window{i, i, i, i};
};

template <class Level> concept one_level_takes = requires(Level level) {
  window_t::one_level(level);
};

template <class Window> concept operates_on = requires(Window window) {
  CarpetX::sync(window, std::span<const gi_t>());
};

struct bare_window_t {
  int min_level, max_level, min_patch, max_patch;
};

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

static_assert(std::is_constructible_v<patch_t, int>);
static_assert(!std::is_convertible_v<int, patch_t>);
static_assert(!std::is_convertible_v<patch_t, int>);
static_assert(std::is_trivially_copyable_v<patch_t>);
static_assert(!std::is_constructible_v<patch_t, gi_t>);
static_assert(!std::is_constructible_v<gi_t, patch_t>);

static_assert(std::is_constructible_v<level_t, patch_t, int>);
static_assert(!std::is_constructible_v<level_t, int, int>);
static_assert(!std::is_constructible_v<level_t, int, patch_t>);
static_assert(!std::is_default_constructible_v<level_t>);
static_assert(std::is_trivially_copyable_v<level_t>);
static_assert(sizeof(level_t) == 2 * sizeof(int));
static_assert(level_t{patch_t{2}, 1}.patch() == patch_t{2});
static_assert(level_t{patch_t{2}, 1}.index() == 1);
static_assert(level_t{patch_t{0}, 1} < level_t{patch_t{1}, 0});
static_assert(
    std::is_same_v<decltype(std::declval<level_t>().parent()), level_t>);
static_assert(std::is_same_v<decltype(std::declval<level_t>().child()),
                             std::optional<level_t> >);

static_assert(reads_index<level_t>);
static_assert(!reads_level_field<level_t>);

static_assert(num_levels_takes<patch_t>);
static_assert(!num_levels_takes<int>);

static_assert(subtracts_int<int>);
static_assert(!subtracts_int<level_t>);
static_assert(!subtracts_int<patch_t>);

static_assert(adds_int<int>);
static_assert(!adds_int<level_t>);
static_assert(!adds_int<patch_t>);
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

static_assert(reads_min_level<bare_window_t>);
static_assert(!reads_min_level<window_t>);
static_assert(braces_from_ints<bare_window_t>);
static_assert(!braces_from_ints<window_t>);
static_assert(!std::is_constructible_v<window_t, int, int, int, int>);
static_assert(!std::is_default_constructible_v<window_t>);
static_assert(std::is_trivially_copyable_v<window_t>);
static_assert(one_level_takes<level_t>);
static_assert(!one_level_takes<int>);
static_assert(!one_level_takes<patch_t>);
static_assert(std::is_same_v<decltype(window_t::levels(0, 1)), window_t>);
static_assert(
    std::is_same_v<decltype(std::declval<window_t>().narrow(0, 1)), window_t>);
static_assert(
    std::is_same_v<decltype(std::declval<window_t>().coarsest()), level_t>);
static_assert(
    std::is_same_v<decltype(std::declval<window_t>().finest()), level_t>);
static_assert(operates_on<window_t>);
static_assert(!operates_on<level_t>);
static_assert(!operates_on<bare_window_t>);

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
static_assert(std::is_same_v<decltype(CarpetX::group_t::level), level_t>);
static_assert(std::is_same_v<decltype(CarpetX::variable_t::gi), gi_t>);
static_assert(std::is_same_v<decltype(CarpetX::variable_t::vi), vi_t>);

} // namespace
} // namespace TestCarpetXAPI
