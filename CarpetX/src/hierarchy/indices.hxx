#ifndef CARPETX_CARPETX_HIERARCHY_INDICES_HXX
#define CARPETX_CARPETX_HIERARCHY_INDICES_HXX

#include "internal.hxx"

#include <carpetx.hxx>

#include <AMReX_FabArrayBase.H>
#include <AMReX_MFIter.H>
#include <AMReX_ParallelDescriptor.H>

#include <cassert>

namespace CarpetX {

template <class Derived>
constexpr Derived next_index(const index_t<Derived> i) {
  return Derived{i.index() + 1};
}
constexpr level_t next_index(const level_t level) {
  return level_t{level.patch(), level.index() + 1};
}

template <class Index> struct index_range {
  struct iterator {
    Index current;
    constexpr Index operator*() const { return current; }
    constexpr iterator &operator++() {
      current = next_index(current);
      return *this;
    }
    friend constexpr bool operator==(iterator, iterator) = default;
  };

  constexpr index_range(const Index begin, const Index end)
      : begin_(begin), end_(end) {}
  constexpr iterator begin() const { return iterator{begin_}; }
  constexpr iterator end() const { return iterator{end_}; }
  constexpr bool contains(const Index i) const {
    return begin_ <= i && i < end_;
  }
  constexpr int size() const { return end_.index() - begin_.index(); }
  constexpr int offset(const Index i) const {
    assert(contains(i));
    return i.index() - begin_.index();
  }
  constexpr Index first() const {
    assert(begin_ < end_);
    return begin_;
  }
  constexpr Index last() const {
    assert(begin_ < end_);
    return Index{end_.index() - 1};
  }

private:
  Index begin_, end_;
};

using component_range = index_range<component_t>;

constexpr component_range components_of(const int count) {
  assert(count >= 0);
  return component_range{component_t{0}, component_t{count}};
}
inline component_range components_of(const amrex::FabArrayBase &fab) {
  return components_of(fab.size());
}

inline tile_t tile_of(const amrex::MFIter &mfi) {
  assert(mfi.isValid());
  assert(amrex::ParallelDescriptor::TeamSize() == 1);
  return tile_t{component_t{mfi.index()}, mfi.tileIndex()};
}

} // namespace CarpetX

#endif
