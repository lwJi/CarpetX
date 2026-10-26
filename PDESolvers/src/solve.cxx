#include "pdesolvers.hxx"

#include <carpetx.hxx>

#include <arr.hxx>
#include <div.hxx>
#include <loop.hxx>
#include <vect.hxx>

#include <cctk.h>
#include <cctk_Arguments.h>
#include <cctk_Parameters.h>

#include <mpi.h>

#include <petscdm.h>
#include <petscsnes.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <functional>
#include <iostream>
#include <optional>
#include <span>
#include <sstream>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace PDESolvers {

const int tl = 0;

// TODO: Generalize this
const Arith::arr<int, 3> indextype{0, 0, 0}; // vertex centred

////////////////////////////////////////////////////////////////////////////////

// Erik Schnetter the blocks for vertex centred grids overlap by one
// point at the block boundaries (see
// https://amrex-codes.github.io/amrex/docs_html/Basics.html#id4).
// does FillPatchSingleLevel ensure these points are consistent? or is
// the domain double-valued there, and only the ghost points (not
// shown in this figure) are set?

// weiqun If the original source data are consistent, the results of
// FillPatchSingleLevel are also consistent.  Otherwise, it's undefined which
// original values will used for which boxes.  It depends on how boxes are
// distributed.

// If you want to the data consistent, you can use MultiFab's OverrideSync
// functions.
//     void OverrideSync (const Periodicity& period =
//         Periodicity::NonPeriodic());
//     void OverrideSync (const iMultiFab& msk,
//         const Periodicity& period = Periodicity::NonPeriodic());

// The firs version will use data from the grid with the lowest grid number to
// override others.

// In the second version, you can pass your own mask.

// Erik Schnetter thanks!

// weiqun Even if you want to use the first version's mask, you might still want
// to call
//     //! Owner is the grid with the lowest grid number containing the data.
//     std::unique_ptr<iMultiFab> OwnerMask (const Periodicity& period =
//         Periodicity::NonPeriodic()) const;
// and then use the second function.

// If you need to call OverrideSync repeatedly, you could save the mask.

// Erik Schnetter thanks!

////////////////////////////////////////////////////////////////////////////////

// 1: interior (defined by user)
// 2: outer boundary (also defined by users)
// 3: defined via synchronization (copied from another box on same level)
// 4: defined via prolongation (interpolated from next coarser level)
// 5: defined via restriction (injected from next finer level)
enum class point_type_t {
  undf = 0,
  intr = 1,
  bdry = 2,
  sync = 3,
  prol = 4,
  rest = 5
};

std::vector<CarpetX::write_t> open_writes(std::span<const CarpetX::gi_t> gis,
                                          const CarpetX::valid_t leaves_valid,
                                          const char *const what) {
  std::vector<CarpetX::write_t> writes;
  CarpetX::active_window().loop_serially([&](const CarpetX::level_t level) {
    for (const CarpetX::gi_t gi : gis)
      writes.push_back(
          CarpetX::write(CarpetX::group_t{level, gi}, tl, leaves_valid, what));
  });
  return writes;
}

void define_point_type() {
  const int vn_pt = CCTK_VarIndex("PDESolvers::point_type");
  assert(vn_pt >= 0);
  const CarpetX::gi_t gi_pt{CCTK_GroupIndexFromVarI(vn_pt)};
  assert(gi_pt.index() >= 0);
  const int v0_pt = CCTK_FirstVarIndexI(gi_pt.index());
  assert(v0_pt >= 0);
  const int vi_pt = vn_pt - v0_pt;
  assert(vi_pt >= 0);
  const int vn_ind = CCTK_VarIndex("PDESolvers::communication_indicator");
  assert(vn_ind >= 0);
  const CarpetX::gi_t gi_ind{CCTK_GroupIndexFromVarI(vn_ind)};
  assert(gi_ind.index() >= 0);
  const int v0_ind = CCTK_FirstVarIndexI(gi_ind.index());
  assert(v0_ind >= 0);
  const int vi_ind = vn_ind - v0_ind;
  assert(vi_ind >= 0);

  {
    const auto writes = open_writes(std::array{gi_pt}, CarpetX::make_valid_all(),
                                    "PDESolver::define_point_type");
    CarpetX::active_window().loop_parallel(
        [&](const CarpetX::level_t lvl, const int index, const int component,
            const cGH *restrict const cctkGH) {
          const Loop::GF3D2layout layout1(cctkGH, indextype);
          const Loop::GF3D2<CCTK_REAL> gf_pt(
              layout1,
              static_cast<CCTK_REAL *>(CCTK_VarDataPtrI(cctkGH, tl, vn_pt)));
          const Loop::GridDescBase grid(cctkGH);
          grid.loop_all<0, 0, 0>(grid.nghostzones,
                                 [&](const Loop::PointDesc &p) ARITH_INLINE {
                                   gf_pt(p.I) = int(point_type_t::undf);
                                 });
          grid.loop_int<0, 0, 0>(
              grid.nghostzones, [&](const Loop::PointDesc &p) ARITH_INLINE {
                assert(gf_pt(p.I) == int(point_type_t::undf));
                gf_pt(p.I) = int(point_type_t::intr);
              });
          grid.loop_bnd<0, 0, 0>(
              grid.nghostzones, [&](const Loop::PointDesc &p) ARITH_INLINE {
                assert(gf_pt(p.I) == int(point_type_t::undf));
                gf_pt(p.I) = int(point_type_t::bdry);
              });
        });
  }

  {
    const auto writes =
        open_writes(std::array{gi_ind}, CarpetX::make_valid_all(),
                    "PDESolver::define_point_type before restricting");
    CarpetX::active_window().loop_parallel(
        [&](const CarpetX::level_t lvl, const int index, const int component,
            const cGH *restrict const cctkGH) {
          const int level = lvl.level;
          const Loop::GF3D2layout layout1(cctkGH, indextype);
          const Loop::GF3D2<CCTK_REAL> gf_ind(
              layout1,
              static_cast<CCTK_REAL *>(CCTK_VarDataPtrI(cctkGH, tl, vn_ind)));
          const Loop::GridDescBase grid(cctkGH);
          grid.loop_all<0, 0, 0>(grid.nghostzones,
                                 [&](const Loop::PointDesc &p)
                                     ARITH_INLINE { gf_ind(p.I) = level; });
        });
  }

  CarpetX::restrict_no_poison(CarpetX::active_window(), std::array{gi_ind});

  CarpetX::active_window().loop_parallel([&](const CarpetX::level_t lvl,
                                             const int index,
                                             const int component,
                                             const cGH *restrict const cctkGH) {
    const int level = lvl.level;
    const Loop::GF3D2layout layout1(cctkGH, indextype);
    const Loop::GF3D2<CCTK_REAL> gf_pt(
        layout1, static_cast<CCTK_REAL *>(CCTK_VarDataPtrI(cctkGH, tl, vn_pt)));
    const Loop::GF3D2<const CCTK_REAL> gf_ind(
        layout1,
        static_cast<const CCTK_REAL *>(CCTK_VarDataPtrI(cctkGH, tl, vn_ind)));
    const Loop::GridDescBase grid(cctkGH);
    grid.loop_all<0, 0, 0>(grid.nghostzones,
                           [&](const Loop::PointDesc &p) ARITH_INLINE {
                             if (gf_ind(p.I) != level) {
                               assert(gf_pt(p.I) != int(point_type_t::bdry));
                               gf_pt(p.I) = int(point_type_t::rest);
                             }
                           });
  });

  {
    const auto writes =
        open_writes(std::array{gi_ind}, CarpetX::make_valid_all(),
                    "PDESolver::define_point_type before synchronizing");
    CarpetX::active_window().loop_parallel(
        [&](const CarpetX::level_t lvl, const int index, const int component,
            const cGH *restrict const cctkGH) {
          const Loop::GF3D2layout layout1(cctkGH, indextype);
          const Loop::GF3D2<CCTK_REAL> gf_ind(
              layout1,
              static_cast<CCTK_REAL *>(CCTK_VarDataPtrI(cctkGH, tl, vn_ind)));
          const Loop::GridDescBase grid(cctkGH);
          grid.loop_all<0, 0, 0>(grid.nghostzones,
                                 [&](const Loop::PointDesc &p)
                                     ARITH_INLINE { gf_ind(p.I) = index; });
        });
  }

  CarpetX::sync_ghosts_only(CarpetX::active_window(), std::array{gi_ind}, tl);

  CarpetX::active_window().loop_parallel([&](const CarpetX::level_t lvl,
                                             const int index,
                                             const int component,
                                             const cGH *restrict const cctkGH) {
    const Loop::GF3D2layout layout1(cctkGH, indextype);
    const Loop::GF3D2<CCTK_REAL> gf_pt(
        layout1, static_cast<CCTK_REAL *>(CCTK_VarDataPtrI(cctkGH, tl, vn_pt)));
    const Loop::GF3D2<const CCTK_REAL> gf_ind(
        layout1,
        static_cast<const CCTK_REAL *>(CCTK_VarDataPtrI(cctkGH, tl, vn_ind)));
    const Loop::GridDescBase grid(cctkGH);
    grid.loop_all<0, 0, 0>(grid.nghostzones,
                           [&](const Loop::PointDesc &p) ARITH_INLINE {
                             if (gf_ind(p.I) != index) {
                               assert(gf_pt(p.I) == int(point_type_t::undf) ||
                                      gf_pt(p.I) == int(point_type_t::bdry));
                               gf_pt(p.I) = int(point_type_t::sync);
                             }
                           });
  });

  {
    const auto writes =
        open_writes(std::array{gi_ind}, CarpetX::make_valid_all(),
                    "PDESolver::define_point_type before prolongating");
    CarpetX::active_window().loop_parallel(
        [&](const CarpetX::level_t lvl, const int index, const int component,
            const cGH *restrict const cctkGH) {
          const int level = lvl.level;
          const Loop::GF3D2layout layout1(cctkGH, indextype);
          const Loop::GF3D2<CCTK_REAL> gf_ind(
              layout1,
              static_cast<CCTK_REAL *>(CCTK_VarDataPtrI(cctkGH, tl, vn_ind)));
          const Loop::GridDescBase grid(cctkGH);
          grid.loop_all<0, 0, 0>(grid.nghostzones,
                                 [&](const Loop::PointDesc &p)
                                     ARITH_INLINE { gf_ind(p.I) = level; });
        });
  }

  CarpetX::prolongate_ghosts(CarpetX::active_window(), std::array{gi_ind}, tl);

  CarpetX::active_window().loop_parallel([&](const CarpetX::level_t lvl,
                                             const int index,
                                             const int component,
                                             const cGH *restrict const cctkGH) {
    const int level = lvl.level;
    const Loop::GF3D2layout layout1(cctkGH, indextype);
    const Loop::GF3D2<CCTK_REAL> gf_pt(
        layout1, static_cast<CCTK_REAL *>(CCTK_VarDataPtrI(cctkGH, tl, vn_pt)));
    const Loop::GF3D2<const CCTK_REAL> gf_ind(
        layout1,
        static_cast<const CCTK_REAL *>(CCTK_VarDataPtrI(cctkGH, tl, vn_ind)));
    const Loop::GridDescBase grid(cctkGH);
    grid.loop_all<0, 0, 0>(
        grid.nghostzones, [&](const Loop::PointDesc &p) ARITH_INLINE {
          if (gf_ind(p.I) != level && gf_pt(p.I) != int(point_type_t::sync)) {
            assert(gf_pt(p.I) == int(point_type_t::undf) ||
                   gf_pt(p.I) == int(point_type_t::bdry));
            assert(gf_pt(p.I) != int(point_type_t::rest));
            gf_pt(p.I) = int(point_type_t::prol);
          }
        });
  });

  CarpetX::invalidate(CarpetX::active_window(), std::array{gi_ind}, tl,
                      "PDESolver::define_point_type");

  Arith::vect<int, 6> npoints{0, 0, 0, 0, 0, 0};
  CarpetX::active_window().loop_parallel(
      [&](const CarpetX::level_t lvl, const int index, const int component,
          const cGH *restrict const cctkGH) {
        const Loop::GF3D2layout layout1(cctkGH, indextype);
        const Loop::GF3D2<const CCTK_REAL> gf_pt(
            layout1, static_cast<const CCTK_REAL *>(
                         CCTK_VarDataPtrI(cctkGH, tl, vn_pt)));
        const Loop::GridDescBase grid(cctkGH);
        Arith::vect<int, 6> npoints1{0, 0, 0, 0, 0, 0};
        grid.loop_all<0, 0, 0>(grid.nghostzones,
                               [&](const Loop::PointDesc &p) ARITH_INLINE {
                                 const int pt = int(gf_pt(p.I));
                                 assert(pt >= 0 && pt < 6);
                                 ++npoints1[pt];
                               });
        for (int n = 0; n < 6; ++n)
#pragma omp atomic
          npoints[n] += npoints1[n];
      });
  CCTK_VINFO("Point type counts:");
  CCTK_VINFO("  undefined:    %d", npoints[int(point_type_t::undf)]);
  CCTK_VINFO("  interior:     %d", npoints[int(point_type_t::intr)]);
  CCTK_VINFO("  boundary:     %d", npoints[int(point_type_t::bdry)]);
  CCTK_VINFO("  synchronized: %d", npoints[int(point_type_t::sync)]);
  CCTK_VINFO("  prolongated:  %d", npoints[int(point_type_t::prol)]);
  CCTK_VINFO("  restricted:   %d", npoints[int(point_type_t::rest)]);
  CCTK_VINFO("  total:        %d", sum(npoints));
  assert(npoints[int(point_type_t::undf)] == 0);
}

////////////////////////////////////////////////////////////////////////////////

void enumerate_points(
    int &restrict npoints_local, int &restrict npoints_global,
    std::vector<std::vector<int> > &restrict component_offsets,
    std::vector<std::vector<int> > &restrict component_sizes,
    int &restrict npoints_prolongated_local,
    int &restrict npoints_prolongated_global,
    std::vector<std::vector<int> > &restrict component_prolongated_offsets,
    std::vector<std::vector<int> > &restrict component_prolongated_sizes,
    csr_t &Jp) {
  int myproc;
  MPI_Comm_rank(MPI_COMM_WORLD, &myproc);

  static const int vn_idx = CCTK_VarIndex("PDESolvers::idx");
  assert(vn_idx >= 0);
  static const CarpetX::gi_t gi_idx{CCTK_GroupIndexFromVarI(vn_idx)};
  assert(gi_idx.index() >= 0);
  static const int v0_idx = CCTK_FirstVarIndexI(gi_idx.index());
  assert(v0_idx >= 0);
  static const int vi_idx = vn_idx - v0_idx;
  assert(vi_idx >= 0);
  const int vn_pt = CCTK_VarIndex("PDESolvers::point_type");
  assert(vn_pt >= 0);
  const int gi_pt = CCTK_GroupIndexFromVarI(vn_pt);
  assert(gi_pt >= 0);
  const int v0_pt = CCTK_FirstVarIndexI(gi_pt);
  assert(v0_pt >= 0);
  const int vi_pt = vn_pt - v0_pt;
  assert(vi_pt >= 0);

  assert(CarpetX::num_patches() == 1);
  const int nlevels = CarpetX::num_levels(0);
  std::vector<int> level_sizes(nlevels, 0);
  std::vector<int> level_maxcomponents(nlevels, -1);
  CarpetX::active_window().loop_parallel(
      [&](const CarpetX::level_t lvl, const int index, const int component,
          const cGH *restrict const cctkGH) {
        const int level = lvl.level;
#pragma omp critical(PDESolvers_enumerate_points)
        {
          ++level_sizes.at(level);
          int &maxcomponent = level_maxcomponents.at(level);
          using std::max;
          maxcomponent = max(maxcomponent, component);
        }
      });

  component_offsets.resize(nlevels);
  component_sizes.resize(nlevels);
  component_prolongated_offsets.resize(nlevels);
  component_prolongated_sizes.resize(nlevels);
  for (int level = 0; level < nlevels; ++level) {
    assert(level_maxcomponents.at(level) + 1 == level_sizes.at(level));
    component_offsets.at(level).resize(level_sizes.at(level));
    component_sizes.at(level).resize(level_sizes.at(level));
    component_prolongated_offsets.at(level).resize(level_sizes.at(level));
    component_prolongated_sizes.at(level).resize(level_sizes.at(level));
  }

  CarpetX::active_window().loop_parallel([&](const CarpetX::level_t lvl,
                                             const int index,
                                             const int component,
                                             const cGH *restrict const cctkGH) {
    const int level = lvl.level;
    const Loop::GF3D2layout layout1(cctkGH, indextype);
    const Loop::GF3D2<const CCTK_REAL> gf_pt(
        layout1,
        static_cast<const CCTK_REAL *>(CCTK_VarDataPtrI(cctkGH, tl, vn_pt)));
    const Loop::GridDescBase grid(cctkGH);
    int npoints = 0;
    int npoints_prolongated = 0;
    grid.loop_all<0, 0, 0>(grid.nghostzones,
                           [&](const Loop::PointDesc &p) ARITH_INLINE {
                             switch (int(gf_pt(p.I))) {
                             case int(point_type_t::intr):
                               ++npoints;
                               break;
                             case int(point_type_t::bdry):
                             case int(point_type_t::rest):
                             case int(point_type_t::sync):
                               // ignore this point
                               break;
                             case int(point_type_t::prol):
                               ++npoints_prolongated;
                               break;
                             default:
                               assert(0);
                             }
                           });
    component_sizes.at(level).at(component) = npoints;
    component_prolongated_sizes.at(level).at(component) = npoints_prolongated;
  });

  // Local exclusive prefix sum
  int npoints = 0;
  int npoints_prolongated = 0;
  for (std::size_t level = 0; level < component_offsets.size(); ++level) {
    for (std::size_t component = 0;
         component < component_offsets.at(level).size(); ++component) {
      component_offsets.at(level).at(component) = npoints;
      npoints += component_sizes.at(level).at(component);
      component_prolongated_offsets.at(level).at(component) =
          npoints_prolongated;
      npoints_prolongated +=
          component_prolongated_sizes.at(level).at(component);
    }
  }
  npoints_local = npoints;
  npoints_prolongated_local = npoints_prolongated;

  // Global exclusive prefix sum
  int npoints_offset = 0;
  MPI_Exscan(&npoints_local, &npoints_offset, 1, MPI_INT, MPI_SUM,
             MPI_COMM_WORLD);
  MPI_Allreduce(&npoints_local, &npoints_global, 1, MPI_INT, MPI_SUM,
                MPI_COMM_WORLD);
  int npoints_prolongated_offset = 0;
  MPI_Exscan(&npoints_prolongated_local, &npoints_prolongated_offset, 1,
             MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&npoints_prolongated_local, &npoints_prolongated_global, 1,
                MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  {
    const auto writes = open_writes(std::array{gi_idx}, CarpetX::make_valid_all(),
                                    "PDESolver::enumerate_points");
    CarpetX::active_window().loop_parallel(
        [&](const CarpetX::level_t lvl, const int index, const int component,
            const cGH *restrict const cctkGH) {
          const int level = lvl.level;
          const Loop::GF3D2layout layout1(cctkGH, indextype);
          const Loop::GF3D2<CCTK_REAL> gf_idx(
              layout1,
              static_cast<CCTK_REAL *>(CCTK_VarDataPtrI(cctkGH, tl, vn_idx)));
          const Loop::GF3D2<const CCTK_REAL> gf_pt(
              layout1, static_cast<const CCTK_REAL *>(
                           CCTK_VarDataPtrI(cctkGH, tl, vn_pt)));
          const Loop::GridDescBase grid(cctkGH);
          int idx = npoints_offset + component_offsets.at(level).at(component);
          int idx_prolongated =
              npoints_prolongated_offset +
              component_prolongated_offsets.at(level).at(component);
          grid.loop_all<0, 0, 0>(
              grid.nghostzones, [&](const Loop::PointDesc &p) ARITH_INLINE {
                switch (int(gf_pt(p.I))) {
                case int(point_type_t::intr):
                  // solve for this point
                  gf_idx(p.I) = idx++;
                  break;
                case int(point_type_t::bdry):
                  gf_idx(p.I) = -1;
                  break;
                case int(point_type_t::rest):
                case int(point_type_t::sync):
                  // undefined (will be set later)
                  gf_idx(p.I) = -2;
                  break;
                case int(point_type_t::prol):
                  gf_idx(p.I) = prolongation_index_offset + idx_prolongated++;
                  break;
                default:
                  assert(0);
                }
              });
          assert(idx == npoints_offset +
                            component_offsets.at(level).at(component) +
                            component_sizes.at(level).at(component));
          if (!(idx_prolongated ==
                npoints_prolongated_offset +
                    component_prolongated_offsets.at(level).at(component) +
                    component_prolongated_sizes.at(level).at(component))) {
            std::cout << "level=" << level << "\n";
            std::cout << "component=" << component << "\n";
            std::cout << "idx_prolongated=" << idx_prolongated << "\n";
            std::cout << "npoints_prolongated_offset="
                      << npoints_prolongated_offset << "\n";
            std::cout << "component_prolongated_offsets="
                      << component_prolongated_offsets.at(level).at(component)
                      << "\n";
            std::cout << "component_prolongated_sizes="
                      << component_prolongated_sizes.at(level).at(component)
                      << "\n";
          }
          assert(idx_prolongated ==
                 npoints_prolongated_offset +
                     component_prolongated_offsets.at(level).at(component) +
                     component_prolongated_sizes.at(level).at(component));
        });
  }

  CarpetX::restrict_no_poison(CarpetX::active_window(), std::array{gi_idx});

  CarpetX::sync_ghosts_only(CarpetX::active_window(), std::array{gi_idx}, tl);

  {
    const auto writes = open_writes(std::array{gi_idx}, CarpetX::make_valid_all(),
                                    "PDESolver::enumerate_points");
    CarpetX::active_window().loop_parallel(
        [&](const CarpetX::level_t lvl, const int index, const int component,
            const cGH *restrict const cctkGH) {
          const Loop::GF3D2layout layout1(cctkGH, indextype);
          const Loop::GF3D2<const CCTK_REAL> gf_pt(
              layout1, static_cast<const CCTK_REAL *>(
                           CCTK_VarDataPtrI(cctkGH, tl, vn_pt)));
          const Loop::GF3D2<const CCTK_REAL> gf_idx(
              layout1, static_cast<const CCTK_REAL *>(
                           CCTK_VarDataPtrI(cctkGH, tl, vn_idx)));
          const Loop::GridDescBase grid(cctkGH);
          grid.loop_all<0, 0, 0>(
              grid.nghostzones, [&](const Loop::PointDesc &p) ARITH_INLINE {
                switch (int(gf_pt(p.I))) {
                case int(point_type_t::intr):
                case int(point_type_t::sync):
                case int(point_type_t::rest):
                  assert(gf_idx(p.I) >= 0 &&
                         gf_idx(p.I) < prolongation_index_offset);
                  break;
                case int(point_type_t::bdry):
                  assert(gf_idx(p.I) == -1);
                  break;
                case int(point_type_t::prol):
                  assert(gf_idx(p.I) >= prolongation_index_offset);
                  break;
                default:
                  assert(0);
                }
              });
        });
  }

  {
    std::vector<
        std::vector<std::vector<std::tuple<int, Arith::vect<int, 3> > > > >
        locations(nlevels);
    for (int level = 0; level < nlevels; ++level)
      locations.at(level).resize(level_sizes.at(level));
    CarpetX::active_window().loop_parallel(
        [&](const CarpetX::level_t lvl, const int index, const int component,
            const cGH *restrict const cctkGH) {
          const int level = lvl.level;
          if (level == 0)
            assert(component_prolongated_sizes.at(level).at(component) == 0);
          if (component_prolongated_sizes.at(level).at(component) == 0)
            return;
          const auto &mfab = CarpetX::data(CarpetX::group_t{lvl, gi_idx}, tl);
          const auto &fabbox = mfab.fabbox(index);
          const Arith::vect<int, 3> amrex_origin{
              fabbox.smallEnd(0), fabbox.smallEnd(1), fabbox.smallEnd(2)};
          const Loop::GF3D2layout layout1(cctkGH, indextype);
          const Loop::GF3D2<const CCTK_REAL> gf_pt(
              layout1, static_cast<const CCTK_REAL *>(
                           CCTK_VarDataPtrI(cctkGH, tl, vn_pt)));
          const Loop::GF3D2<const CCTK_REAL> gf_idx(
              layout1, static_cast<const CCTK_REAL *>(
                           CCTK_VarDataPtrI(cctkGH, tl, vn_idx)));
          const Loop::GridDescBase grid(cctkGH);
          const auto offset = amrex_origin;

          std::vector<std::tuple<int, Arith::vect<int, 3> > > locs;
          grid.loop_all<0, 0, 0>(
              grid.nghostzones, [&](const Loop::PointDesc &p) ARITH_INLINE {
                if (int(gf_pt(p.I)) == int(point_type_t::prol)) {
                  const int fL = level;
                  const Arith::vect<int, 3> fI = p.I + offset;
                  const int cL = fL - 1;
                  const Arith::vect<int, 3> cI0 = div_floor(fI, 2);
                  const Arith::vect<int, 3> cIm = mod_floor(fI, 2);
                  // linear interpolation
                  for (int k = 0; k < cIm[2] + 1; ++k) {
                    for (int j = 0; j < cIm[1] + 1; ++j) {
                      for (int i = 0; i < cIm[0] + 1; ++i) {
                        const Arith::vect<int, 3> cI =
                            cI0 + Arith::vect<int, 3>{i, j, k};
                        locs.emplace_back(cL, cI);
                      }
                    }
                  }
                }
              });
          locations.at(level).at(component) = std::move(locs);
        });

    std::vector<std::vector<std::vector<int> > > indices(nlevels);
    for (int level = 0; level < nlevels; ++level) {
      indices.at(level).resize(level_sizes.at(level));
      for (int component = 0; component < level_sizes.at(level); ++component) {
        indices.at(level).at(component).resize(
            locations.at(level).at(component).size(), 0x80000000U);
      }
    }
    CarpetX::active_window().loop_parallel(
        [&](const CarpetX::level_t lvl, const int index, const int component,
            const cGH *restrict const cctkGH) {
          const int level = lvl.level;
          if (level == CarpetX::num_levels(lvl.patch) - 1)
            return;
          const auto &mfab = CarpetX::data(CarpetX::group_t{lvl, gi_idx}, tl);
          const auto &fabbox = mfab.fabbox(index);
          const Arith::vect<int, 3> amrex_origin{
              fabbox.smallEnd(0), fabbox.smallEnd(1), fabbox.smallEnd(2)};
          const Loop::GF3D2layout layout1(cctkGH, indextype);
          const Loop::GF3D2<const CCTK_REAL> gf_pt(
              layout1, static_cast<const CCTK_REAL *>(
                           CCTK_VarDataPtrI(cctkGH, tl, vn_pt)));
          const Loop::GF3D2<const CCTK_REAL> gf_idx(
              layout1, static_cast<const CCTK_REAL *>(
                           CCTK_VarDataPtrI(cctkGH, tl, vn_idx)));
          const Loop::GridDescBase grid(cctkGH);
          Arith::vect<int, 3> imin, imax;
          grid.box_all<0, 0, 0>(grid.nghostzones, imin, imax);
          const auto offset = amrex_origin;

          for (int level1 = 0; level1 < nlevels; ++level1) {
            for (int component1 = 0; component1 < level_sizes.at(level1);
                 ++component1) {
              const auto &locs = locations.at(level1).at(component1);
              auto &idxs = indices.at(level1).at(component1);
              for (int n = 0; n < int(locs.size()); ++n) {
                const auto &loc = locs.at(n);
                const auto &cL = std::get<0>(loc);
                const auto &cI = std::get<1>(loc);
                if (cL == level) {
                  const Arith::vect<int, 3> I = cI - offset;
                  if (all(imin <= I && I < imax)) {
                    const int idx = int(gf_idx(I));
                    // We require that a prolongation source is not prolongated
                    // itself. (We could extend the algorithm to handle this.)
                    assert(idx < prolongation_index_offset);
                    int &idxn = idxs.at(n);
#pragma omp atomic write
                    idxn = idx;
                  }
                }
              }
            }
          }
        });
    for (int level = 0; level < nlevels; ++level) {
      for (int component = 0; component < level_sizes.at(level); ++component) {
        for (const auto &idx : indices.at(level).at(component))
          assert(idx != int(0x80000000U));
      }
    }

    std::vector<std::vector<std::vector<std::tuple<int, int, CCTK_REAL> > > >
        Jpvalss(nlevels);
    for (int level = 0; level < nlevels; ++level)
      Jpvalss.at(level).resize(level_sizes.at(level));
    CarpetX::active_window().loop_parallel(
        [&](const CarpetX::level_t lvl, const int index, const int component,
            const cGH *restrict const cctkGH) {
          const int level = lvl.level;
          if (level == 0)
            assert(component_prolongated_sizes.at(level).at(component) == 0);
          if (component_prolongated_sizes.at(level).at(component) == 0)
            return;
          const auto &mfab = CarpetX::data(CarpetX::group_t{lvl, gi_idx}, tl);
          const auto &fabbox = mfab.fabbox(index);
          const Arith::vect<int, 3> amrex_origin{
              fabbox.smallEnd(0), fabbox.smallEnd(1), fabbox.smallEnd(2)};
          const Loop::GF3D2layout layout1(cctkGH, indextype);
          const Loop::GF3D2<const CCTK_REAL> gf_pt(
              layout1, static_cast<const CCTK_REAL *>(
                           CCTK_VarDataPtrI(cctkGH, tl, vn_pt)));
          const Loop::GF3D2<const CCTK_REAL> gf_idx(
              layout1, static_cast<const CCTK_REAL *>(
                           CCTK_VarDataPtrI(cctkGH, tl, vn_idx)));
          const Loop::GridDescBase grid(cctkGH);
          const auto offset = amrex_origin;

          const auto &locs = locations.at(level).at(component);
          const auto &idxs = indices.at(level).at(component);
          assert(idxs.size() == locs.size());
          int n = 0;
          std::vector<std::tuple<int, int, CCTK_REAL> > Jpvals;
          Jpvals.reserve(locs.size());
          grid.loop_all<0, 0, 0>(
              grid.nghostzones, [&](const Loop::PointDesc &p) ARITH_INLINE {
                if (int(gf_pt(p.I)) == int(point_type_t::prol)) {
                  assert(n < int(idxs.size()));
                  const int idx = int(gf_idx(p.I)) - prolongation_index_offset;
                  const int cL = level - 1;
                  const Arith::vect<int, 3> cIm = mod_floor(p.I + offset, 2);
                  const Arith::vect<int, 3> cI0 = div_floor(p.I + offset, 2);
                  for (int k = 0; k < cIm[2] + 1; ++k) {
                    for (int j = 0; j < cIm[1] + 1; ++j) {
                      for (int i = 0; i < cIm[0] + 1; ++i) {
                        const Arith::vect<int, 3> cI =
                            cI0 + Arith::vect<int, 3>{i, j, k};
                        assert(std::get<0>(locs.at(n)) == cL);
                        assert(all(std::get<1>(locs.at(n)) == cI));
                        const int row = idx;
                        const int col = idxs.at(n);
                        // Ignore boundary points
                        if (col >= 0) {
                          // We assume that a prolongation source is not
                          // prolongated itself. (We could extend the algorithm
                          // to handle this.)
                          assert(col < prolongation_index_offset);
                          const CCTK_REAL val =
                              1.0 /
                              ((cIm[0] + 1) * (cIm[1] + 1) * (cIm[2] + 1));
                          Jpvals.emplace_back(row, col, val);
                        }
                        ++n;
                      }
                    }
                  }
                }
              });
          assert(n == int(locs.size()));
          Jpvalss.at(level).at(component) = std::move(Jpvals);
        });

    // Convert Jpvals into sparse matrix
    Jp = csr_t(npoints_prolongated_global, npoints_global, Jpvalss);
  }
}

void copy_Cactus_to_PETSc(
    Vec vec, const std::vector<int> &varinds,
    const std::vector<std::vector<int> > &component_offsets,
    const std::vector<std::vector<int> > &component_sizes) {
  PetscErrorCode ierr;

  int myproc;
  MPI_Comm_rank(MPI_COMM_WORLD, &myproc);

  const int nvars = varinds.size();
  std::vector<CarpetX::gi_t> gis;
  std::vector<int> vis;
  gis.reserve(nvars);
  vis.reserve(nvars);
  for (int vn : varinds) {
    assert(vn >= 0);
    const CarpetX::gi_t gi{CCTK_GroupIndexFromVarI(vn)};
    assert(gi.index() >= 0);
    gis.push_back(gi);
    const int v0 = CCTK_FirstVarIndexI(gi.index());
    assert(v0 >= 0);
    const int vi = vn - v0;
    assert(vi >= 0);
    vis.push_back(vi);
  }
  static const int vn_pt = CCTK_VarIndex("PDESolvers::point_type");
  assert(vn_pt >= 0);
  static const int gi_pt = CCTK_GroupIndexFromVarI(vn_pt);
  assert(gi_pt >= 0);
  static const int v0_pt = CCTK_FirstVarIndexI(gi_pt);
  assert(v0_pt >= 0);
  static const int vi_pt = vn_pt - v0_pt;
  assert(vi_pt >= 0);

  // Get vector from PETSc
  CCTK_REAL *vec_ptr;
  ierr = VecGetArray(vec, &vec_ptr);
  assert(!ierr);
  CCTK_REAL *restrict const petsc_ptr = vec_ptr;

  CarpetX::active_window().loop_serially([&](const CarpetX::level_t level) {
    for (int n = 0; n < nvars; ++n)
      CarpetX::require_valid(CarpetX::group_t{level, gis.at(n)}, tl,
                             CarpetX::make_valid_int(),
                             "PDESolver::copy_Cactus_to_PETSc");
  });
  CarpetX::active_window().loop_parallel([&](const CarpetX::level_t lvl,
                                             const int index,
                                             const int component,
                                             const cGH *restrict const cctkGH) {
    const int level = lvl.level;
    const Loop::GF3D2layout layout1(cctkGH, indextype);
    const Loop::GridDescBase grid(cctkGH);
    const Loop::GF3D2<const CCTK_REAL> gf_pt(
        layout1,
        static_cast<const CCTK_REAL *>(CCTK_VarDataPtrI(cctkGH, tl, vn_pt)));
    std::vector<Loop::GF3D2<const CCTK_REAL> > gfs;
    gfs.reserve(nvars);
    for (int n = 0; n < nvars; ++n)
      gfs.emplace_back(layout1, static_cast<const CCTK_REAL *>(CCTK_VarDataPtrI(
                                    cctkGH, tl, varinds.at(n))));
    const int component_offset = component_offsets.at(level).at(component);
    int nelems = 0;
    grid.loop_all<0, 0, 0>(
        grid.nghostzones, [&](const Loop::PointDesc &p) ARITH_INLINE {
          switch (int(gf_pt(p.I))) {
          case int(point_type_t::intr):
            for (int n = 0; n < nvars; ++n)
              petsc_ptr[nvars * component_offset + nelems++] = gfs.at(n)(p.I);
            break;
          case int(point_type_t::bdry):
          case int(point_type_t::rest):
          case int(point_type_t::sync):
          case int(point_type_t::prol):
            // ignore this point
            break;
          default:
            assert(0);
          }
        });
    assert(nelems == nvars * component_sizes.at(level).at(component));
  });

  ierr = VecRestoreArray(vec, &vec_ptr);
  assert(!ierr);
}

void copy_PETSc_to_Cactus(
    Vec vec, const std::vector<int> &varinds,
    const std::vector<std::vector<int> > &component_offsets,
    const std::vector<std::vector<int> > &component_sizes) {
  PetscErrorCode ierr;

  int myproc;
  MPI_Comm_rank(MPI_COMM_WORLD, &myproc);

  const int nvars = varinds.size();
  std::vector<CarpetX::gi_t> gis;
  std::vector<int> vis;
  gis.reserve(nvars);
  vis.reserve(nvars);
  for (int vn : varinds) {
    assert(vn >= 0);
    const CarpetX::gi_t gi{CCTK_GroupIndexFromVarI(vn)};
    assert(gi.index() >= 0);
    gis.push_back(gi);
    const int v0 = CCTK_FirstVarIndexI(gi.index());
    assert(v0 >= 0);
    const int vi = vn - v0;
    assert(vi >= 0);
    vis.push_back(vi);
  }
  static const int vn_pt = CCTK_VarIndex("PDESolvers::point_type");
  assert(vn_pt >= 0);
  static const int gi_pt = CCTK_GroupIndexFromVarI(vn_pt);
  assert(gi_pt >= 0);
  static const int v0_pt = CCTK_FirstVarIndexI(gi_pt);
  assert(v0_pt >= 0);
  static const int vi_pt = vn_pt - v0_pt;
  assert(vi_pt >= 0);

  // Get vector from PETSc
  const CCTK_REAL *vec_ptr;
  ierr = VecGetArrayRead(vec, &vec_ptr);
  assert(!ierr);
  const CCTK_REAL *restrict const petsc_ptr = vec_ptr;

  {
    const auto writes = open_writes(gis, CarpetX::make_valid_int(),
                                    "PDESolver::copy_PETSc_to_Cactus");
    CarpetX::active_window().loop_parallel(
        [&](const CarpetX::level_t lvl, const int index, const int component,
            const cGH *restrict const cctkGH) {
          const int level = lvl.level;
          const Loop::GF3D2layout layout1(cctkGH, indextype);
          const Loop::GridDescBase grid(cctkGH);
          const Loop::GF3D2<const CCTK_REAL> gf_pt(
              layout1, static_cast<const CCTK_REAL *>(
                           CCTK_VarDataPtrI(cctkGH, tl, vn_pt)));
          std::vector<Loop::GF3D2<CCTK_REAL> > gfs;
          gfs.reserve(nvars);
          for (int n = 0; n < nvars; ++n)
            gfs.emplace_back(layout1, static_cast<CCTK_REAL *>(CCTK_VarDataPtrI(
                                          cctkGH, tl, varinds.at(n))));
          const int component_offset =
              component_offsets.at(level).at(component);
          int nelems = 0;
          grid.loop_all<0, 0, 0>(
              grid.nghostzones, [&](const Loop::PointDesc &p) ARITH_INLINE {
                switch (int(gf_pt(p.I))) {
                case int(point_type_t::intr):
                  for (int n = 0; n < nvars; ++n)
                    gfs.at(n)(p.I) =
                        petsc_ptr[nvars * component_offset + nelems++];
                  break;
                case int(point_type_t::bdry):
                case int(point_type_t::rest):
                case int(point_type_t::sync):
                case int(point_type_t::prol):
                  // ignore this point
                  break;
                default:
                  assert(0);
                }
              });
          assert(nelems == nvars * component_sizes.at(level).at(component));
        });
  }

  ierr = VecRestoreArrayRead(vec, &vec_ptr);
  assert(!ierr);
}

////////////////////////////////////////////////////////////////////////////////

extern "C" void PDESolvers_Setup(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTS_PDESolvers_Setup;
  DECLARE_CCTK_PARAMETERS;

  // Parse Cactus parameter into PETSc command line options
  /*static*/ std::vector<std::string> args;
  args.push_back("cactus");
  std::istringstream buf(petsc_options);
  std::copy(std::istream_iterator<std::string>(buf),
            std::istream_iterator<std::string>(), std::back_inserter(args));

  /*static*/ std::vector<char *> argptrs;
  for (auto &arg : args)
    argptrs.push_back(arg.data());
  argptrs.push_back(nullptr);

  /*static*/ int argc = args.size();
  /*static*/ char **argv = argptrs.data();
  PetscErrorCode ierr = PetscInitialize(&argc, &argv, NULL, NULL);
  assert(!ierr);
}

namespace {
PetscErrorCode FormFunction(SNES snes, Vec x, Vec f, void *fun) {
  return (
      *static_cast<std::function<PetscErrorCode(SNES snes, Vec x, Vec f)> *>(
          fun))(snes, x, f);
}

PetscErrorCode FormJacobian(SNES snes, Vec x, Mat J, Mat B, void *fun) {
  return (*static_cast<
          std::function<PetscErrorCode(SNES snes, Vec x, Mat J, Mat B)> *>(
      fun))(snes, x, J, B);
}
} // namespace

extern "C" void PDESolvers_Solve(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTS_PDESolvers_Solve;
  DECLARE_CCTK_PARAMETERS;

  CCTK_VINFO("PDESolvers_Solve: Solving on %d levels", CarpetX::num_levels(0));

  PetscErrorCode ierr;

  // Grid layout

  define_point_type();

  int npoints_local, npoints_global;
  int npoints_prolongated_local, npoints_prolongated_global;
  // component_offsets are process local
  std::vector<std::vector<int> > component_offsets, component_sizes;
  std::vector<std::vector<int> > component_prolongated_offsets,
      component_prolongated_sizes;
  csr_t Jp;
  enumerate_points(npoints_local, npoints_global, component_offsets,
                   component_sizes, npoints_prolongated_local,
                   npoints_prolongated_global, component_prolongated_offsets,
                   component_prolongated_sizes, Jp);

  // TODO: fix this
  const std::vector<int> solinds{CCTK_VarIndex("PoissonX::sol")};
  const std::vector<int> resinds{CCTK_VarIndex("PoissonX::res")};
  const int nvars = solinds.size();
  assert(int(resinds.size()) == nvars);

  // Nonlinear solver

  SNES snes;
  ierr = SNESCreate(PETSC_COMM_WORLD, &snes);
  assert(!ierr);
  ierr = SNESSetFromOptions(snes);
  assert(!ierr);
  ierr = SNESSetAlwaysComputesFinalResidual(snes, PETSC_TRUE);
  assert(!ierr);

  // State vector and evaluation function

  Vec r;
  ierr = VecCreateMPI(PETSC_COMM_WORLD, nvars * npoints_local,
                      nvars * npoints_global, &r);
  assert(!ierr);
  ierr = VecSetFromOptions(r);
  assert(!ierr);

  // Row equilibration: a left scaling applied to both the residual and the
  // Jacobian. It is the identity until it is computed below, just before the
  // solve; the lambdas capture it by reference and apply whatever it holds at
  // the time they run.
  Vec rowscale;
  ierr = VecDuplicate(r, &rowscale);
  assert(!ierr);
  ierr = VecSet(rowscale, 1.0);
  assert(!ierr);

  std::function<PetscErrorCode(SNES snes, Vec x, Vec f)> evalf =
      [&](SNES snes, Vec x, Vec f) {
        copy_PETSc_to_Cactus(x, solinds, component_offsets, component_sizes);
        CallScheduleGroup(cctkGH, "PDESolvers_Residual");
        copy_Cactus_to_PETSc(f, resinds, component_offsets, component_sizes);
        const PetscErrorCode ierr = VecPointwiseMult(f, rowscale, f);
        assert(!ierr);
        return 0;
      };
  ierr = SNESSetFunction(snes, r, FormFunction, &evalf);
  assert(!ierr);

  // Matrix and Jacobian evaluation function

  // The matrix is deliberately created without preallocation:
  // `jacobians_t::define_matrix` preallocates it from the entries it is about
  // to insert, once it knows them.
  Mat J;
  ierr = MatCreate(PETSC_COMM_WORLD, &J);
  assert(!ierr);
  ierr = MatSetSizes(J, nvars * npoints_local, nvars * npoints_local,
                     nvars * npoints_global, nvars * npoints_global);
  assert(!ierr);
  ierr = MatSetType(J, MATAIJ);
  assert(!ierr);
  ierr = MatSetFromOptions(J);
  assert(!ierr);
  jacobians = std::make_optional<jacobians_t>();

  std::function<PetscErrorCode(SNES snes, Vec x, Mat J, Mat B)> evalJ =
      [&](SNES snes, Vec x, Mat J, Mat B) {
        copy_PETSc_to_Cactus(x, solinds, component_offsets, component_sizes);
        CallScheduleGroup(cctkGH, "PDESolvers_Jacobian");
        jacobians->define_matrix(Jp, J);
        jacobians->clear();
        {
          const PetscErrorCode ierr = MatDiagonalScale(J, rowscale, NULL);
          assert(!ierr);
        }
        if (false) {
          MatInfo info;
          PetscErrorCode ierr;
          ierr = MatGetInfo(J, MAT_GLOBAL_SUM, &info);
          assert(!ierr);
          CCTK_VINFO("Jacobian info: nz_allocated=%g nz_used=%g nz_unneeded=%g "
                     "memory=%g",
                     double(info.nz_allocated), double(info.nz_used),
                     double(info.nz_unneeded), double(info.memory));
        }
        return 0;
      };
  ierr = SNESSetJacobian(snes, J, J, FormJacobian, &evalJ);
  assert(!ierr);

  {
    double atol, rtol, stol;
    int maxit, maxf;
    ierr = SNESGetTolerances(snes, &atol, &rtol, &stol, &maxit, &maxf);
    assert(!ierr);
    CCTK_VINFO("Settings: atol=%g, rtol=%g, stol=%g, maxit=%d, maxf=%d", atol,
               rtol, stol, maxit, maxf);
  }

  // Initialize solution

  Vec x;
  ierr = VecDuplicate(r, &x);
  assert(!ierr);
  copy_Cactus_to_PETSc(x, solinds, component_offsets, component_sizes);

  // Row equilibration
  //
  // The rows of the flat matrix come from several refinement levels. For a
  // second order operator their magnitudes differ by a factor 4^level, and
  // algebraic multigrid then loses its resolution independence: AMG relies on
  // the near null space (the constants) leaving a small residual, which
  // requires the rows to be scaled comparably. Left scaling by 1/|J_ii|
  // preserves each row sum and restores this. (PETSc's -ksp_diagonal_scale
  // scales symmetrically, which does not preserve row sums and makes AMG
  // converge worse; do not use it here.)
  //
  // This is a left preconditioner: it scales the residual and the Jacobian
  // identically, so the Newton step and the solution are unchanged. It does
  // change the residual norms that SNES and KSP test their tolerances against.
  //
  // The scaling is taken from the Jacobian at the initial guess and held fixed
  // for the whole solve, so that residual and Jacobian can never be scaled
  // inconsistently. Evaluating the Jacobian here costs one extra assembly;
  // `define_matrix` zeroes the matrix first, so this is idempotent.
  if (row_equilibration) {
    ierr = evalJ(snes, x, J, J);
    assert(!ierr);
    ierr = MatGetDiagonal(J, rowscale);
    assert(!ierr);
    ierr = VecAbs(rowscale);
    assert(!ierr);
    PetscInt nlocal;
    ierr = VecGetLocalSize(rowscale, &nlocal);
    assert(!ierr);
    PetscScalar *rowscale_ptr;
    ierr = VecGetArray(rowscale, &rowscale_ptr);
    assert(!ierr);
    for (PetscInt i = 0; i < nlocal; ++i)
      rowscale_ptr[i] =
          rowscale_ptr[i] > 0 ? 1.0 / rowscale_ptr[i] : CCTK_REAL(1);
    ierr = VecRestoreArray(rowscale, &rowscale_ptr);
    assert(!ierr);
  }

  // Solve

  CCTK_INFO("Solve");
  ierr = SNESSolve(snes, NULL, x);
  assert(!ierr);

  {
    SNESConvergedReason reason;
    ierr = SNESGetConvergedReason(snes, &reason);
    assert(!ierr);
    switch (reason) {
    case SNES_CONVERGED_FNORM_ABS:
    case SNES_CONVERGED_FNORM_RELATIVE:
    case SNES_CONVERGED_SNORM_RELATIVE:
    case SNES_CONVERGED_ITS:
      // SNES converged; do nothing
      CCTK_INFO("SNES iterations converged");
      break;
    case SNES_DIVERGED_FUNCTION_DOMAIN:
    case SNES_DIVERGED_FUNCTION_COUNT:
    case SNES_DIVERGED_LINEAR_SOLVE:
    case SNES_DIVERGED_FNORM_NAN:
    case SNES_DIVERGED_MAX_IT:
    case SNES_DIVERGED_LINE_SEARCH:
    case SNES_DIVERGED_INNER:
    case SNES_DIVERGED_LOCAL_MIN:
    case SNES_DIVERGED_DTOL:
    case SNES_DIVERGED_JACOBIAN_DOMAIN:
    case SNES_DIVERGED_TR_DELTA:
    case SNES_CONVERGED_ITERATING:
      CCTK_WARN(CCTK_WARN_ALERT, "****************************************");
      CCTK_WARN(CCTK_WARN_ALERT, "SNES iterations diverged");
      CCTK_WARN(CCTK_WARN_ALERT, "****************************************");
      break;
    default:
      CCTK_ERROR("Unknown SNES converged reason");
    }
  }

  {
    int niters;
    ierr = SNESGetIterationNumber(snes, &niters);
    assert(!ierr);
    int liters;
    ierr = SNESGetLinearSolveIterations(snes, &liters);
    assert(!ierr);
    CCTK_VINFO("Iterations: %d Newton, %d linear", niters, liters);
  }

  // Extract solution

  copy_PETSc_to_Cactus(x, solinds, component_offsets, component_sizes);
  CallScheduleGroup(cctkGH, "PDESolvers_Residual");
  {
    const int nvars = resinds.size();
    std::vector<CarpetX::gi_t> gis;
    std::vector<int> vis;
    gis.reserve(nvars);
    vis.reserve(nvars);
    for (int vn : resinds) {
      assert(vn >= 0);
      const CarpetX::gi_t gi{CCTK_GroupIndexFromVarI(vn)};
      assert(gi.index() >= 0);
      gis.push_back(gi);
      const int v0 = CCTK_FirstVarIndexI(gi.index());
      assert(v0 >= 0);
      const int vi = vn - v0;
      assert(vi >= 0);
      vis.push_back(vi);
    }
    const CarpetX::reduction<CCTK_REAL, 3> res_norm =
        CarpetX::reduce(gis.at(0), vis.at(0), tl);
    std::ostringstream buf;
    buf << res_norm;
    CCTK_VINFO("Residual norm: %s", buf.str().c_str());
  }

  // Clean up

  jacobians.reset();
  VecDestroy(&x);
  VecDestroy(&r);
  VecDestroy(&rowscale);
  MatDestroy(&J);
  SNESDestroy(&snes);
  CCTK_INFO("Done.");
}

extern "C" void PDESolvers_Shutdown(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTS_PDESolvers_Shutdown;
  PetscFinalize();
}

} // namespace PDESolvers
