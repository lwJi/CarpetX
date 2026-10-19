#include "boundaries.hxx"
#include "boundaries_impl.hxx"

#include <vect.hxx>

#include <AMReX.H>

#include <algorithm>
#include <cmath>
#include <sstream>
#include <type_traits>
#include <vector>

namespace CarpetX {

BoundaryCondition::BoundaryCondition(
    const GHExt::PatchData::LevelData::GroupData &groupdata,
    amrex::FArrayBox &dest)
    : groupdata(groupdata), patchdata(ghext->patchdata.at(groupdata.patch)),
      geom(patchdata.amrcore->Geom(groupdata.level)), dest(dest),
      imin{geom.Domain().smallEnd(0), geom.Domain().smallEnd(1),
           geom.Domain().smallEnd(2)},
      imax{geom.Domain().bigEnd(0) + 1 + !groupdata.indextype[0],
           geom.Domain().bigEnd(1) + 1 + !groupdata.indextype[1],
           geom.Domain().bigEnd(2) + 1 + !groupdata.indextype[2]},
      xmin{geom.ProbLo(0) + CCTK_REAL(0.5) * groupdata.indextype[0] - 1,
           geom.ProbLo(1) + CCTK_REAL(0.5) * groupdata.indextype[1] - 1,
           geom.ProbLo(2) + CCTK_REAL(0.5) * groupdata.indextype[2] - 1},
      xmax{geom.ProbHi(0) - CCTK_REAL(0.5) * groupdata.indextype[0] + 1,
           geom.ProbHi(1) - CCTK_REAL(0.5) * groupdata.indextype[1] + 1,
           geom.ProbHi(2) - CCTK_REAL(0.5) * groupdata.indextype[2] + 1},
      dx{geom.CellSize(0), geom.CellSize(1), geom.CellSize(2)},
      dmin{dest.box().smallEnd(0), dest.box().smallEnd(1),
           dest.box().smallEnd(2)},
      dmax{dest.box().bigEnd(0) + 1, dest.box().bigEnd(1) + 1,
           dest.box().bigEnd(2) + 1},
      layout(dmin, dmax), destptr(dest.dataPtr()) {
  for (int d = 0; d < dim; ++d)
    assert((dest.box().type(d) == amrex::IndexType::CELL) ==
           groupdata.indextype[d]);

  for (int d = 0; d < dim; ++d) {
    assert(dest.box().smallEnd(d) <= dmin[d]);
    assert(dest.box().bigEnd(d) + 1 >= dmax[d]);
  }
}

void BoundaryCondition::apply() const {
  apply_on_face<NEG, NEG, NEG>();
  apply_on_face<INT, NEG, NEG>();
  apply_on_face<POS, NEG, NEG>();
  apply_on_face<NEG, INT, NEG>();
  apply_on_face<INT, INT, NEG>();
  apply_on_face<POS, INT, NEG>();
  apply_on_face<NEG, POS, NEG>();
  apply_on_face<INT, POS, NEG>();
  apply_on_face<POS, POS, NEG>();

  apply_on_face<NEG, NEG, INT>();
  apply_on_face<INT, NEG, INT>();
  apply_on_face<POS, NEG, INT>();
  apply_on_face<NEG, INT, INT>();
  apply_on_face<POS, INT, INT>();
  apply_on_face<NEG, POS, INT>();
  apply_on_face<INT, POS, INT>();
  apply_on_face<POS, POS, INT>();

  apply_on_face<NEG, NEG, POS>();
  apply_on_face<INT, NEG, POS>();
  apply_on_face<POS, NEG, POS>();
  apply_on_face<NEG, INT, POS>();
  apply_on_face<INT, INT, POS>();
  apply_on_face<POS, INT, POS>();
  apply_on_face<NEG, POS, POS>();
  apply_on_face<INT, POS, POS>();
  apply_on_face<POS, POS, POS>();
}

} // namespace CarpetX
