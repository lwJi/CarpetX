#include "checkpoint.hxx"

#include "driver.hxx"

#include <cctk.h>
#include <cctk_Parameters.h>

#include <AMReX_BoxArray.H>
#include <AMReX_BoxList.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Geometry.H>
#include <AMReX_Vector.H>

#include <cassert>
#include <optional>
#include <vector>

namespace CarpetX {

bool finer_level_is_behind(const checkpoint_record_t &record,
                           const level_t level) {
  const std::vector<level_record_t> &levels =
      record.levels.at(level.patch().index());
  const std::optional<rat64> &self = levels.at(level.index()).iteration;
  if (!self)
    return false;
  for (int finer = level.index() + 1; finer < int(levels.size()); ++finer) {
    const std::optional<rat64> &finer_iteration = levels.at(finer).iteration;
    if (finer_iteration && *finer_iteration < *self)
      return true;
  }
  return false;
}

void apply_structure(cGH *const cctkGH, const checkpoint_record_t &record) {
  DECLARE_CCTK_PARAMETERS;

  const int npatches = record.levels.size();
  if (npatches != ghext->num_patches())
    CCTK_VERROR(
        "Wrong number of patches: Expected %d, found %d in the checkpoint file",
        ghext->num_patches(), npatches);

  cctkGH->cctk_iteration = record.iteration;
  cctkGH->cctk_time = record.time;

  for (auto &patchdata : ghext->patches()) {
    const std::vector<level_record_t> &levels =
        record.levels.at(patchdata.patch.index());
    const int nlevels = levels.size();
    assert(nlevels > 0);
    CactusAmrCore &amrcore = *patchdata.amrcore;
    amrcore.SetFinestLevel(nlevels - 1);

    for (int lev = 0; lev < nlevels; ++lev) {
      const level_t level{patchdata.patch, lev};

      if (lev > 0) {
        amrex::Geometry geom = amrcore.Geom(lev - 1);
        geom.refine({2, 2, 2});
        amrcore.SetGeometry(lev, geom);
      }

      const std::vector<amrex::Box> &boxes = levels.at(lev).boxes;
      amrex::BoxArray ba(amrex::BoxList(
          amrex::Vector<amrex::Box>(boxes.begin(), boxes.end())));
      if (rechop_on_recovery)
        ba = amrcore.RechopLevel(level, ba);
      amrcore.SetBoxArray(lev, ba);

      const amrex::DistributionMapping dm(ba);
      amrcore.SetDistributionMap(lev, dm);

      amrcore.make_level(level, ba, dm, "Recovering");
    }
  }
}

} // namespace CarpetX
