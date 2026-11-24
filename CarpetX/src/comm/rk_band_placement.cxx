#include "comm/rk_band_placement_inside.hxx"

#include <AMReX_FabArrayBase.H>

#include <cassert>

namespace CarpetX {

rk_band_layout_t rk_band_layout_of(const amrex::FabArrayBase::FPinfo &fp) {
  assert(!fp.ba_crse_patch.empty());
  assert(fp.ba_crse_patch.size() == fp.ba_fine_patch.size());
  assert(fp.fact_crse_patch);
  assert(fp.fact_fine_patch);
  const amrex::DistributionMapping &owner = fp.dm_patch;
  assert(owner.size() == fp.ba_fine_patch.size());
  return {fp.ba_crse_patch, fp.ba_fine_patch, owner, *fp.fact_crse_patch,
          *fp.fact_fine_patch};
}

} // namespace CarpetX
