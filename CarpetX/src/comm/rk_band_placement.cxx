#include "comm/rk_band_placement_inside.hxx"

#include "comm/prolongate_3d_rf2_inside.hxx"
#include <carpetx.hxx>

#include <AMReX_Box.H>
#include <AMReX_BoxArray.H>
#include <AMReX_BoxList.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_FabArrayBase.H>
#include <AMReX_IntVect.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Vector.H>

#include <algorithm>
#include <cassert>
#include <iterator>
#include <map>
#include <utility>
#include <vector>

namespace CarpetX {

namespace {

struct band_cut_t {
  amrex::BoxArray fine;
  amrex::DistributionMapping owner;
  std::vector<interp_region_t> overlaps;
};

std::vector<int> local_indices_on(const int rank,
                                  const amrex::DistributionMapping &dm) {
  std::vector<int> local_index(dm.size(), -1);
  int boxes_before = 0;
  for (int k = 0; k < int(dm.size()); ++k)
    if (dm[k] == rank)
      local_index[k] = boxes_before++;
  return local_index;
}

std::vector<interp_region_t>
overlaps_on_this_rank(const amrex::BoxArray &gba,
                      const amrex::DistributionMapping &dm,
                      const amrex::IntVect &nghost, const amrex::BoxArray &fine,
                      const amrex::DistributionMapping &owner) {
  const int this_rank = amrex::ParallelDescriptor::MyProc();
  const std::vector<int> child_box_of = local_indices_on(this_rank, dm);
  const std::vector<int> band_box_of = local_indices_on(this_rank, owner);
  std::vector<interp_region_t> overlaps;
  for (int k = 0; k < int(gba.size()); ++k) {
    if (dm[k] != this_rank)
      continue;
    for (const auto &[b, ghosts] :
         fine.intersections(amrex::grow(gba[k], nghost))) {
      if (owner[b] != this_rank)
        continue;
      overlaps.push_back({child_box_of[k], band_box_of[b], ghosts});
    }
  }
  return overlaps;
}

band_cut_t cut_by_child_rank(const amrex::BoxArray &gba,
                             const amrex::DistributionMapping &dm,
                             const amrex::IntVect &nghost,
                             const amrex::Box &domain) {
  std::map<int, amrex::BoxList> child_boxes_of;
  for (int k = 0; k < int(gba.size()); ++k)
    child_boxes_of.try_emplace(dm[k], gba.ixType())
        .first->second.push_back(gba[k]);
  const amrex::BoxArray level = gba.simplified();
  amrex::BoxList fine(gba.ixType());
  amrex::Vector<int> owner;
  for (const auto &[rank, child_boxes] : child_boxes_of) {
    const amrex::BoxArray mine = amrex::BoxArray(child_boxes).simplified();
    for (int m = 0; m < int(mine.size()); ++m)
      for (const amrex::Box &ghosts_outside_level :
           level.complementIn(amrex::grow(mine[m], nghost) & domain)) {
        fine.push_back(ghosts_outside_level);
        owner.push_back(rank);
      }
  }
  assert(!owner.empty());
  amrex::BoxArray band(std::move(fine));
  amrex::DistributionMapping band_owner(std::move(owner));
  std::vector<interp_region_t> overlaps =
      overlaps_on_this_rank(gba, dm, nghost, band, band_owner);
  return {std::move(band), std::move(band_owner), std::move(overlaps)};
}

amrex::BoxArray coarsened(const amrex::BoxArray &fine,
                          const amrex::BoxConverter &coarsener) {
  amrex::BoxList crse(fine.ixType());
  for (int b = 0; b < int(fine.size()); ++b)
    crse.push_back(coarsener.doit(fine[b]));
  return amrex::BoxArray(std::move(crse));
}

} // namespace

rk_band_layout_t rk_band_cuts_t::layout_of(
    const amrex::FabArrayBase::FPinfo &fp, const amrex::BoxArray &gba,
    const amrex::DistributionMapping &dm, const amrex::IntVect &nghost) {
  assert(!fp.ba_crse_patch.empty());
  assert(fp.ba_crse_patch.size() == fp.ba_fine_patch.size());
  assert(fp.fact_crse_patch);
  assert(fp.ba_fine_patch.ixType() == gba.ixType());
  assert(dm.size() == gba.size());
  assert(fp.m_coarsener);
  assert(fp.m_dstng == nghost);
  const amrex::Box &domain = fp.m_dstdomain;
  auto cut = std::find_if(cuts_.begin(), cuts_.end(), [&](const cut_t &made) {
    return made.gba == gba && made.dm == dm && made.nghost == nghost &&
           made.domain == domain;
  });
  if (cut == cuts_.end()) {
    const band_cut_t by_rank = cut_by_child_rank(gba, dm, nghost, domain);
    assert(by_rank.owner.size() == by_rank.fine.size());
    assert(by_rank.fine.contains(fp.ba_fine_patch));
    assert(fp.ba_fine_patch.contains(by_rank.fine));
    cuts_.push_back({gba, dm, nghost, domain, by_rank.fine, by_rank.owner,
                     by_rank.overlaps});
    cut = std::prev(cuts_.end());
  }
  const amrex::BoxArray crse = coarsened(cut->fine, *fp.m_coarsener);
  assert(crse.size() == cut->fine.size());
  return {.crse = crse,
          .fine = cut->fine,
          .owner = cut->owner,
          .crse_factory = *fp.fact_crse_patch,
          .overlaps = cut->overlaps,
          .footprint = fp.ba_fine_patch};
}

} // namespace CarpetX
