#include "comm/rk_band_placement_inside.hxx"

#include <carpetx.hxx>

#include <cctk.h>

#include <AMReX_Box.H>
#include <AMReX_BoxArray.H>
#include <AMReX_BoxList.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_FabArrayBase.H>
#include <AMReX_IntVect.H>
#include <AMReX_Periodicity.H>
#include <AMReX_Vector.H>

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iterator>
#include <map>
#include <utility>

namespace CarpetX {

namespace {

using cells_by_rank_t = std::map<int, std::int64_t>;

cells_by_rank_t stored_cells(const amrex::Box &crse_patch_box,
                             const amrex::BoxArray &parent_gba,
                             const amrex::DistributionMapping &parent_dm,
                             const amrex::Periodicity &period) {
  cells_by_rank_t cells;
  for (const amrex::IntVect &shift : period.shiftIntVect())
    for (const auto &[parent_box, interior_read] :
         parent_gba.intersections(crse_patch_box + shift))
      cells[parent_dm[parent_box]] += interior_read.numPts();
  return cells;
}

struct split_by_owner_t {
  std::int64_t local_cells = 0, remote_cells = 0;
};

split_by_owner_t split_by_owner(const cells_by_rank_t &cells, const int owner) {
  split_by_owner_t split;
  for (const auto &[rank, count] : cells)
    (rank == owner ? split.local_cells : split.remote_cells) += count;
  return split;
}

struct band_cut_t {
  amrex::BoxArray fine;
  amrex::DistributionMapping owner;
};

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
  return {amrex::BoxArray(std::move(fine)),
          amrex::DistributionMapping(std::move(owner))};
}

amrex::BoxArray coarsened(const amrex::BoxArray &fine,
                          const amrex::BoxConverter &coarsener) {
  amrex::BoxList crse(fine.ixType());
  for (int b = 0; b < int(fine.size()); ++b)
    crse.push_back(coarsener.doit(fine[b]));
  return amrex::BoxArray(std::move(crse));
}

constexpr double bytes_per_megabyte = 1.0e6;

double megabytes(const std::int64_t bytes) {
  return double(bytes) / bytes_per_megabyte;
}

double remote_percent(const std::int64_t local_bytes,
                      const std::int64_t remote_bytes) {
  const std::int64_t moved_bytes = local_bytes + remote_bytes;
  return moved_bytes == 0 ? 0.0
                          : 100.0 * double(remote_bytes) / double(moved_bytes);
}

} // namespace

rk_band_layout_t rk_band_cuts_t::layout_of(
    const amrex::FabArrayBase::FPinfo &fp, const amrex::BoxArray &gba,
    const amrex::DistributionMapping &dm, const amrex::IntVect &nghost) {
  assert(!fp.ba_crse_patch.empty());
  assert(fp.ba_crse_patch.size() == fp.ba_fine_patch.size());
  assert(fp.fact_crse_patch);
  assert(fp.fact_fine_patch);
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
    cuts_.push_back({gba, dm, nghost, domain, by_rank.fine, by_rank.owner});
    cut = std::prev(cuts_.end());
  }
  const amrex::BoxArray crse = coarsened(cut->fine, *fp.m_coarsener);
  assert(crse.size() == cut->fine.size());
  return {crse,
          cut->fine,
          cut->owner,
          *fp.fact_crse_patch,
          *fp.fact_fine_patch,
          fp.ba_fine_patch};
}

rk_band_locality_t &
rk_band_locality_t::operator+=(const rk_band_locality_t &other) {
  stored_local_bytes += other.stored_local_bytes;
  stored_remote_bytes += other.stored_remote_bytes;
  return *this;
}

rk_band_locality_t
rk_band_locality_of(const rk_band_layout_t &layout, const int numvars,
                    const amrex::BoxArray &parent_gba,
                    const amrex::DistributionMapping &parent_dm,
                    const amrex::Periodicity &period) {
  assert(layout.owner.size() == layout.crse.size());
  assert(layout.crse.ixType() == parent_gba.ixType());
  assert(parent_dm.size() == parent_gba.size());
  assert(numvars > 0);

  const std::int64_t bytes_per_cell =
      std::int64_t(numvars) * std::int64_t(sizeof(CCTK_REAL));
  rk_band_locality_t locality;
  const int num_band_boxes = int(layout.crse.size());
  for (int b = 0; b < num_band_boxes; ++b) {
    const split_by_owner_t stored = split_by_owner(
        stored_cells(layout.crse[b], parent_gba, parent_dm, period),
        layout.owner[b]);
    locality.stored_local_bytes += stored.local_cells * bytes_per_cell;
    locality.stored_remote_bytes += stored.remote_cells * bytes_per_cell;
  }
  return locality;
}

void report_rk_band_locality(const level_t level, const int iteration,
                             const rk_band_locality_t &total) {
  if (CCTK_MyProc(nullptr) != 0)
    return;
  CCTK_VINFO(
      "RK band locality, level %d, built at iteration %d\n"
      "  %-19s%13s%12s%9s\n"
      "  %-19s%13.6f%12.6f%8.0f%%",
      level.index(), iteration, "hop", "local MB", "remote MB", "remote",
      "parent -> band", megabytes(total.stored_local_bytes),
      megabytes(total.stored_remote_bytes),
      remote_percent(total.stored_local_bytes, total.stored_remote_bytes));
}

} // namespace CarpetX
