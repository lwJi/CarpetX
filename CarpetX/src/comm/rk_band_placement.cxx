#include "comm/rk_band_placement_inside.hxx"

#include "util/refuse.hxx"
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

#include <cassert>
#include <cstdint>
#include <map>
#include <source_location>
#include <utility>

namespace CarpetX {

namespace {

using cells_by_rank_t = std::map<int, std::int64_t>;

cells_by_rank_t received_cells(const amrex::Box &fine_patch_box,
                               const amrex::BoxArray &gba,
                               const amrex::DistributionMapping &dm,
                               const amrex::IntVect &nghost) {
  cells_by_rank_t cells;
  for (const auto &[child_box, ghosts_filled] :
       gba.intersections(fine_patch_box, false, nghost))
    cells[dm[child_box]] += ghosts_filled.numPts();
  return cells;
}

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
  amrex::BoxArray crse, fine;
  amrex::DistributionMapping owner;
  bool each_rank_owns_its_ghosts;
};

band_cut_t cut_by_child_rank(const amrex::FabArrayBase::FPinfo &fp,
                             const amrex::BoxArray &gba,
                             const amrex::DistributionMapping &dm,
                             const amrex::IntVect &nghost) {
  std::map<int, amrex::BoxList> child_boxes_of;
  for (int k = 0; k < int(gba.size()); ++k)
    child_boxes_of.try_emplace(dm[k], gba.ixType())
        .first->second.push_back(gba[k]);
  const amrex::BoxArray level = gba.simplified();
  amrex::BoxList crse(gba.ixType()), fine(gba.ixType());
  amrex::Vector<int> owner;
  for (const auto &[rank, child_boxes] : child_boxes_of) {
    const amrex::BoxArray mine = amrex::BoxArray(child_boxes).simplified();
    for (int m = 0; m < int(mine.size()); ++m)
      for (const amrex::Box &ghosts_outside_level :
           level.complementIn(amrex::grow(mine[m], nghost) & fp.m_dstdomain)) {
        crse.push_back(fp.m_coarsener->doit(ghosts_outside_level));
        fine.push_back(ghosts_outside_level);
        owner.push_back(rank);
      }
  }
  assert(!owner.empty());
  return {amrex::BoxArray(std::move(crse)), amrex::BoxArray(std::move(fine)),
          amrex::DistributionMapping(std::move(owner)), true};
}

band_cut_t cut_of(const rk_band_placement_t placement,
                  const amrex::FabArrayBase::FPinfo &fp,
                  const amrex::BoxArray &gba,
                  const amrex::DistributionMapping &dm,
                  const amrex::IntVect &nghost) {
  switch (placement) {
  case rk_band_placement_t::child:
    return cut_by_child_rank(fp, gba, dm, nghost);
  case rk_band_placement_t::knapsack:
    return {fp.ba_crse_patch, fp.ba_fine_patch, fp.dm_patch, false};
  }
  refuse(std::source_location::current(), "rk_band_placement_t has no value %d",
         int(placement));
}

const char *name_of(const rk_band_placement_t placement) {
  switch (placement) {
  case rk_band_placement_t::child:
    return "child";
  case rk_band_placement_t::knapsack:
    return "knapsack";
  }
  refuse(std::source_location::current(), "rk_band_placement_t has no value %d",
         int(placement));
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

rk_band_layout_t rk_band_layout_of(const rk_band_placement_t placement,
                                   const amrex::FabArrayBase::FPinfo &fp,
                                   const amrex::BoxArray &gba,
                                   const amrex::DistributionMapping &dm,
                                   const amrex::IntVect &nghost) {
  assert(!fp.ba_crse_patch.empty());
  assert(fp.ba_crse_patch.size() == fp.ba_fine_patch.size());
  assert(fp.fact_crse_patch);
  assert(fp.fact_fine_patch);
  assert(fp.ba_fine_patch.ixType() == gba.ixType());
  assert(dm.size() == gba.size());
  assert(fp.m_coarsener);
  assert(fp.m_dstng == nghost);
  const band_cut_t cut = cut_of(placement, fp, gba, dm, nghost);
  assert(cut.crse.size() == cut.fine.size());
  assert(cut.owner.size() == cut.fine.size());
  assert(cut.fine.contains(fp.ba_fine_patch));
  assert(fp.ba_fine_patch.contains(cut.fine));
  return {cut.crse,
          cut.fine,
          cut.owner,
          *fp.fact_crse_patch,
          *fp.fact_fine_patch,
          cut.each_rank_owns_its_ghosts,
          fp.ba_fine_patch};
}

rk_band_locality_t &
rk_band_locality_t::operator+=(const rk_band_locality_t &other) {
  stored_local_bytes += other.stored_local_bytes;
  stored_remote_bytes += other.stored_remote_bytes;
  received_local_bytes += other.received_local_bytes;
  received_remote_bytes += other.received_remote_bytes;
  return *this;
}

rk_band_locality_t rk_band_locality_of(
    const rk_band_layout_t &layout, const int numvars,
    const amrex::BoxArray &gba, const amrex::DistributionMapping &dm,
    const amrex::IntVect &nghost, const amrex::BoxArray &parent_gba,
    const amrex::DistributionMapping &parent_dm,
    const amrex::Periodicity &period) {
  assert(layout.crse.size() == layout.fine.size());
  assert(layout.owner.size() == layout.fine.size());
  assert(layout.fine.ixType() == gba.ixType());
  assert(layout.crse.ixType() == parent_gba.ixType());
  assert(dm.size() == gba.size());
  assert(parent_dm.size() == parent_gba.size());
  assert(numvars > 0);

  const std::int64_t bytes_per_cell =
      std::int64_t(numvars) * std::int64_t(sizeof(CCTK_REAL));
  rk_band_locality_t locality;
  const int num_patch_boxes = int(layout.fine.size());
  for (int b = 0; b < num_patch_boxes; ++b) {
    const int owner = layout.owner[b];
    const split_by_owner_t stored = split_by_owner(
        stored_cells(layout.crse[b], parent_gba, parent_dm, period), owner);
    const split_by_owner_t received =
        split_by_owner(received_cells(layout.fine[b], gba, dm, nghost), owner);
    const std::int64_t sent_to_other_ranks =
        layout.each_rank_owns_its_ghosts ? 0 : received.remote_cells;
    locality.stored_local_bytes += stored.local_cells * bytes_per_cell;
    locality.stored_remote_bytes += stored.remote_cells * bytes_per_cell;
    locality.received_local_bytes += received.local_cells * bytes_per_cell;
    locality.received_remote_bytes += sent_to_other_ranks * bytes_per_cell;
  }
  return locality;
}

void report_rk_band_locality(const level_t level, const int iteration,
                             const rk_band_placement_t placement,
                             const rk_band_locality_t &total) {
  if (CCTK_MyProc(nullptr) != 0)
    return;
  CCTK_VINFO(
      "RK band locality, level %d, built at iteration %d, placement: %s\n"
      "  %-19s%13s%12s%9s\n"
      "  %-19s%13.6f%12.6f%8.0f%%\n"
      "  %-19s%13.6f%12.6f%8.0f%%",
      level.index(), iteration, name_of(placement), "hop", "local MB",
      "remote MB", "remote", "parent -> band",
      megabytes(total.stored_local_bytes), megabytes(total.stored_remote_bytes),
      remote_percent(total.stored_local_bytes, total.stored_remote_bytes),
      "band -> fine ghosts", megabytes(total.received_local_bytes),
      megabytes(total.received_remote_bytes),
      remote_percent(total.received_local_bytes, total.received_remote_bytes));
}

} // namespace CarpetX
