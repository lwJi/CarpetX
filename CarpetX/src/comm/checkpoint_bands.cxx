#include "comm/checkpoint_bands.hxx"

#include "config/indices.hxx"
#include "config/stages.hxx"
#include "hierarchy/model.hxx"
#include "comm/buffers.hxx"
#include <carpetx.hxx>

#include <AMReX_FluxRegister.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Orientation.H>

#include <cassert>
#include <iomanip>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace CarpetX {

namespace {

std::string ks_source_band_tag(const stage_t stage) {
  std::ostringstream buf;
  buf << "kss_s" << std::setw(2) << std::setfill('0')
      << all_rk_stages.offset(stage);
  return buf.str();
}

std::string old_source_band_tag() { return "olds"; }

std::string flux_register_band_tag(const amrex::Orientation face) {
  assert(face.coordDir() >= 0 && face.coordDir() < dim);
  std::ostringstream buf;
  buf << "freg_" << "xyz"[face.coordDir()] << (face.isHigh() ? "hi" : "lo");
  return buf.str();
}

template <class Band>
std::vector<Band>
list_bands(group_buffers_t &buffers,
           const std::optional<rk_bands_t::slots_t> rk_slots) {
  std::vector<Band> bands;
  const auto add_band = [&](amrex::MultiFab *const band, std::string tag) {
    if (band && !band->empty())
      bands.push_back({std::move(tag), band});
  };
  if (rk_slots) {
    const stage_range stages = rk_stages(int(rk_slots->ks.size()));
    for (const stage_t stage : stages)
      add_band(&rk_slots->ks[stages.offset(stage)], ks_source_band_tag(stage));
    add_band(&rk_slots->old, old_source_band_tag());
  }
  if (buffers.freg)
    for (int dir = 0; dir < dim; ++dir)
      for (const auto side :
           {amrex::Orientation::low, amrex::Orientation::high}) {
        const amrex::Orientation face(dir, side);
        add_band(&(*buffers.freg)[face].multiFab(),
                 flux_register_band_tag(face));
      }
  return bands;
}

} // namespace

std::vector<checkpoint_band_t>
checkpoint_bands(const level_t level, const gi_t gi, const rat64 parent_step) {
  const std::optional<level_t> child = level.child();
  if (!child)
    return {};
  group_buffers_t &childbuffers = buffers_of(driver().level(*child), gi);
  std::optional<rk_bands_t> &rk_bands = childbuffers.rk_bands;
  return list_bands<checkpoint_band_t>(
      childbuffers, rk_bands ? rk_bands->stored(parent_step) : std::nullopt);
}

std::vector<restore_band_t> restore_bands(const level_t level, const gi_t gi,
                                          const rat64 parent_step) {
  const std::optional<level_t> child = level.child();
  if (!child)
    return {};
  group_buffers_t &childbuffers = buffers_of(driver().level(*child), gi);
  std::optional<rk_bands_t> &rk_bands = childbuffers.rk_bands;
  return list_bands<restore_band_t>(
      childbuffers, rk_bands ? std::optional<rk_bands_t::slots_t>(
                                   rk_bands->restore(parent_step))
                             : std::nullopt);
}

} // namespace CarpetX
