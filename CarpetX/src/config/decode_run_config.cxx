#include "config/decode_run_config.hxx"
#include "config/run_config.hxx"

#include <cctk.h>
#include <cctk_Parameters.h>

#include <AMReX_RealBox.H>

namespace CarpetX {

namespace {

int ghosts_or_default(const CCTK_INT ghosts, const CCTK_INT default_ghosts) {
  return ghosts >= 0 ? ghosts : default_ghosts;
}

} // namespace

fixed_config_t decode_fixed_config() {
  DECLARE_CCTK_PARAMETERS;
  return fixed_config_t{
      .use_subcycling = bool(use_subcycling),
      .do_reflux = bool(do_reflux),
      .max_num_levels = int(max_num_levels),
      .nghostzones = {ghosts_or_default(ghost_size, ghost_size_x),
                      ghosts_or_default(ghost_size, ghost_size_y),
                      ghosts_or_default(ghost_size, ghost_size_z)},
      .domain = amrex::RealBox({xmin, ymin, zmin}, {xmax, ymax, zmax}),
      .ncells = {int(ncells_x), int(ncells_y), int(ncells_z)},
      .prolongate_per_group = bool(prolongate_per_group),
  };
}

} // namespace CarpetX
