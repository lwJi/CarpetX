#ifndef CARPETX_CARPETX_CONFIG_RUN_CONFIG_HXX
#define CARPETX_CARPETX_CONFIG_RUN_CONFIG_HXX

#include "internal.hxx"

#include "config/indices.hxx"

#include <cctk.h>

#include <AMReX_RealBox.H>

#include <array>

namespace CarpetX {

struct fixed_config_t {
  bool use_subcycling;
  bool do_reflux;
  int max_num_levels;
  std::array<int, dim> nghostzones;
  amrex::RealBox domain;
  std::array<int, dim> ncells;
  bool prolongate_per_group;
};

struct steered_config_t {
  bool verbose;
  bool poison_undefined_values;
  bool poison_undefined_arrays;
  int interpolation_order;
  CCTK_REAL regrid_error_threshold;
};

} // namespace CarpetX

#endif
