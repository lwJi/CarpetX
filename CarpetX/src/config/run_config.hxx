#ifndef CARPETX_CARPETX_CONFIG_RUN_CONFIG_HXX
#define CARPETX_CARPETX_CONFIG_RUN_CONFIG_HXX

#include "internal.hxx"

#include "config/indices.hxx"

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

} // namespace CarpetX

#endif
