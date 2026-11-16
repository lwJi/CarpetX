#ifndef CARPETX_CARPETX_COMM_CHECKPOINT_BANDS_HXX
#define CARPETX_CARPETX_COMM_CHECKPOINT_BANDS_HXX

#include "internal.hxx"

#include <carpetx.hxx>

#include <AMReX_MultiFab.H>

#include <string>
#include <vector>

namespace CarpetX {

struct checkpoint_band_t {
  std::string tag;
  const amrex::MultiFab *band;
};

struct restore_band_t {
  std::string tag;
  amrex::MultiFab *band;
};

std::vector<checkpoint_band_t> checkpoint_bands(level_t level, gi_t gi,
                                                rat64 parent_step);

std::vector<restore_band_t> restore_bands(level_t level, gi_t gi,
                                          rat64 parent_step);

} // namespace CarpetX

#endif
