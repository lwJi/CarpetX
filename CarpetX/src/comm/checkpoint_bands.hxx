#ifndef CARPETX_CARPETX_COMM_CHECKPOINT_BANDS_HXX
#define CARPETX_CARPETX_COMM_CHECKPOINT_BANDS_HXX

#include "internal.hxx"

#include <carpetx.hxx>

#include <AMReX_MultiFab.H>

#include <source_location>
#include <string>
#include <vector>

namespace CarpetX {

struct checkpoint_codec_t;
struct checkpoint_record_t;
struct group_selection_t;

class checkpoint_read_t {
  checkpoint_read_t() = default;
  friend void read_record(checkpoint_codec_t &codec,
                          const checkpoint_record_t &record,
                          const group_selection_t &selection, bool restartable,
                          const std::string &dir, const std::string &file,
                          const std::string &event, std::source_location site);

public:
  checkpoint_read_t(const checkpoint_read_t &) = delete;
  checkpoint_read_t &operator=(const checkpoint_read_t &) = delete;
};

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
                                          rat64 parent_step,
                                          const checkpoint_read_t &reading);

} // namespace CarpetX

#endif
