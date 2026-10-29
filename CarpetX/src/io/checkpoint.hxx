#ifndef CARPETX_CARPETX_IO_CHECKPOINT_HXX
#define CARPETX_CARPETX_IO_CHECKPOINT_HXX

#include "internal.hxx"

#include "util/var_type.hxx"
#include <carpetx.hxx>

#include <cctk.h>

#include <AMReX_Box.H>
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>

#include <memory>
#include <optional>
#include <set>
#include <source_location>
#include <string>
#include <variant>
#include <vector>

namespace CarpetX {

struct group_selection_t;

enum class record_purpose { checkpoint, output };

struct buffer_id_t {
  std::string group;
  std::vector<std::string> vars;
  std::optional<level_t> level;
  std::variant<tl_t, std::string> which;
  std::string name() const;
};

struct grid_buffer_t {
  amrex::MultiFab *mfab;
};

struct array_buffer_t {
  void *data;
  var_type vartype;
  std::vector<int> shape;
};

struct buffer_t {
  buffer_id_t id;
  std::variant<grid_buffer_t, array_buffer_t> data;
};

struct level_record_t {
  std::vector<amrex::Box> boxes;
  std::optional<rat64> iteration;
  std::optional<int> last_step_iteration;
  amrex::Geometry geom;
  const amrex::MultiFab *vertex_coords = nullptr;
};

struct checkpoint_record_t {
  record_purpose purpose = record_purpose::checkpoint;
  int iteration;
  CCTK_REAL time;
  CCTK_REAL delta_time;
  std::string parameters;
  std::vector<std::vector<level_record_t> > levels;
  std::set<std::string> present;
  std::vector<buffer_t> buffers;
};

struct codec_limits_t {
  bool carries_arrays;
  int max_timelevels;
  bool rechops_on_recovery;
};

struct checkpoint_codec_t {
  virtual ~checkpoint_codec_t() = default;
  virtual codec_limits_t limits() const = 0;
  virtual std::string key(const buffer_id_t &id) const = 0;
  virtual int read_parameters(const std::string &dir,
                              const std::string &file) = 0;
  virtual void encode(const checkpoint_record_t &record, const std::string &dir,
                      const std::string &file) = 0;
  virtual checkpoint_record_t decode_structure(const std::string &dir,
                                               const std::string &file,
                                               int iteration) = 0;
  virtual void decode_data(const checkpoint_record_t &record,
                           const std::string &dir, const std::string &file) = 0;
};

std::unique_ptr<checkpoint_codec_t>
make_codec(const char *method,
           std::source_location site = std::source_location::current());

std::unique_ptr<checkpoint_codec_t> make_openpmd_codec();

std::unique_ptr<checkpoint_codec_t> make_silo_codec();

void refuse_unless_recoverable(
    const codec_limits_t &limits,
    std::source_location site = std::source_location::current());

bool finer_level_is_behind(const checkpoint_record_t &record, level_t level);

checkpoint_record_t
build_record(const cGH *cctkGH, const group_selection_t &selection,
             record_purpose purpose, const codec_limits_t &limits,
             std::source_location site = std::source_location::current());

void build_buffers(checkpoint_record_t &record,
                   const group_selection_t &selection, record_purpose purpose,
                   const codec_limits_t &limits);

void apply_structure(
    cGH *cctkGH, checkpoint_record_t &record,
    std::source_location site = std::source_location::current());

void read_record(checkpoint_codec_t &codec, const checkpoint_record_t &record,
                 const std::string &dir, const std::string &file,
                 const std::string &event,
                 std::source_location site = std::source_location::current());

} // namespace CarpetX

#endif
