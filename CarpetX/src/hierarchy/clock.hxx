#ifndef CARPETX_CARPETX_HIERARCHY_CLOCK_HXX
#define CARPETX_CARPETX_HIERARCHY_CLOCK_HXX

#include "internal.hxx"

#include "hierarchy/model.hxx"

#include <cctk.h>

#include <functional>
#include <optional>
#include <string>

namespace CarpetX {

struct cctkGH_clock {
private:
  static void set_run(const clock_writer_t &writer, const GHExt &ext,
                      GHExt::run_clock_t next);
  static void show_stage(const stage_writer_t &writer, const GHExt &ext,
                         std::optional<GHExt::stage_clock_t> stage);
  friend class clock_writer_t;
  friend class stage_writer_t;
};

void require_current_clock(const GHExt &ext, const cGH *cctkGH,
                           const std::function<std::string()> &who);

} // namespace CarpetX

#endif
