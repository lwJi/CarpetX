#include "api/api.hxx"
#include "hierarchy/writers.hxx"
#include "util/refuse.hxx"

#include <optional>
#include <source_location>

namespace CarpetX {

namespace {
struct lent_writers_t {
  registry_writer_t *registry;
  scratch_writer_t *scratch;
  stage_writer_t *stage;
};

std::optional<lent_writers_t> lent_writers;

const lent_writers_t &adopted_writers() {
  if (!lent_writers)
    refuse(std::source_location::current(),
           "the driver's SetupGH has not lent api/ the registry, scratch and "
           "stage writers: the public API is available between the driver's "
           "SetupGH and its Shutdown");
  return *lent_writers;
}
} // namespace

void adopt_writers(registry_writer_t &registry, scratch_writer_t &scratch,
                   stage_writer_t &stage) {
  if (lent_writers)
    refuse(std::source_location::current(),
           "api/ already holds the registry, scratch and stage writers: "
           "SetupGH lends them once per run");
  lent_writers = lent_writers_t{&registry, &scratch, &stage};
}

void return_writers() { lent_writers.reset(); }

registry_writer_t &registry_writer() { return *adopted_writers().registry; }

scratch_writer_t &scratch_writer() { return *adopted_writers().scratch; }

stage_writer_t &stage_writer() { return *adopted_writers().stage; }

} // namespace CarpetX
