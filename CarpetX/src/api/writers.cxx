#include "api/api.hxx"
#include "hierarchy/operations.hxx"
#include "util/refuse.hxx"

#include <optional>
#include <source_location>

namespace CarpetX {

namespace {
struct lent_writers_t {
  registry_writer_t *registry;
  scratch_writer_t *scratch;
};

std::optional<lent_writers_t> lent_writers;

const lent_writers_t &adopted_writers() {
  if (!lent_writers)
    refuse(std::source_location::current(),
           "the driver's SetupGH has not lent api/ the registry and scratch "
           "writers: the public API is available between the driver's SetupGH "
           "and its Shutdown");
  return *lent_writers;
}
} // namespace

void adopt_writers(registry_writer_t &registry, scratch_writer_t &scratch) {
  if (lent_writers)
    refuse(std::source_location::current(),
           "api/ already holds the registry and scratch writers: SetupGH "
           "lends them once per run");
  lent_writers = lent_writers_t{&registry, &scratch};
}

void return_writers() { lent_writers.reset(); }

registry_writer_t &registry_writer() { return *adopted_writers().registry; }

scratch_writer_t &scratch_writer() { return *adopted_writers().scratch; }

} // namespace CarpetX
