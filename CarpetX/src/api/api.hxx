#ifndef CARPETX_CARPETX_API_API_HXX
#define CARPETX_CARPETX_API_API_HXX

#include "internal.hxx"

#include <carpetx.hxx>

#include "hierarchy/driver.hxx"
#include "hierarchy/operations.hxx"

#include <span>

namespace CarpetX {

void adopt_writers(registry_writer_t &registry, scratch_writer_t &scratch);
void return_writers();
registry_writer_t &registry_writer();
scratch_writer_t &scratch_writer();

void require_no_open_write(
    const GHExt::PatchData::LevelData::GroupData &groupdata);

void require_operable(window_t window, std::span<const gi_t> groups);

} // namespace CarpetX

#endif
