#ifndef CARPETX_CARPETX_CONFIG_DECODE_BOUNDARIES_HXX
#define CARPETX_CARPETX_CONFIG_DECODE_BOUNDARIES_HXX

#include "internal.hxx"

#include "config/boundaries.hxx"
#include <carpetx.hxx>

#include <optional>

namespace CarpetX {

faces_t<symmetry_t> get_symmetries(std::optional<patch_t> patch);
faces_t<boundary_t> get_default_boundaries();

} // namespace CarpetX

#endif
