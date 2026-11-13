#ifndef CARPETX_CARPETX_HIERARCHY_GROUP_TAGS_HXX
#define CARPETX_CARPETX_HIERARCHY_GROUP_TAGS_HXX

#include "internal.hxx"

#include "hierarchy/driver.hxx"

#include <carpetx.hxx>

#include <array>
#include <optional>
#include <string>

namespace CarpetX {

std::array<std::array<symmetry_t, dim>, 2>
get_symmetries(std::optional<patch_t> patch);
std::array<std::array<boundary_t, dim>, 2> get_default_boundaries();

bool get_group_checkpoint_flag(gi_t gi);
std::optional<bool> get_group_restrict_tag(gi_t gi);
bool get_group_evolve_flag(gi_t gi);

std::string get_group_prolongation_type(gi_t gi);
int get_group_prolongation_order(gi_t gi);

} // namespace CarpetX

#endif
