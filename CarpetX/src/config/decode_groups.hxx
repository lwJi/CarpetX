#ifndef CARPETX_CARPETX_CONFIG_DECODE_GROUPS_HXX
#define CARPETX_CARPETX_CONFIG_DECODE_GROUPS_HXX

#include "internal.hxx"

#include "config/groups.hxx"
#include "config/run_config.hxx"
#include <carpetx.hxx>

namespace CarpetX {

group_declaration_t declare_group(gi_t gi, const fixed_config_t &fixed);

} // namespace CarpetX

#endif
