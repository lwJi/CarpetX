#ifndef CARPETX_CARPETX_HIERARCHY_CLOCK_HXX
#define CARPETX_CARPETX_HIERARCHY_CLOCK_HXX

#include "internal.hxx"

#include "hierarchy/model.hxx"

#include <cctk.h>

#include <functional>
#include <string>

namespace CarpetX {

void publish_clock(const GHExt &ext);

void require_current_clock(const GHExt &ext, const cGH *cctkGH,
                           const std::function<std::string()> &who);

} // namespace CarpetX

#endif
