#ifndef CARPETX_CARPETX_HIERARCHY_BINDING_HXX
#define CARPETX_CARPETX_HIERARCHY_BINDING_HXX

#include "internal.hxx"

#include "hierarchy/driver.hxx"

#include <cctk.h>

namespace CarpetX {

struct cctkGH_binding {
  static void bind_arrays(cGH *cctkGH);
  static void rotate(gi_t gi);
};

} // namespace CarpetX

#endif
