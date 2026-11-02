#ifndef CARPETX_CARPETX_HIERARCHY_BINDING_HXX
#define CARPETX_CARPETX_HIERARCHY_BINDING_HXX

#include "internal.hxx"

#include "hierarchy/driver.hxx"
#include "hierarchy/loops.hxx"

#include <cctk.h>

namespace CarpetX {

struct cctkGH_binding {
  static void bind_arrays(cGH *cctkGH);
  static void unbind_arrays(cGH *cctkGH);
  static void bind_local(cGH *cctkGH,
                         const GHExt::PatchData::LevelData &leveldata,
                         const MFPointer &mfp);
  static void unbind_local(cGH *cctkGH,
                           const GHExt::PatchData::LevelData &leveldata,
                           const MFPointer &mfp);
  static void rotate(window_t window, gi_t gi);
  static void rotate(gi_t gi);
};

} // namespace CarpetX

#endif
