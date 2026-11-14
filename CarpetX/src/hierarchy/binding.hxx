#ifndef CARPETX_CARPETX_HIERARCHY_BINDING_HXX
#define CARPETX_CARPETX_HIERARCHY_BINDING_HXX

#include "internal.hxx"

#include "hierarchy/model.hxx"
#include "hierarchy/loops.hxx"

#include <cctk.h>

namespace CarpetX {

class clock_writer_t;

struct cctkGH_binding {
  static void bind_arrays(cGH *cctkGH);
  static void unbind_arrays(cGH *cctkGH);
  static void bind_local(cGH *cctkGH, GHExt::PatchData::LevelData &leveldata,
                         const MFPointer &mfp);
  static void unbind_local(cGH *cctkGH,
                           const GHExt::PatchData::LevelData &leveldata,
                           const MFPointer &mfp);

private:
  static void rotate(GHExt &ext, window_t window, gi_t gi);
  static void rotate(GHExt &ext, gi_t gi);
  friend class clock_writer_t;
};

} // namespace CarpetX

#endif
