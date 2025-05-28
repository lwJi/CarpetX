#include <loop_device.hxx>

#include <cctk.h>
#include <cctk_Arguments.h>
#include <cctk_Parameters.h>

namespace Subcycling {

extern "C" void Subcycling_SetLevelNeighbor(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTSX_Subcycling_SetLevelNeighbor;

  // Set level values
  grid.loop_int_device<1, 1, 1>(
      grid.nghostzones,
      [=] CCTK_DEVICE(const Loop::PointDesc &p)
          CCTK_ATTRIBUTE_ALWAYS_INLINE { isrmbndry_ccc(p.I) = cctk_level; });

  grid.loop_int_device<0, 1, 1>(
      grid.nghostzones,
      [=] CCTK_DEVICE(const Loop::PointDesc &p)
          CCTK_ATTRIBUTE_ALWAYS_INLINE { isrmbndry_vcc(p.I) = cctk_level; });
  grid.loop_int_device<1, 0, 1>(
      grid.nghostzones,
      [=] CCTK_DEVICE(const Loop::PointDesc &p)
          CCTK_ATTRIBUTE_ALWAYS_INLINE { isrmbndry_cvc(p.I) = cctk_level; });
  grid.loop_int_device<1, 1, 0>(
      grid.nghostzones,
      [=] CCTK_DEVICE(const Loop::PointDesc &p)
          CCTK_ATTRIBUTE_ALWAYS_INLINE { isrmbndry_ccv(p.I) = cctk_level; });

  grid.loop_int_device<1, 0, 0>(
      grid.nghostzones,
      [=] CCTK_DEVICE(const Loop::PointDesc &p)
          CCTK_ATTRIBUTE_ALWAYS_INLINE { isrmbndry_cvv(p.I) = cctk_level; });
  grid.loop_int_device<0, 1, 0>(
      grid.nghostzones,
      [=] CCTK_DEVICE(const Loop::PointDesc &p)
          CCTK_ATTRIBUTE_ALWAYS_INLINE { isrmbndry_vcv(p.I) = cctk_level; });
  grid.loop_int_device<0, 0, 1>(
      grid.nghostzones,
      [=] CCTK_DEVICE(const Loop::PointDesc &p)
          CCTK_ATTRIBUTE_ALWAYS_INLINE { isrmbndry_vvc(p.I) = cctk_level; });

  grid.loop_int_device<0, 0, 0>(
      grid.nghostzones,
      [=] CCTK_DEVICE(const Loop::PointDesc &p)
          CCTK_ATTRIBUTE_ALWAYS_INLINE { isrmbndry_vvv(p.I) = cctk_level; });
}

extern "C" void Subcycling_SetIsRMBndry(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTSX_Subcycling_SetIsRMBndry;

  grid.loop_ghosts_device<1, 1, 1>(
      grid.nghostzones,
      [=] CCTK_DEVICE(const Loop::PointDesc &p) CCTK_ATTRIBUTE_ALWAYS_INLINE {
        isrmbndry_ccc(p.I) = (isrmbndry_ccc(p.I) == cctk_level) ? 0 : 1;
      });

  grid.loop_ghosts_device<0, 1, 1>(
      grid.nghostzones,
      [=] CCTK_DEVICE(const Loop::PointDesc &p) CCTK_ATTRIBUTE_ALWAYS_INLINE {
        isrmbndry_vcc(p.I) = (isrmbndry_vcc(p.I) == cctk_level) ? 0 : 1;
      });
  grid.loop_ghosts_device<1, 0, 1>(
      grid.nghostzones,
      [=] CCTK_DEVICE(const Loop::PointDesc &p) CCTK_ATTRIBUTE_ALWAYS_INLINE {
        isrmbndry_cvc(p.I) = (isrmbndry_cvc(p.I) == cctk_level) ? 0 : 1;
      });
  grid.loop_ghosts_device<1, 1, 0>(
      grid.nghostzones,
      [=] CCTK_DEVICE(const Loop::PointDesc &p) CCTK_ATTRIBUTE_ALWAYS_INLINE {
        isrmbndry_ccv(p.I) = (isrmbndry_ccv(p.I) == cctk_level) ? 0 : 1;
      });

  grid.loop_ghosts_device<1, 0, 0>(
      grid.nghostzones,
      [=] CCTK_DEVICE(const Loop::PointDesc &p) CCTK_ATTRIBUTE_ALWAYS_INLINE {
        isrmbndry_cvv(p.I) = (isrmbndry_cvv(p.I) == cctk_level) ? 0 : 1;
      });
  grid.loop_ghosts_device<0, 1, 0>(
      grid.nghostzones,
      [=] CCTK_DEVICE(const Loop::PointDesc &p) CCTK_ATTRIBUTE_ALWAYS_INLINE {
        isrmbndry_vcv(p.I) = (isrmbndry_vcv(p.I) == cctk_level) ? 0 : 1;
      });
  grid.loop_ghosts_device<0, 0, 1>(
      grid.nghostzones,
      [=] CCTK_DEVICE(const Loop::PointDesc &p) CCTK_ATTRIBUTE_ALWAYS_INLINE {
        isrmbndry_vvc(p.I) = (isrmbndry_vvc(p.I) == cctk_level) ? 0 : 1;
      });

  grid.loop_ghosts_device<0, 0, 0>(
      grid.nghostzones,
      [=] CCTK_DEVICE(const Loop::PointDesc &p) CCTK_ATTRIBUTE_ALWAYS_INLINE {
        isrmbndry_vvv(p.I) = (isrmbndry_vvv(p.I) == cctk_level) ? 0 : 1;
      });
}

} // namespace Subcycling
