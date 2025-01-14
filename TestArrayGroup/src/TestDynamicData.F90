#include <cctk.h>
#include <cctk_Arguments.h>
#include <cctk_Parameters.h>

subroutine TestArrayGroup_DynamicDataF(CCTK_ARGUMENTS)
  DECLARE_CCTK_PARAMETERS
  DECLARE_CCTK_ARGUMENTS

  integer, dimension(3) :: dim3

  ! RANK() is not supported by all compilers, so we fail to complile here instead
  ! Validate grid array dynamic data
  dim3 = SHAPE(test1)
  if(SIZE(test1, 1) /= 5 .or. SIZE(test1, 2) /= 6 .or. SIZE(test1, 3) /= 4) then
      call CCTK_ERROR("incorrect size in test1 array dynamic data")
  endif

  dim3 = SHAPE(test2)
  if(SIZE(test2, 1) /= 5 .or. SIZE(test2, 2) /= 6 .or. SIZE(test2, 3) /= 4) then
      call CCTK_ERROR("incorrect size in test2 array dynamic data")
  endif

  dim3 = SHAPE(test3)
  if(SIZE(test3, 1) /= 5 .or. SIZE(test3, 2) /= 6 .or. SIZE(test3, 3) /= 4) then
      call CCTK_ERROR("incorrect size in test3 array dynamic data")
  endif

end subroutine TestArrayGroup_DynamicDataF
