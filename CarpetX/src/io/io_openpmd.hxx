#ifndef CARPETX_CARPETX_IO_IO_OPENPMD_HXX
#define CARPETX_CARPETX_IO_IO_OPENPMD_HXX

#include "internal.hxx"

#include <cctk.h>

namespace CarpetX {

void OutputOpenPMD(const cGH *cctkGH);
void ShutdownOpenPMD();

} // namespace CarpetX

#endif
