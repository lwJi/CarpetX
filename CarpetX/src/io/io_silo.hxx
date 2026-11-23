#ifndef CARPETX_CARPETX_IO_IO_SILO_HXX
#define CARPETX_CARPETX_IO_IO_SILO_HXX

#include "internal.hxx"

#include <cctk.h>

namespace CarpetX {

void OutputSilo(const cGH *cctkGH);

int silo_proc_stride(int nprocs);

} // namespace CarpetX

#endif
