#ifndef CARPETX_CARPETX_IO_IO_HXX
#define CARPETX_CARPETX_IO_IO_HXX

#include "internal.hxx"

#include "io/checkpoint.hxx"

#include <cctk.h>

namespace CarpetX {

checkpoint_record_t RecoverGridStructure(cGH *cctkGH);
void RecoverGH(const cGH *cctkGH, const checkpoint_record_t &record);
void InputGH(const cGH *cctkGH);

int OutputGH(const cGH *cctkGH);

} // namespace CarpetX

#endif
