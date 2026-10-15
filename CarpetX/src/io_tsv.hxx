#ifndef CARPETX_CARPETX_IO_TSV_HXX
#define CARPETX_CARPETX_IO_TSV_HXX

#include "io.hxx"

#include <cctk.h>

namespace CarpetX {

// Both write grid functions on `levels` only; grid scalars and arrays have no
// levels and are always written
void OutputTSVold(const cGH *restrict cctkGH, const level_range_t &levels);
void OutputTSV(const cGH *restrict cctkGH, const level_range_t &levels);

} // namespace CarpetX

#endif // #ifndef CARPETX_CARPETX_IO_TSV_HXX
