#ifndef CARPETX_CARPETX_IO_TSV_HXX
#define CARPETX_CARPETX_IO_TSV_HXX

#ifndef CARPETX_INTERNAL
#error                                                                         \
    "CarpetX internal header; include <carpetx.hxx> instead (see agent_docs/carpetx-public-api.md)"
#endif

#include <cctk.h>

namespace CarpetX {

void OutputTSVold(const cGH *restrict cctkGH);
void OutputTSV(const cGH *restrict cctkGH);

} // namespace CarpetX

#endif // #ifndef CARPETX_CARPETX_IO_TSV_HXX
