#ifndef CARPETX_CARPETX_IO_NORM_HXX
#define CARPETX_CARPETX_IO_NORM_HXX

#ifndef CARPETX_INTERNAL
#error                                                                         \
    "CarpetX internal header; include <carpetx.hxx> instead (see agent_docs/carpetx-public-api.md)"
#endif

#include <cctk.h>

namespace CarpetX {

void OutputNorms(const cGH *restrict cctkGH);

}

#endif // #define CARPETX_CARPETX_IO_NORM_HXX
