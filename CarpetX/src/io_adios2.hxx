#ifndef CARPETX_CARPETX_IO_ADIOS2_HXX
#define CARPETX_CARPETX_IO_ADIOS2_HXX

#include <cctk.h>

#ifdef HAVE_CAPABILITY_ADIOS2

#include <string>
#include <vector>

namespace CarpetX {

void OutputADIOS2(const cGH *cctkGH, const std::vector<bool> &output_group,
                  const std::string &output_dir,
                  const std::string &output_file);

} // namespace CarpetX

#endif

#endif
