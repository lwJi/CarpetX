#ifndef CARPETX_CARPETX_IO_OPENPMD_HXX
#define CARPETX_CARPETX_IO_OPENPMD_HXX

#include "internal.hxx"

#include "io_slice.hxx"

#include <cctk.h>

#include <string>
#include <vector>

namespace CarpetX {

void OutputOpenPMD(const cGH *cctkGH, const std::vector<bool> &output_group,
                   const std::string &output_dir,
                   const std::string &output_file, const slice_t &slice);

} // namespace CarpetX

#endif // #ifndef CARPETX_CARPETX_IO_OPENPMD_HXX
