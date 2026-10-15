#ifndef CARPETX_CARPETX_IO_ADIOS2_HXX
#define CARPETX_CARPETX_IO_ADIOS2_HXX

#include <cctk.h>

#ifdef HAVE_CAPABILITY_ADIOS2

#include "io.hxx"

#include <string>
#include <vector>

namespace CarpetX {

// int InputADIOS2Parameters(const std::string &input_dir,
//                         const std::string &input_file);
// void InputADIOS2GridStructure(cGH *cctkGH, const std::string &input_dir,
//                             const std::string &input_file, int
//                             input_iteration);
// void InputADIOS2(const cGH *cctkGH, const std::vector<bool> &input_group,
//                const std::string &input_dir, const std::string &input_file);

// Writes the grid functions on `levels` only
void OutputADIOS2(const cGH *cctkGH, const std::vector<bool> &output_group,
                  const std::string &output_dir, const std::string &output_file,
                  const level_range_t &levels);

} // namespace CarpetX

#endif

#endif // #ifndef CARPETX_CARPETX_IO_ADIOS2_HXX
