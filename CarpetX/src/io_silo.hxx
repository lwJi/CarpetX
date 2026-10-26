#ifndef CARPETX_CARPETX_IO_SILO_HXX
#define CARPETX_CARPETX_IO_SILO_HXX

#include "internal.hxx"

#include <cctk.h>

#ifdef HAVE_CAPABILITY_Silo

#include "io_slice.hxx"

#include <optional>
#include <string>
#include <vector>

namespace CarpetX {

void InputSilo(const cGH *cctkGH, const std::vector<bool> &input_group,
               const std::string &input_dir, const std::string &input_file);

void OutputSilo(const cGH *cctkGH, const std::vector<bool> &output_group,
                const std::string &output_dir, const std::string &output_file,
                std::optional<slice_t> slice = std::nullopt);

} // namespace CarpetX

#endif

#endif // #ifndef CARPETX_CARPETX_IO_SILO_HXX
