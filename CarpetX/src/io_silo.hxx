#ifndef CARPETX_CARPETX_IO_SILO_HXX
#define CARPETX_CARPETX_IO_SILO_HXX

#include "internal.hxx"

#include "io_slice.hxx"

#include <cctk.h>

#include <optional>
#include <string>
#include <vector>

namespace CarpetX {

void OutputSilo(const cGH *cctkGH, const std::vector<bool> &output_group,
                const std::string &output_dir, const std::string &output_file,
                const slice_t &slice);

void RegisterSiloOutput(const cGH *cctkGH,
                        const std::vector<bool> &output_group,
                        const std::string &output_dir,
                        const std::string &output_file,
                        const std::optional<slice_t> &slice);

} // namespace CarpetX

#endif // #ifndef CARPETX_CARPETX_IO_SILO_HXX
