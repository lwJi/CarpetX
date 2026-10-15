#ifndef CARPETX_CARPETX_IO_SILO_HXX
#define CARPETX_CARPETX_IO_SILO_HXX

#include <cctk.h>

#ifdef HAVE_CAPABILITY_Silo

#include "io.hxx"
#include "io_slice.hxx"

#include <optional>
#include <string>
#include <vector>

namespace CarpetX {

int InputSiloParameters(const std::string &input_dir,
                        const std::string &input_file);
void InputSiloGridStructure(cGH *cctkGH, const std::string &input_dir,
                            const std::string &input_file, int input_iteration);
void InputSilo(const cGH *cctkGH, const std::vector<bool> &input_group,
               const std::string &input_dir, const std::string &input_file);

// Writes the grid functions on `levels` only, with their per-level driver
// metadata; the driver's nlevels still counts every level
void OutputSilo(const cGH *cctkGH, const std::vector<bool> &output_group,
                const std::string &output_dir, const std::string &output_file,
                std::optional<slice_t> slice = std::nullopt,
                const level_range_t &levels = every_level);

} // namespace CarpetX

#endif

#endif // #ifndef CARPETX_CARPETX_IO_SILO_HXX
