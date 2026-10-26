#ifndef CARPETX_CARPETX_IO_SILO_HXX
#define CARPETX_CARPETX_IO_SILO_HXX

#include "internal.hxx"

#include <cctk.h>

#ifdef HAVE_CAPABILITY_Silo

#include "checkpoint.hxx"
#include "io_slice.hxx"

#include <optional>
#include <string>
#include <vector>

namespace CarpetX {

int InputSiloParameters(const std::string &input_dir,
                        const std::string &input_file);
checkpoint_record_t InputSiloGridStructure(const std::string &input_dir,
                                           const std::string &input_file,
                                           int input_iteration);
void InputSilo(const cGH *cctkGH, const std::vector<bool> &input_group,
               const std::string &input_dir, const std::string &input_file,
               const checkpoint_record_t *recovering);

void OutputSilo(const cGH *cctkGH, const std::vector<bool> &output_group,
                const std::string &output_dir, const std::string &output_file,
                std::optional<slice_t> slice = std::nullopt);

} // namespace CarpetX

#endif

#endif // #ifndef CARPETX_CARPETX_IO_SILO_HXX
