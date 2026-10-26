#ifndef CARPETX_CARPETX_IO_OPENPMD_HXX
#define CARPETX_CARPETX_IO_OPENPMD_HXX

#include "internal.hxx"

#include <cctk.h>

#ifdef HAVE_CAPABILITY_openPMD_api

#include "checkpoint.hxx"
#include "io_slice.hxx"

#include <optional>
#include <string>
#include <vector>

namespace CarpetX {

enum class TimeLevelMode { Current, All };

int InputOpenPMDParameters(const std::string &input_dir,
                           const std::string &input_file);
checkpoint_record_t InputOpenPMDGridStructure(const std::string &input_dir,
                                              const std::string &input_file,
                                              int input_iteration);
void InputOpenPMD(const cGH *cctkGH, const std::vector<bool> &input_group,
                  const std::string &input_dir, const std::string &input_file,
                  TimeLevelMode tl_mode, const checkpoint_record_t *recovering);

void OutputOpenPMD(const cGH *cctkGH, const std::vector<bool> &output_group,
                   const std::string &output_dir,
                   const std::string &output_file,
                   TimeLevelMode tl_mode = TimeLevelMode::Current,
                   std::optional<slice_t> slice = std::nullopt);

} // namespace CarpetX

#endif

#endif // #ifndef CARPETX_CARPETX_IO_OPENPMD_HXX
