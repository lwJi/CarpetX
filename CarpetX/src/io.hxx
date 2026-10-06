#ifndef CARPETX_CARPETX_IO_HXX
#define CARPETX_CARPETX_IO_HXX

#include <cctk.h>

#include <string>

namespace CarpetX {

void RecoverGridStructure(cGH *cctkGH);
void RecoverGH(const cGH *cctkGH);
void InputGH(const cGH *cctkGH);

// A checkpoint="yes" group with storage has no dataset in the checkpoint being
// recovered. Names the group, file, iteration, the dataset looked up, and the
// two ways out (restart from a current checkpoint, or checkpoint="no" +
// recompute IN CarpetX_RecomputeAfterRecovery). The readers are shared with
// the file reader (`InputGH`), which gets a message without the checkpoint
// advice. Never returns.
[[noreturn]] void
error_missing_checkpoint_group(const cGH *cctkGH, int groupindex,
                               const std::string &checkpoint_path,
                               const std::string &dataset);

int OutputGH(const cGH *cctkGH);

} // namespace CarpetX

#endif // #ifndef CARPETX_CARPETX_IO_HXX
