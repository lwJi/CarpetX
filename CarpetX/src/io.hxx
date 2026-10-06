#ifndef CARPETX_CARPETX_IO_HXX
#define CARPETX_CARPETX_IO_HXX

#include <cctk.h>

#include <string>

namespace CarpetX {

void RecoverGridStructure(cGH *cctkGH);
void RecoverGH(const cGH *cctkGH);
void InputGH(const cGH *cctkGH);

int OutputGH(const cGH *cctkGH);

// A group selected for input has no dataset in the file being read. Called by
// the readers at the lookup that decides "is this group in the file"; a
// dataset that exists but has the wrong shape is a corrupt file and stays an
// assertion there.
//
// During recovery (`RecoverGH`) the selected groups are exactly the
// checkpointed ones (`checkpoint="yes"`, the default) that have storage, so a
// missing dataset means the checkpoint predates the group being checkpointed.
// The message names the group, the file, the iteration, the dataset looked up
// and the two ways out: restart from a checkpoint written with the current
// thorns, or tag the group `checkpoint="no"` and recompute it `IN
// CarpetX_RecomputeAfterRecovery`. Nothing is recovered silently.
//
// The file reader (`InputGH`) shares the readers; there the group was chosen
// via `CarpetX::filereader_ID_vars` and the message says that instead.
//
// Never returns.
[[noreturn]] void
error_missing_checkpoint_group(const cGH *cctkGH, int groupindex,
                               const std::string &checkpoint_path,
                               const std::string &dataset);

} // namespace CarpetX

#endif // #ifndef CARPETX_CARPETX_IO_HXX
