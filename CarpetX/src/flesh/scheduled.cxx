#include "io/io.hxx"
#include "io/checkpoint.hxx"
#include "traversal/traversal.hxx"
#include <carpetx.hxx>

#include <cctk.h>
#include <cctk_Arguments.h>
#include <cctk_Parameters.h>

#include <mpi.h>

#include <memory>

namespace CarpetX {

extern "C" int CarpetX_RecoverParameters() {
  DECLARE_CCTK_PARAMETERS;

  static Timer timer("RecoverParameters");
  Interval interval(timer);

  record_recovered_iteration(
      make_codec(recover_method)->read_parameters(recover_dir, recover_file));

  return recovered_iteration() >= 0;
}

extern "C" void CarpetX_CheckpointInitial(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTS;
  DECLARE_CCTK_PARAMETERS;

  int runtime = CCTK_RunTime();
  MPI_Bcast(&runtime, 1, MPI_INT, 0, MPI_COMM_WORLD);

  const bool checkpoint_by_walltime = CheckWalltimeTrigger(runtime);

  if (checkpoint_ID || checkpoint_by_walltime) {
    CCTK_VINFO("Checkpointing initial conditions at iteration %d, time %f, run "
               "time %.2f h (triggered by %s)",
               cctk_iteration, double(cctk_time), double(runtime / 3600.0),
               DescribeTriggers("checkpoint_ID", checkpoint_ID,
                                "checkpoint_every_walltime_hours",
                                checkpoint_by_walltime)
                   .c_str());
    Checkpoint(cctkGH);
  }
}

extern "C" void CarpetX_Checkpoint(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTS;
  DECLARE_CCTK_PARAMETERS;

  int runtime = CCTK_RunTime();
  MPI_Bcast(&runtime, 1, MPI_INT, 0, MPI_COMM_WORLD);

  const bool checkpoint_by_iteration =
      checkpoint_every > 0 && cctk_iteration % checkpoint_every == 0;
  const bool checkpoint_by_walltime = CheckWalltimeTrigger(runtime);

  if (checkpoint_by_iteration || checkpoint_by_walltime) {
    CCTK_VINFO("Checkpointing at iteration %d, time %f, run time %.2f h "
               "(triggered by %s)",
               cctk_iteration, double(cctk_time), double(runtime / 3600.0),
               DescribeTriggers("checkpoint_every", checkpoint_by_iteration,
                                "checkpoint_every_walltime_hours",
                                checkpoint_by_walltime)
                   .c_str());
    Checkpoint(cctkGH);
  }
}

extern "C" void CarpetX_CheckpointTerminate(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTS;
  DECLARE_CCTK_PARAMETERS;

  if (checkpoint_on_terminate) {
    const int runtime = CCTK_RunTime();
    CCTK_VINFO("Checkpointing before terminating at iteration %d, time %f, run "
               "time %.2f h",
               cctk_iteration, double(cctk_time), double(runtime / 3600.0));
    Checkpoint(cctkGH);
  }
}

} // namespace CarpetX
