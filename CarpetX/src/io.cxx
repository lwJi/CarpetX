#include "driver.hxx"
#include "io.hxx"
#include "io_adios2.hxx"
#include "io_meta.hxx"
#include "io_norm.hxx"
#include "io_openpmd.hxx"
#include "io_silo.hxx"
#include "io_slice.hxx"
#include "io_tsv.hxx"
#include "schedule.hxx"
#include "subcycling.hxx"
#include <carpetx.hxx>

#include <CactusBase/IOUtil/src/ioGH.h>
#include <CactusBase/IOUtil/src/ioutil_CheckpointRecovery.h>

#include <cctk.h>
#include <cctk_Arguments.h>
#include <cctk_IOMethods.h>
#include <cctk_Parameters.h>
#include <util_Table.h>

#include <AMReX.H>
#include <AMReX_Orientation.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_VisMF.H>

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <regex>
#include <string>
#include <utility>
#include <vector>

namespace CarpetX {

////////////////////////////////////////////////////////////////////////////////

namespace {
std::vector<bool> find_groups(const char *const method,
                              const char *const out_vars) {
  DECLARE_CCTK_PARAMETERS;

  std::vector<bool> enabled(CCTK_NumGroups(), false);
  const auto callback{
      [](const int index, const char *const optstring, void *const arg) {
        std::vector<bool> &enabled = *static_cast<std::vector<bool> *>(arg);
        enabled.at(CCTK_GroupIndexFromVarI(index)) = true;
      }};
  CCTK_TraverseString(out_vars, callback, &enabled, CCTK_GROUP_OR_VAR);
  if (verbose) {
    CCTK_VINFO("%s output for groups:", method);
    for (int gi = 0; gi < CCTK_NumGroups(); ++gi)
      if (enabled.at(gi))
        CCTK_VINFO("  %s", CCTK_FullGroupName(gi));
  }
  return enabled;
}

std::string get_parameter_filename() {
  std::vector<char> buf(10000);
  int ilen = CCTK_ParameterFilename(buf.size(), buf.data());
  assert(ilen < int(buf.size() - 1));
  std::string parfilename(buf.data());
  // Remove directory prefix, if any
  auto slash = parfilename.rfind('/');
  if (slash != std::string::npos)
    parfilename = parfilename.substr(slash + 1);
  // Remove suffix, if it is there
  auto suffix = parfilename.rfind('.');
  if (suffix != std::string::npos && parfilename.substr(suffix) == ".par")
    parfilename = parfilename.substr(0, suffix);
  return parfilename;
}

std::string get_simulation_name() {
  std::string name = get_parameter_filename();
  const size_t last_slash = name.rfind('/');
  if (last_slash != std::string::npos && last_slash < name.length())
    name = name.substr(last_slash + 1);
  const size_t last_dot = name.rfind('.');
  if (last_dot != std::string::npos && last_dot > 0)
    name = name.substr(0, last_dot);
  return name;
}
} // namespace

////////////////////////////////////////////////////////////////////////////////

// Recovering

int recover_iteration = -1;

int recovered_iteration() { return recover_iteration; }

extern "C" int CarpetX_RecoverParameters() {
  DECLARE_CCTK_PARAMETERS;

  static Timer timer("RecoverParameters");
  Interval interval(timer);

  recover_iteration =
      make_codec(recover_method)->read_parameters(recover_dir, recover_file);

  return recover_iteration >= 0;
}

////////////////////////////////////////////////////////////////////////////////

checkpoint_record_t RecoverGridStructure(cGH *restrict cctkGH) {
  DECLARE_CCTK_PARAMETERS;

  const std::unique_ptr<checkpoint_codec_t> codec = make_codec(recover_method);
  refuse_unless_recoverable(codec->limits());
  checkpoint_record_t record =
      codec->decode_structure(recover_dir, recover_file, recover_iteration);
  apply_structure(cctkGH, record);
  return record;
}

////////////////////////////////////////////////////////////////////////////////

void RecoverGH(const cGH *restrict cctkGH, const checkpoint_record_t &record) {
  DECLARE_CCTK_ARGUMENTS;
  DECLARE_CCTK_PARAMETERS;

  static Timer timer("RecoverGH");
  Interval interval(timer);

  const bool is_root = CCTK_MyProc(nullptr) == 0;
  if (is_root) {
    const int runtime = CCTK_RunTime(); // seconds
    CCTK_VINFO("RecoverGH: iteration %d, time %f, run time %.2f h",
               cctk_iteration, double(cctk_time), double(runtime / 3600));
  }

  const std::vector<bool> group_enabled = [&] {
    std::vector<bool> enabled(CCTK_NumGroups(), false);
    for (const auto &group : ghext->records())
      enabled.at(group.groupindex.index()) = group.do_checkpoint;
    return enabled;
  }();

  if (ghext->use_subcycling) {
    for (const auto &patchdata : ghext->patches())
      for (const auto &leveldata : patchdata.levels()) {
        if (leveldata.level.index() == 0)
          continue;
        for (const auto &groupdata : leveldata.groups())
          EnsureRKBuffers(leveldata.level, groupdata.record.groupindex);
      }
  }

  const std::unique_ptr<checkpoint_codec_t> codec = make_codec(recover_method);
  checkpoint_record_t recovering = record;
  build_buffers(recovering, group_enabled, TimeLevelMode::All, codec->limits());
  read_record(*codec, recovering, recover_dir, recover_file, "RecoverGH");

  // Increment epoch
  carpetx_epoch.fetch_add(1, std::memory_order_relaxed);
}

void InputGH(const cGH *restrict cctkGH) {
  DECLARE_CCTK_ARGUMENTS;
  DECLARE_CCTK_PARAMETERS;

  static Timer timer("InputGH");
  Interval interval(timer);

  if (filereader_ID_files[0] == '\0')
    return;

  const bool is_root = CCTK_MyProc(nullptr) == 0;
  if (is_root) {
    const int runtime = CCTK_RunTime(); // seconds
    CCTK_VINFO("InputGH: iteration %d, time %f, run time %.2f h",
               cctk_iteration, double(cctk_time), double(runtime / 3600.0));
  }

  if (CCTK_EQUALS(filereader_method, "openpmd")) {

#ifdef HAVE_CAPABILITY_openPMD_api
    const std::vector<bool> input_group =
        find_groups("openPMD", filereader_ID_vars);
    const std::string simulation_name = get_simulation_name();
    InputOpenPMD(cctkGH, input_group, filereader_ID_dir, filereader_ID_files,
                 TimeLevelMode::Current, nullptr);
#else
    // TODO: Check this at paramcheck
    CCTK_VERROR(
        "CarpetX::filereader_method is set to \"openpmd\", but openPMD_api "
        "is not enabled");
#endif

  } else if (CCTK_EQUALS(filereader_method, "silo")) {

    const std::vector<bool> input_group =
        find_groups("Silo", filereader_ID_vars);
#ifdef HAVE_CAPABILITY_Silo
    const std::string simulation_name = get_simulation_name();
    InputSilo(cctkGH, input_group, filereader_ID_dir, filereader_ID_files);
#else
    // TODO: Check this at paramcheck
    CCTK_VERROR("CarpetX::filereader_method is set to \"silo\", but Silo is "
                "not enabled");
#endif

  } else {
    CCTK_ERROR("unknown value for paramater CarpetX::filereader_method");
  }
}

////////////////////////////////////////////////////////////////////////////////

void OutputPlotfile(const cGH *restrict cctkGH) {
  DECLARE_CCTK_ARGUMENTS;
  DECLARE_CCTK_PARAMETERS;

  static Timer timer("OutputPlotfile");
  Interval interval(timer);

  const int numgroups = CCTK_NumGroups();
  std::vector<bool> group_enabled(numgroups, false);
  auto enable_group{[](int index, const char *optstring, void *callback) {
    std::vector<bool> &group_enabled =
        *static_cast<std::vector<bool> *>(callback);
    // CCTK_TraverseString expands the given groups into their variables
    // so I condense it down again to groups only
    group_enabled.at(CCTK_GroupIndexFromVarI(index)) = true;
  }};
  CCTK_TraverseString(out_plotfile_groups, enable_group, &group_enabled,
                      CCTK_GROUP_OR_VAR);

  for (int gi0 = 0; gi0 < numgroups; ++gi0) {
    const gi_t gi{gi0};
    if (!group_enabled.at(gi.index()))
      continue;
    const auto &group = ghext->record(gi);
    if (group.kind != group_kind::grid_function)
      continue;

    if (group.has_storage()) {
      const tl_t tl{0};

      std::string groupname = CCTK_FullGroupName(gi.index());
      groupname = regex_replace(groupname, std::regex("::"), "-");
      for (auto &c : groupname)
        c = tolower(c);

      for (const auto &restrict patchdata : ghext->patches()) {

        const std::string basename = [&]() {
          std::ostringstream buf;
          buf << groupname << ".it" << std::setw(8) << std::setfill('0')
              << cctk_iteration;
          if (ghext->num_patches() > 1)
            buf << ".m" << std::setw(2) << std::setfill('0') << patchdata.patch;
          return buf.str();
        }();
        const std::string filename = std::string(out_dir) + "/" + basename;

        amrex::Vector<std::string> varnames(group.numvars);
        for (const vi_t vi : group.variables()) {
          std::ostringstream buf;
          buf << CCTK_VarName(group.var(vi).index());
          for (int i = 0; i < tl.index(); ++i)
            buf << "_p";
          varnames.at(vi.index()) = buf.str();
        }

        amrex::Vector<const amrex::MultiFab *> mfabs(patchdata.num_levels());
        amrex::Vector<amrex::Geometry> geoms(patchdata.num_levels());
        amrex::Vector<int> iters(patchdata.num_levels());
        amrex::Vector<amrex::IntVect> reffacts(patchdata.num_levels());
        for (const auto &restrict leveldata : patchdata.levels()) {
          const int level = leveldata.level.index();
          mfabs.at(level) = &*leveldata.group(gi).timelevel(tl).mfab;
          geoms.at(level) = patchdata.amrcore->Geom(level);
          iters.at(level) = cctk_iteration;
          reffacts.at(level) = amrex::IntVect{2, 2, 2};
        }

        // TODO: Output all groups into a single file
        WriteMultiLevelPlotfile(filename, mfabs.size(), mfabs, varnames, geoms,
                                cctk_time, iters, reffacts);

        const bool is_root = CCTK_MyProc(nullptr) == 0;
        if (is_root) {
          const std::string visitname = [&]() {
            std::ostringstream buf;
            buf << out_dir << "/" << groupname << ".visit";
            return buf.str();
          }();
          std::ofstream visit(visitname, std::ios::app);
          assert(visit.good());
          visit << basename << "/Header\n";
          visit.close();
        } // if is_root
      } // for patchdata
    }
  }
}

////////////////////////////////////////////////////////////////////////////////

struct parameters {};
YAML::Emitter &operator<<(YAML::Emitter &yaml, parameters) {
  // Collect all parameters and their values
  std::vector<std::pair<std::string, const cParamData *> > parameter_values;
  int first = 1;
  for (;;) {
    const cParamData *data;
    // This call is not thread safe
    const int iret = CCTK_ParameterWalk(first, nullptr, nullptr, &data);
    assert(iret >= 0);
    if (iret)
      break;
    first = 0;
    // Ignore inactive thorns
    if (!CCTK_IsThornActive(data->thorn))
      continue;
    const std::string fullname = std::string(data->thorn) + "::" + data->name;
    parameter_values.emplace_back(fullname, data);
  }

  // Sort parameters alphabetically
  std::sort(parameter_values.begin(), parameter_values.end());

  // Output parameters
  yaml << YAML::LocalTag("parameters-1.0.0");
  yaml << YAML::BeginMap;
  for (const auto &parameter_value : parameter_values) {
    const std::string &fullname = parameter_value.first;
    const cParamData *const data = parameter_value.second;
    yaml << YAML::Key << fullname << YAML::Value << YAML::Flow;
    int type;
    const void *const pvalue =
        CCTK_ParameterGet(data->name, data->thorn, &type);
    switch (type) {
    case PARAMETER_KEYWORD:
    case PARAMETER_STRING:
      yaml << *static_cast<const char *const *>(pvalue);
      break;
    case PARAMETER_INT:
      yaml << *static_cast<const CCTK_INT *>(pvalue);
      break;
    case PARAMETER_REAL:
      yaml << *static_cast<const CCTK_REAL *>(pvalue);
      break;
    case PARAMETER_BOOLEAN:
      yaml << static_cast<bool>(*static_cast<const CCTK_INT *>(pvalue));
      break;
    default:
      assert(0);
    }
  }
  yaml << YAML::EndMap;

  return yaml;
}

void OutputMetadata(const cGH *restrict cctkGH) {
  DECLARE_CCTK_ARGUMENTS;
  DECLARE_CCTK_PARAMETERS;

  if (!out_metadata)
    return;

  static Timer timer("OutputMetadata");
  Interval interval(timer);

  const bool is_root = CCTK_MyProc(nullptr) == 0;
  if (!is_root)
    return;

  {
    YAML::Emitter yaml;
    yaml << YAML::Comment("CarpetX Metadata");
    yaml << YAML::BeginDoc;
    yaml << YAML::LocalTag("carpetx-metadata-1.0.0");
    yaml << YAML::BeginMap;
    yaml << YAML::Key << "nghostzones";
    yaml << YAML::Value << YAML::Flow << YAML::BeginSeq;
    for (int d = 0; d < dim; ++d)
      yaml << cctk_nghostzones[d];
    yaml << YAML::EndSeq;
    yaml << YAML::Key << "iteration";
    yaml << YAML::Value << cctk_iteration;
    yaml << YAML::Key << "time";
    yaml << YAML::Value << cctk_time;
    yaml << YAML::Key << "delta_time";
    yaml << YAML::Value << cctk_delta_time;
    yaml << YAML::Key << "ghext";
    yaml << YAML::Value << *ghext;
    yaml << YAML::EndMap;
    yaml << YAML::EndDoc;

    std::ostringstream buf;
    buf << out_dir << "/carpetx-metadata"
        << ".it" << std::setw(8) << std::setfill('0') << cctk_iteration
        << ".yaml";
    const std::string filename = buf.str();
    std::ofstream file(filename.c_str(), std::ofstream::out);
    file << yaml.c_str();
  }

  {
    YAML::Emitter yaml;
    yaml << YAML::Comment("Parameters");
    yaml << YAML::BeginDoc;
    yaml << parameters();
    yaml << YAML::EndDoc;

    std::ostringstream buf;
    buf << out_dir << "/parameters"
        << ".it" << std::setw(8) << std::setfill('0') << cctk_iteration
        << ".yaml";
    const std::string filename = buf.str();
    std::ofstream file(filename.c_str(), std::ofstream::out);
    file << yaml.c_str();
  }
}

int OutputGH(const cGH *restrict cctkGH) {
  DECLARE_CCTK_ARGUMENTS;
  DECLARE_CCTK_PARAMETERS;

  static Timer timer("OutputGH");
  Interval interval(timer);

  const bool is_root = CCTK_MyProc(nullptr) == 0;
  if (is_root && verbose)
    CCTK_VINFO("OutputGH: iteration %d, time %f, run time %d s", cctk_iteration,
               double(cctk_time), CCTK_RunTime());

  {
    const int every = out_metadata_every == -1 ? out_every : out_metadata_every;
    if (every > 0 && cctk_iteration % every == 0)
      OutputMetadata(cctkGH);
  }

  {
    const int every = out_norm_every == -1 ? out_every : out_norm_every;
    if (every > 0 && cctk_iteration % every == 0)
      OutputNorms(cctkGH);
  }

  {
    const int every = out_adios2_every == -1 ? out_every : out_adios2_every;
    if (every > 0 && cctk_iteration % every == 0) {
      const std::vector<bool> group_enabled =
          find_groups("ADIOS2", out_adios2_vars);
#ifdef HAVE_CAPABILITY_ADIOS2
      // TODO: Stop at paramcheck time when ADIOS2 output parameters
      // are set, but ADIOS2 is not available
      const std::string simulation_name = get_simulation_name();
      OutputADIOS2(cctkGH, group_enabled, out_dir, simulation_name);
#else
      if (strlen(out_adios2_vars) != 0)
        CCTK_VERROR("ADIOS2 is not enabled. The parameter "
                    "CarpetX::out_adios2_vars must be empty.");
#endif
    }
  }

  {
    const int every = out_openpmd_every == -1 ? out_every : out_openpmd_every;
    if (every > 0 && cctk_iteration % every == 0) {
      const std::vector<bool> group_enabled =
          find_groups("openPMD", out_openpmd_vars);
#ifdef HAVE_CAPABILITY_openPMD_api
      // TODO: Stop at paramcheck time when openPMD output parameters
      // are set, but openPMD is not available
      const std::string simulation_name = get_simulation_name();
      OutputOpenPMD(cctkGH, group_enabled, out_dir, simulation_name);
#else
      if (strlen(out_openpmd_vars) != 0)
        CCTK_VERROR("openPMD is not enabled. The parameter "
                    "CarpetX::out_openpmd_vars must be empty.");
#endif
    }
  }

  {
    const int every =
        out_openpmd_2d_every == -1 ? out_every : out_openpmd_2d_every;
    if (every > 0 && cctk_iteration % every == 0) {
      const std::vector<bool> group_enabled =
          find_groups("openPMD-2D", out_openpmd_2d_vars);
#ifdef HAVE_CAPABILITY_openPMD_api
      const std::string simulation_name = get_simulation_name();
      // xy plane: normal = z; xz: normal = y; yz: normal = x
      OutputOpenPMD(cctkGH, group_enabled, out_dir, simulation_name + ".xy",
                    TimeLevelMode::Current, slice_t{2, out_xyplane_z});
      OutputOpenPMD(cctkGH, group_enabled, out_dir, simulation_name + ".xz",
                    TimeLevelMode::Current, slice_t{1, out_xzplane_y});
      OutputOpenPMD(cctkGH, group_enabled, out_dir, simulation_name + ".yz",
                    TimeLevelMode::Current, slice_t{0, out_yzplane_x});
#else
      if (strlen(out_openpmd_2d_vars) != 0)
        CCTK_VERROR("openPMD is not enabled. The parameter "
                    "CarpetX::out_openpmd_2d_vars must be empty.");
#endif
    }
  }

  {
    const int every = out_plotfile_every == -1 ? out_every : out_plotfile_every;
    if (every > 0 && cctk_iteration % every == 0)
      OutputPlotfile(cctkGH);
  }

  {
    const int every = out_silo_every == -1 ? out_every : out_silo_every;
    if (every > 0 && cctk_iteration % every == 0) {
      const std::vector<bool> group_enabled =
          find_groups("Silo", out_silo_vars);
#ifdef HAVE_CAPABILITY_Silo
      // TODO: Stop at paramcheck time when Silo output parameters are
      // set, but Silo is not available
      const std::string simulation_name = get_simulation_name();
      OutputSilo(cctkGH, group_enabled, out_dir, simulation_name);
#else
      if (strlen(out_silo_vars) != 0)
        CCTK_VERROR("Silo is not enabled. The parameter "
                    "CarpetX::out_silo_vars must be empty.");
#endif
    }
  }

  {
    const int every = out_silo_2d_every == -1 ? out_every : out_silo_2d_every;
    if (every > 0 && cctk_iteration % every == 0) {
      const std::vector<bool> group_enabled =
          find_groups("Silo-2D", out_silo_2d_vars);
#ifdef HAVE_CAPABILITY_Silo
      const std::string simulation_name = get_simulation_name();
      // xy plane: normal = z; xz: normal = y; yz: normal = x
      OutputSilo(cctkGH, group_enabled, out_dir, simulation_name + ".xy",
                 slice_t{2, out_xyplane_z});
      OutputSilo(cctkGH, group_enabled, out_dir, simulation_name + ".xz",
                 slice_t{1, out_xzplane_y});
      OutputSilo(cctkGH, group_enabled, out_dir, simulation_name + ".yz",
                 slice_t{0, out_yzplane_x});
#else
      if (strlen(out_silo_2d_vars) != 0)
        CCTK_VERROR("Silo is not enabled. The parameter "
                    "CarpetX::out_silo_2d_vars must be empty.");
#endif
    }
  }

  OutputTSVold(cctkGH);

  {
    const int every = out_tsv_every == -1 ? out_every : out_tsv_every;
    if (every > 0 && cctk_iteration % every == 0)
      OutputTSV(cctkGH);
  }

  // Call the IO methods registered via CCTK_RegisterIOMethod; the flesh's
  // default traversal never runs since CCTK_OutputGH is overloaded.
  {
    const int num_methods = CCTK_NumIOMethods();
    for (int handle = 0; handle < num_methods; ++handle) {
      const IOMethod *const method = CCTK_IOMethod(handle);
      if (method && method->OutputGH)
        method->OutputGH(cctkGH);
    }
  }

  // Describe all output files
  OutputMeta(cctkGH);

  if (is_root && verbose)
    CCTK_VINFO("OutputGH done.");

  // TODO: This should be the number of variables output
  return 0;
}

////////////////////////////////////////////////////////////////////////////////

void Checkpoint(const cGH *const restrict cctkGH) {
  DECLARE_CCTK_PARAMETERS;

  static int last_checkpoint_iteration = -1;

  if (cctkGH->cctk_iteration == recover_iteration) {
    CCTK_VINFO(
        "Recovered from checkpoint at iteration %d; skipping checkpointing",
        cctkGH->cctk_iteration);
    return;
  }

  if (cctkGH->cctk_iteration <= last_checkpoint_iteration) {
    CCTK_VINFO(
        "Already wrote checkpoint at iteration %d; skipping checkpointing",
        cctkGH->cctk_iteration);
    return;
  }
  last_checkpoint_iteration = cctkGH->cctk_iteration;

  static Timer timer("Checkpoint");
  Interval interval(timer);

  const std::vector<bool> checkpoint_group = [&] {
    std::vector<bool> enabled(CCTK_NumGroups(), false);
    for (const auto &group : ghext->records())
      enabled.at(group.groupindex.index()) = group.do_checkpoint;
    return enabled;
  }();
  const std::unique_ptr<checkpoint_codec_t> codec =
      make_codec(checkpoint_method);
  const checkpoint_record_t record = build_record(
      cctkGH, checkpoint_group, TimeLevelMode::All, codec->limits());
  codec->encode(record, checkpoint_dir, checkpoint_file);
}

namespace {

// Trigger state for `IO::checkpoint_every_walltime_hours`.
//
// Every checkpoint trigger mechanism is independent: it consults only its own
// state, never checkpoints written by the other mechanisms.
//
// This is the run time (seconds since the start of this run) at which the
// walltime trigger last fired; 0 means it has not fired yet. It is advanced
// whenever the trigger fires, even if the resulting checkpoint is then skipped
// because one had already been written for this iteration.
int last_walltime_trigger_runtime = 0; // seconds

// Whether the walltime criterion `IO::checkpoint_every_walltime_hours` is met
// at run time `runtime` (seconds since the start of this run), advancing the
// trigger's state when it is.
//
// The criterion is anchored to the start of the run, not to the previous
// checkpoint: slot boundaries lie at run times N, 2N, 3N, ... for an interval
// of N seconds, and the trigger fires at the first evaluation point that falls
// into a later slot than the previous firing did. The long-term rate is thus
// exactly one checkpoint per N, with no drift accumulating from the delay
// between a boundary and the next evaluation point.
//
// When a single time step spans several slots, the state snaps to the current
// slot instead of advancing one slot at a time, so that the slots skipped over
// do not make the trigger fire again on the following time steps.
//
// Both callers are scheduled with `OPTIONS: meta`, so this runs once per
// iteration. It must be called on every process with the same `runtime`, since
// it mutates state and decides whether a collective operation happens.
bool CheckWalltimeTrigger(const int runtime) {
  DECLARE_CCTK_PARAMETERS;

  if (!(checkpoint_every_walltime_hours > 0))
    return false;
  // Round to whole seconds, the resolution of `CCTK_RunTime`, but keep the
  // interval positive so that the slot arithmetic below is well defined
  const int interval =
      std::max(int(lrint(checkpoint_every_walltime_hours * 3600)), 1);
  if (runtime / interval <= last_walltime_trigger_runtime / interval)
    return false;
  last_walltime_trigger_runtime = runtime;
  return true;
}

// Name the triggers that fired, for the log message
std::string DescribeTriggers(const char *const name1, const bool fired1,
                             const char *const name2, const bool fired2) {
  std::string description;
  if (fired1)
    description += name1;
  if (fired2) {
    if (!description.empty())
      description += ", ";
    description += name2;
  }
  return description;
}

} // namespace

extern "C" void CarpetX_CheckpointInitial(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTS;
  DECLARE_CCTK_PARAMETERS;

  int runtime = CCTK_RunTime(); // seconds
  MPI_Bcast(&runtime, 1, MPI_INT, 0, MPI_COMM_WORLD);

  // Setting up initial conditions can take a long time, so the walltime
  // criterion is evaluated here as well and not only during the evolution
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

  int runtime = CCTK_RunTime(); // seconds
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
    const int runtime = CCTK_RunTime(); // seconds
    CCTK_VINFO("Checkpointing before terminating at iteration %d, time %f, run "
               "time %.2f h",
               cctk_iteration, double(cctk_time), double(runtime / 3600.0));
    Checkpoint(cctkGH);
  }
}

} // namespace CarpetX
