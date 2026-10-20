#include "driver.hxx"
#include "io.hxx"
#include "io_adios2.hxx"
#include "io_meta.hxx"
#include "io_norm.hxx"
#include "io_openpmd.hxx"
#include "io_silo.hxx"
#include "io_tsv.hxx"
#include "schedule.hxx"
#include "timer.hxx"

#include <CactusBase/IOUtil/src/ioGH.h>
#include <CactusBase/IOUtil/src/ioutil_CheckpointRecovery.h>

#include <cctk.h>
#include <cctk_Arguments.h>
#include <cctk_Parameters.h>
#include <util_Table.h>

#include <AMReX.H>
#include <AMReX_Orientation.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_VisMF.H>

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cmath>
#include <regex>
#include <string>
#include <utility>
#include <vector>

namespace CarpetX {

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
  auto slash = parfilename.rfind('/');
  if (slash != std::string::npos)
    parfilename = parfilename.substr(slash + 1);
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

int recover_iteration = -1;

extern "C" int CarpetX_RecoverParameters() {
  DECLARE_CCTK_PARAMETERS;

  static Timer timer("RecoverParameters");
  Interval interval(timer);

  if (CCTK_EQUALS(recover_method, "openpmd")) {

#ifdef HAVE_CAPABILITY_openPMD_api
    recover_iteration = InputOpenPMDParameters(recover_dir, recover_file);
#else
    CCTK_VERROR(
        "CarpetX::recover_method is set to \"openpmd\", but openPMD_api "
        "is not enabled");
#endif

  } else if (CCTK_EQUALS(recover_method, "silo")) {

#ifdef HAVE_CAPABILITY_Silo
    if (recover_iteration < 0)
      recover_iteration = InputSiloParameters(recover_dir, recover_file);
#else
    CCTK_VERROR("CarpetX::recover_method is set to \"silo\", but Silo "
                "is not enabled");
#endif

  } else {
    CCTK_ERROR("unknown value for paramater CarpetX::recover_method");
  }

  return recover_iteration >= 0;
}

void RecoverGridStructure(cGH *restrict cctkGH) {
  DECLARE_CCTK_ARGUMENTS;
  DECLARE_CCTK_PARAMETERS;

  if (CCTK_EQUALS(recover_method, "openpmd")) {

#ifdef HAVE_CAPABILITY_openPMD_api
    InputOpenPMDGridStructure(cctkGH, recover_dir, recover_file,
                              recover_iteration);
#else
    CCTK_VERROR(
        "CarpetX::recover_method is set to \"openpmd\", but openPMD_api "
        "is not enabled");
#endif

  } else if (CCTK_EQUALS(recover_method, "silo")) {

#ifdef HAVE_CAPABILITY_Silo
    InputSiloGridStructure(cctkGH, recover_dir, recover_file,
                           recover_iteration);
#else
    CCTK_VERROR("CarpetX::recover_method is set to \"silo\", but Silo "
                "is not enabled");
#endif

  } else {
    CCTK_ERROR("unknown value for paramater CarpetX::recover_method");
  }
}

void RecoverGH(const cGH *restrict cctkGH) {
  DECLARE_CCTK_ARGUMENTS;
  DECLARE_CCTK_PARAMETERS;

  static Timer timer("RecoverGH");
  Interval interval(timer);

  const bool is_root = CCTK_MyProc(nullptr) == 0;
  if (is_root) {
    const int runtime = CCTK_RunTime();
    CCTK_VINFO("RecoverGH: iteration %d, time %f, run time %.2f h",
               cctk_iteration, double(cctk_time), double(runtime / 3600));
  }

  const std::vector<bool> group_enabled = [&] {
    std::vector<bool> enabled(CCTK_NumGroups(), false);
    for (int gi = 0; gi < CCTK_NumGroups(); ++gi) {
      const GHExt::GlobalData &globaldata = ghext->globaldata;
      if (globaldata.arraygroupdata.at(gi)) {
        enabled.at(gi) = globaldata.arraygroupdata.at(gi)->do_checkpoint;
      } else {
        const int patch = 0;
        const int level = 0;
        const GHExt::PatchData::LevelData &leveldata =
            ghext->patchdata.at(patch).leveldata.at(level);
        assert(leveldata.groupdata.at(gi));
        enabled.at(gi) = leveldata.groupdata.at(gi)->do_checkpoint;
      }
    }
    return enabled;
  }();

  if (CCTK_EQUALS(recover_method, "openpmd")) {

#ifdef HAVE_CAPABILITY_openPMD_api
    InputOpenPMD(cctkGH, group_enabled, recover_dir, recover_file);
#else
    CCTK_VERROR(
        "CarpetX::recover_method is set to \"openpmd\", but openPMD_api "
        "is not enabled");
#endif

  } else if (CCTK_EQUALS(recover_method, "silo")) {

#ifdef HAVE_CAPABILITY_Silo
    InputSilo(cctkGH, group_enabled, recover_dir, recover_file);
#else
    CCTK_VERROR("CarpetX::recover_method is set to \"silo\", but Silo "
                "is not enabled");
#endif

  } else {
    CCTK_ERROR("unknown value for paramater CarpetX::recover_method");
  }

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
    const int runtime = CCTK_RunTime();
    CCTK_VINFO("InputGH: iteration %d, time %f, run time %.2f h",
               cctk_iteration, double(cctk_time), double(runtime / 3600.0));
  }

  if (CCTK_EQUALS(filereader_method, "openpmd")) {

#ifdef HAVE_CAPABILITY_openPMD_api
    const std::vector<bool> input_group =
        find_groups("openPMD", filereader_ID_vars);
    const std::string simulation_name = get_simulation_name();
    InputOpenPMD(cctkGH, input_group, filereader_ID_dir, filereader_ID_files);
#else
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
    CCTK_VERROR("CarpetX::filereader_method is set to \"silo\", but Silo is "
                "not enabled");
#endif

  } else {
    CCTK_ERROR("unknown value for paramater CarpetX::filereader_method");
  }
}

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
    group_enabled.at(CCTK_GroupIndexFromVarI(index)) = true;
  }};
  CCTK_TraverseString(out_plotfile_groups, enable_group, &group_enabled,
                      CCTK_GROUP_OR_VAR);

  for (int gi = 0; gi < numgroups; ++gi) {
    if (!group_enabled.at(gi))
      continue;
    if (CCTK_GroupTypeI(gi) != CCTK_GF)
      continue;

    auto &restrict groupdata0 =
        *ghext->patchdata.at(0).leveldata.at(0).groupdata.at(gi);
    if (groupdata0.mfab.size() > 0) {
      const int tl = 0;

      std::string groupname = CCTK_FullGroupName(gi);
      groupname = regex_replace(groupname, std::regex("::"), "-");
      for (auto &c : groupname)
        c = tolower(c);

      for (const auto &restrict patchdata : ghext->patchdata) {

        const std::string basename = [&]() {
          std::ostringstream buf;
          buf << groupname << ".it" << std::setw(8) << std::setfill('0')
              << cctk_iteration;
          if (ghext->num_patches() > 1)
            buf << ".m" << std::setw(2) << std::setfill('0') << patchdata.patch;
          return buf.str();
        }();
        const std::string filename = std::string(out_dir) + "/" + basename;

        amrex::Vector<std::string> varnames(groupdata0.numvars);
        for (int vi = 0; vi < groupdata0.numvars; ++vi) {
          std::ostringstream buf;
          buf << CCTK_VarName(groupdata0.firstvarindex + vi);
          for (int i = 0; i < tl; ++i)
            buf << "_p";
          varnames.at(vi) = buf.str();
        }

        amrex::Vector<const amrex::MultiFab *> mfabs(
            patchdata.leveldata.size());
        amrex::Vector<amrex::Geometry> geoms(patchdata.leveldata.size());
        amrex::Vector<int> iters(patchdata.leveldata.size());
        amrex::Vector<amrex::IntVect> reffacts(patchdata.leveldata.size());
        for (const auto &restrict leveldata : patchdata.leveldata) {
          mfabs.at(leveldata.level) = &*leveldata.groupdata.at(gi)->mfab.at(tl);
          geoms.at(leveldata.level) = patchdata.amrcore->Geom(leveldata.level);
          iters.at(leveldata.level) = cctk_iteration;
          reffacts.at(leveldata.level) = amrex::IntVect{2, 2, 2};
        }

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
        }
      }
    }
  }
}

struct parameters {};
YAML::Emitter &operator<<(YAML::Emitter &yaml, parameters) {
  std::vector<std::pair<std::string, const cParamData *> > parameter_values;
  int first = 1;
  for (;;) {
    const cParamData *data;
    const int iret = CCTK_ParameterWalk(first, nullptr, nullptr, &data);
    assert(iret >= 0);
    if (iret)
      break;
    first = 0;
    if (!CCTK_IsThornActive(data->thorn))
      continue;
    const std::string fullname = std::string(data->thorn) + "::" + data->name;
    parameter_values.emplace_back(fullname, data);
  }

  std::sort(parameter_values.begin(), parameter_values.end());

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
  if (is_root)
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
      const std::string simulation_name = get_simulation_name();
      OutputSilo(cctkGH, group_enabled, out_dir, simulation_name);
#else
      if (strlen(out_silo_vars) != 0)
        CCTK_VERROR("Silo is not enabled. The parameter "
                    "CarpetX::out_silo_vars must be empty.");
#endif
    }
  }

  OutputTSVold(cctkGH);

  {
    const int every = out_tsv_every == -1 ? out_every : out_tsv_every;
    if (every > 0 && cctk_iteration % every == 0)
      OutputTSV(cctkGH);
  }

  OutputMeta(cctkGH);

  if (is_root)
    CCTK_VINFO("OutputGH done.");

  return 0;
}

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

  if (CCTK_EQUALS(checkpoint_method, "openpmd")) {

#ifdef HAVE_CAPABILITY_openPMD_api
    const std::vector<bool> checkpoint_group = [&] {
      std::vector<bool> enabled(CCTK_NumGroups(), false);
      for (int gi = 0; gi < CCTK_NumGroups(); ++gi) {
        const GHExt::GlobalData &globaldata = ghext->globaldata;
        if (globaldata.arraygroupdata.at(gi)) {
          enabled.at(gi) = globaldata.arraygroupdata.at(gi)->do_checkpoint;
        } else {
          const int patch = 0;
          const GHExt::PatchData &patchdata = ghext->patchdata.at(patch);
          const int level = 0;
          const GHExt::PatchData::LevelData &leveldata =
              patchdata.leveldata.at(level);
          assert(leveldata.groupdata.at(gi));
          enabled.at(gi) = leveldata.groupdata.at(gi)->do_checkpoint;
        }
      }
      return enabled;
    }();
    OutputOpenPMD(cctkGH, checkpoint_group, checkpoint_dir, checkpoint_file);
#else
    CCTK_VERROR(
        "CarpetX::checkpoint_method is set to \"openpmd\", but openPMD_api is "
        "not enabled");
#endif

  } else if (CCTK_EQUALS(checkpoint_method, "silo")) {

#ifdef HAVE_CAPABILITY_Silo
    const std::vector<bool> checkpoint_group = [&] {
      std::vector<bool> enabled(CCTK_NumGroups(), false);
      for (int gi = 0; gi < CCTK_NumGroups(); ++gi) {
        const GHExt::GlobalData &globaldata = ghext->globaldata;
        if (globaldata.arraygroupdata.at(gi)) {
          enabled.at(gi) = globaldata.arraygroupdata.at(gi)->do_checkpoint;
        } else {
          const int patch = 0;
          const GHExt::PatchData &patchdata = ghext->patchdata.at(patch);
          const int level = 0;
          const GHExt::PatchData::LevelData &leveldata =
              patchdata.leveldata.at(level);
          assert(leveldata.groupdata.at(gi));
          enabled.at(gi) = leveldata.groupdata.at(gi)->do_checkpoint;
        }
      }
      return enabled;
    }();
    OutputSilo(cctkGH, checkpoint_group, checkpoint_dir, checkpoint_file);
#else
    CCTK_VERROR("CarpetX::checkpoint_method is set to \"silo\", but Silo is "
                "not enabled");
#endif

  } else {
    CCTK_ERROR("unknown value for paramater CarpetX::checkpoint_method");
  }
}

namespace {

int last_walltime_trigger_runtime = 0;

bool CheckWalltimeTrigger(const int runtime) {
  DECLARE_CCTK_PARAMETERS;

  if (!(checkpoint_every_walltime_hours > 0))
    return false;
  const int interval =
      std::max(int(lrint(checkpoint_every_walltime_hours * 3600)), 1);
  if (runtime / interval <= last_walltime_trigger_runtime / interval)
    return false;
  last_walltime_trigger_runtime = runtime;
  return true;
}

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
