#include "io/io_norm.hxx"

#include "hierarchy/driver.hxx"
#include "io/io_meta.hxx"
#include "io/selection.hxx"
#include <loop.hxx>
#include "ops/reduction.hxx"
#include "util/refuse.hxx"
#include <carpetx.hxx>

#include <cctk.h>
#include <cctk_Arguments.h>
#include <cctk_Parameters.h>

#include <cassert>
#include <fstream>
#include <iomanip>
#include <limits>
#include <mutex>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

namespace CarpetX {

void OutputNorms(const cGH *restrict cctkGH) {
  const operation_scope_t op("CarpetX::OutputNorms");
  DECLARE_CCTK_ARGUMENTS;
  DECLARE_CCTK_PARAMETERS;

  const group_selection_t selection =
      group_selection_t::select_groups("out_norm_vars", out_norm_vars);
  if (selection.empty())
    return;
  selection.log("Norm", group_kind::grid_function);

  static Timer timer("OutputNorms");
  Interval interval(timer);

  static group_selection_t previous_selection;
  const bool selection_changed = selection != previous_selection;
  previous_selection = selection;

  const bool is_root = CCTK_MyProc(nullptr) == 0;
  const std::string sep = "\t";
  const tl_t tl{0};
  std::vector<std::string> reductions = {"min",    "max",    "sum",
                                         "avg",    "stddev", "volume",
                                         "L1norm", "L2norm", "maxabs"};
  if (!out_norm_omit_unstable) {
    for (int d = 0; d < dim; ++d) {
      std::ostringstream buf;
      buf << "minloc[" << d << "]";
      reductions.push_back(buf.str());
    }
    for (int d = 0; d < dim; ++d) {
      std::ostringstream buf;
      buf << "maxloc[" << d << "]";
      reductions.push_back(buf.str());
    }
  }
  if (!out_norm_omit_sumloc_for_backward_compatibility) {
    for (int d = 0; d < dim; ++d) {
      std::ostringstream buf;
      buf << "sumloc[" << d << "]";
      reductions.push_back(buf.str());
    }
  }

  if (is_root) {
    static std::once_flag create_directory;
    std::call_once(create_directory, [&]() {
      const int mode = 0755;
      int ierr = CCTK_CreateDirectory(mode, out_dir);
      assert(ierr >= 0);
      std::ostringstream buf;
      buf << out_dir << "/norms";
      ierr = CCTK_CreateDirectory(mode, buf.str().c_str());
      assert(ierr >= 0);
    });
  }

  const GHExt &ext = driver();
  if (is_root) {
    if (selection_changed) {
      for (const GHExt::GroupRecord &group :
           selection.records(group_kind::grid_function)) {
        const patch_t patch{0};
        const GHExt::PatchData &restrict patchdata = ext.patch(patch);
        const level_t level{patch, 0};
        const GHExt::PatchData::LevelData &restrict leveldata =
            patchdata.level(level);
        const GHExt::PatchData::LevelData::GroupData &restrict groupdata =
            leveldata.group(group.groupindex);

        bool group_has_valid_variables = false;
        for (const vi_t vi : group.variables()) {
          if (!groupdata.timelevel(tl).valid(vi).valid_int)
            continue;
          group_has_valid_variables = true;
        }
        if (!group_has_valid_variables)
          continue;

        std::ostringstream buf;
        std::string groupname = group.groupname;
        groupname = std::regex_replace(groupname, std::regex("::"), "-");
        for (auto &ch : groupname)
          ch = std::tolower(ch);
        buf << out_dir << "/norms/" << groupname << ".tsv";
        const std::string filename = buf.str();
        std::ofstream file;
        file.open(filename, std::ios_base::app);

        int col = 0;
        file << "# " << ++col << ":iteration";
        file << sep << ++col << ":time";

        for (const vi_t vi : group.variables()) {
          if (!groupdata.timelevel(tl).valid(vi).valid_int)
            continue;

          std::string varname = CCTK_FullVarName(group.var(vi).index());
          for (auto &ch : varname)
            ch = std::tolower(ch);
          for (const auto &reduction : reductions)
            file << sep << ++col << ":" << varname << "." << reduction;
        }
        file << "\n";

        file.close();
      }
    }
  }

  for (const GHExt::GroupRecord &group :
       selection.records(group_kind::grid_function)) {
    const patch_t patch{0};
    const GHExt::PatchData &restrict patchdata = ext.patch(patch);
    const level_t level{patch, 0};
    const GHExt::PatchData::LevelData &restrict leveldata =
        patchdata.level(level);
    const GHExt::PatchData::LevelData::GroupData &restrict groupdata =
        leveldata.group(group.groupindex);

    bool group_has_valid_variables = false;
    for (const vi_t vi : group.variables()) {
      if (!groupdata.timelevel(tl).valid(vi).valid_int)
        continue;
      group_has_valid_variables = true;
    }
    if (!group_has_valid_variables)
      continue;

    std::ofstream file;
    output_file_description_t ofd;
    if (is_root) {
      std::ostringstream buf;
      std::string groupname = group.groupname;
      groupname = std::regex_replace(groupname, std::regex("::"), "-");
      for (auto &ch : groupname)
        ch = std::tolower(ch);
      buf << out_dir << "/norms/" << groupname << ".tsv";
      const std::string filename = buf.str();
      file.open(filename, std::ios_base::app);
      ofd.filename = filename;

      // get more precision for floats, could also use
      // https://stackoverflow.com/a/30968371
      file << std::setprecision(std::numeric_limits<CCTK_REAL>::digits10 + 1)
           << std::scientific;

      file << cctk_iteration << sep << cctk_time;
    }

    for (const vi_t vi : groupdata.record.variables()) {
      if (!groupdata.timelevel(tl).valid(vi).valid_int)
        continue;

      ofd.variables.push_back(
          CCTK_FullVarName(groupdata.record.var(vi).index()));

      const reduction<CCTK_REAL, dim> red = reduce(group.groupindex, vi, tl);

      if (is_root) {
        file << sep << red.min << sep << red.max << sep << red.sum << sep
             << red.avg() << sep << red.sdv() << sep << red.norm0() << sep
             << red.norm1() << sep << red.norm2() << sep << red.norm_inf();
        if (!out_norm_omit_unstable) {
          for (int d = 0; d < dim; ++d)
            file << sep << red.minloc[d];
          for (int d = 0; d < dim; ++d)
            file << sep << red.maxloc[d];
        }
        if (!out_norm_omit_sumloc_for_backward_compatibility) {
          for (int d = 0; d < dim; ++d)
            file << sep << red.sumloc[d];
        }
      }
    }

    if (is_root) {
      file << "\n";
      file.close();

      ofd.description = "CarpetX TSV norms output";
      ofd.writer_thorn = CCTK_THORNSTRING;
      ofd.iterations = {cctk_iteration};
      ofd.reductions = {
          reduction_t::minimum,
          reduction_t::maximum,
          reduction_t::sum,
          reduction_t::average,
          reduction_t::standard_deviation,
          reduction_t::volume,
          reduction_t::norm1,
          reduction_t::norm2,
          reduction_t::norm_inf,
      };
      if (!out_norm_omit_unstable) {
        ofd.reductions.push_back(reduction_t::minimum_location);
        ofd.reductions.push_back(reduction_t::maximum_location);
      }
      if (!out_norm_omit_sumloc_for_backward_compatibility)
        ofd.reductions.push_back(reduction_t::sum_location);
      ofd.format_name = "CarpetX/norms/TSV";
      ofd.format_version = {1, 1, 0};

      OutputMeta_RegisterOutputFile(std::move(ofd));
    }
  }
}

} // namespace CarpetX
