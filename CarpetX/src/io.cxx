#include "driver.hxx"
#include "io.hxx"
#include "io_silo.hxx"
#include "reduction.hxx"
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

#include <array>
#include <cassert>
#include <cctype>
#include <fstream>
#include <functional>
#include <memory>
#include <regex>
#include <sstream>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace CarpetX {
using namespace amrex;
using namespace std;

////////////////////////////////////////////////////////////////////////////////

int InputGH(const cGH *restrict cctkGH) {
  DECLARE_CCTK_ARGUMENTS;
  DECLARE_CCTK_PARAMETERS;

  static Timer timer("InputGH");
  Interval interval(timer);

  const bool is_root = CCTK_MyProc(nullptr) == 0;
  if (is_root)
    cout << "InputGH: iteration " << cctk_iteration << ", time " << cctk_time
         << "\n";

#ifdef HAVE_CAPABILITY_Silo
  // TODO: Stop at paramcheck time when Silo input parameters are
  // set, but Silo is not available
  InputSilo(cctkGH);
#endif

  // TODO: This should be the number of variables input
  return 0;
}

////////////////////////////////////////////////////////////////////////////////

void OutputPlotfile(const cGH *restrict cctkGH) {
  DECLARE_CCTK_ARGUMENTS;
  DECLARE_CCTK_PARAMETERS;

  static Timer timer("OutputPlotfile");
  Interval interval(timer);

  const int numgroups = CCTK_NumGroups();
  vector<bool> group_enabled(numgroups, false);
  auto enable_group{[](int index, const char *optstring, void *callback) {
    vector<bool> &group_enabled = *static_cast<vector<bool> *>(callback);
    // CCTK_TraverseString expands the given groups into their variables
    // so I condense it down again to groups only
    group_enabled.at(CCTK_GroupIndexFromVarI(index)) = true;
  }};
  CCTK_TraverseString(out_plotfile_groups, enable_group, &group_enabled,
                      CCTK_GROUP_OR_VAR);

  for (int gi = 0; gi < numgroups; ++gi) {
    if (!group_enabled.at(gi))
      continue;
    if (CCTK_GroupTypeI(gi) != CCTK_GF)
      continue;

    auto &restrict groupdata0 = *ghext->leveldata.at(0).groupdata.at(gi);
    if (groupdata0.mfab.size() > 0) {
      const int tl = 0;

      string groupname = unique_C_ptr<char>(CCTK_GroupName(gi)).get();
      groupname = regex_replace(groupname, regex("::"), "-");
      for (auto &c : groupname)
        c = tolower(c);
      const string filename = [&]() {
        ostringstream buf;
        buf << out_dir << "/" << groupname << ".it" << setw(6) << setfill('0')
            << cctk_iteration;
        return buf.str();
      }();

      Vector<string> varnames(groupdata0.numvars);
      for (int vi = 0; vi < groupdata0.numvars; ++vi) {
        ostringstream buf;
        buf << CCTK_VarName(groupdata0.firstvarindex + vi);
        for (int i = 0; i < tl; ++i)
          buf << "_p";
        varnames.at(vi) = buf.str();
      }

      Vector<const MultiFab *> mfabs(ghext->leveldata.size());
      Vector<Geometry> geoms(ghext->leveldata.size());
      Vector<int> iters(ghext->leveldata.size());
      Vector<IntVect> reffacts(ghext->leveldata.size());
      for (const auto &restrict leveldata : ghext->leveldata) {
        mfabs.at(leveldata.level) = &*leveldata.groupdata.at(gi)->mfab.at(tl);
        geoms.at(leveldata.level) = ghext->amrcore->Geom(leveldata.level);
        iters.at(leveldata.level) = cctk_iteration;
        reffacts.at(leveldata.level) = IntVect{2, 2, 2};
      }

      // TODO: Output all groups into a single file
      WriteMultiLevelPlotfile(filename, mfabs.size(), mfabs, varnames, geoms,
                              cctk_time, iters, reffacts);

      const bool is_root = CCTK_MyProc(nullptr) == 0;
      if (is_root) {
        const string visitname = [&]() {
          ostringstream buf;
          buf << out_dir << "/" << groupname << ".visit";
          return buf.str();
        }();
        ofstream visit(visitname, ios::app);
        assert(visit.good());
        // visit << filename << "/Header\n";
        visit << groupname << ".it" << setw(6) << setfill('0') << cctk_iteration
              << "/Header\n";
        visit.close();
      }
    }
  }
}

////////////////////////////////////////////////////////////////////////////////

void WriteASCII(const cGH *restrict cctkGH, const string &filename, int gi,
                const vector<string> &varnames) {
  ostringstream buf;
  buf << filename << ".tsv";
  ofstream file(buf.str());
  const string sep = "\t";

  // get more precision for floats, could also use
  // https://stackoverflow.com/a/30968371
  file << setprecision(numeric_limits<CCTK_REAL>::digits10 + 1) << scientific;

  // Output header
  file << "# 1:iteration" << sep << "2:time" << sep << "3:level" << sep
       << "4:component" << sep << "5:i" << sep << "6:j" << sep << "7:k" << sep
       << "8:x" << sep << "9:y" << sep << "10:z";
  int col = 11;
  for (const auto &varname : varnames)
    file << sep << col++ << ":" << varname;
  file << "\n";

  for (const auto &leveldata : ghext->leveldata) {
    const auto &groupdata = *leveldata.groupdata.at(gi);
    const int tl = 0;
    const auto &geom = ghext->amrcore->Geom(leveldata.level);
    const auto &mfab = *groupdata.mfab.at(tl);
    for (MFIter mfi(mfab); mfi.isValid(); ++mfi) {
      const Array4<const CCTK_REAL> &vars = mfab.array(mfi);
      const auto &imin = vars.begin;
      const auto &imax = vars.end;
      for (int k = imin.z; k < imax.z; ++k) {
        for (int j = imin.y; j < imax.y; ++j) {
          for (int i = imin.x; i < imax.x; ++i) {
            const array<int, dim> I{i, j, k};
            array<CCTK_REAL, dim> x;
            for (int d = 0; d < dim; ++d)
              x[d] = geom.ProbLo(d) +
                     (I[d] + 0.5 * groupdata.indextype[d]) * geom.CellSize(d);
            file << cctkGH->cctk_iteration << sep << cctkGH->cctk_time << sep
                 << leveldata.level << sep << mfi.index() << sep << I[0] << sep
                 << I[1] << sep << I[2] << sep << x[0] << sep << x[1] << sep
                 << x[2];
            for (int n = 0; n < groupdata.numvars; ++n)
              file << sep << vars(i, j, k, n);
            file << "\n";
          }
        }
      }
    }
  }

  file.close();
}

void OutputASCII(const cGH *restrict cctkGH) {
  DECLARE_CCTK_ARGUMENTS;
  DECLARE_CCTK_PARAMETERS;

  if (!out_tsv)
    return;

  static Timer timer("OutputASCII");
  Interval interval(timer);

  const int numgroups = CCTK_NumGroups();
  for (int gi = 0; gi < numgroups; ++gi) {
    cGroup group;
    int ierr = CCTK_GroupData(gi, &group);
    assert(!ierr);

    if (group.grouptype != CCTK_GF)
      continue;

    auto &restrict groupdata0 = *ghext->leveldata.at(0).groupdata.at(gi);
    if (groupdata0.mfab.size() > 0) {
      const int tl = 0;

      string groupname = unique_C_ptr<char>(CCTK_GroupName(gi)).get();
      groupname = regex_replace(groupname, regex("::"), "-");
      for (auto &c : groupname)
        c = tolower(c);
      ostringstream buf;
      buf << out_dir << "/" << groupname;
      buf << ".it" << setw(6) << setfill('0') << cctk_iteration;
      buf << ".p" << setw(4) << setfill('0') << CCTK_MyProc(nullptr);
      const string filename = buf.str();

      Vector<string> varnames(groupdata0.numvars);
      for (int vi = 0; vi < groupdata0.numvars; ++vi) {
        ostringstream buf;
        buf << CCTK_VarName(groupdata0.firstvarindex + vi);
        for (int i = 0; i < tl; ++i)
          buf << "_p";
        varnames.at(vi) = buf.str();
      }

      WriteASCII(cctkGH, filename, gi, varnames);
    }
  }
}

////////////////////////////////////////////////////////////////////////////////

void OutputNorms(const cGH *restrict cctkGH) {
  DECLARE_CCTK_ARGUMENTS;
  DECLARE_CCTK_PARAMETERS;

  static Timer timer("OutputNorms");
  Interval interval(timer);

  const bool is_root = CCTK_MyProc(nullptr) == 0;
  const string sep = "\t";

  ofstream file;
  if (is_root) {
    ostringstream buf;
    buf << out_dir << "/norms.it" << setw(6) << setfill('0') << cctk_iteration
        << ".tsv";
    const string filename = buf.str();
    file = ofstream(filename);

    // get more precision for floats, could also use
    // https://stackoverflow.com/a/30968371
    file << setprecision(numeric_limits<CCTK_REAL>::digits10 + 1) << scientific;

    int col = 0;
    file << "# " << ++col << ":iteration";
    file << sep << ++col << ":time";
    file << sep << ++col << ":varname";
    file << sep << ++col << ":min";
    file << sep << ++col << ":max";
    file << sep << ++col << ":sum";
    file << sep << ++col << ":avg";
    file << sep << ++col << ":stddev";
    file << sep << ++col << ":volume";
    file << sep << ++col << ":L1norm";
    file << sep << ++col << ":L2norm";
    file << sep << ++col << ":maxabs";
    for (int d = 0; d < dim; ++d)
      file << sep << ++col << ":minloc[" << d << "]";
    for (int d = 0; d < dim; ++d)
      file << sep << ++col << ":maxloc[" << d << "]";
    file << "\n";
  }

  const int numgroups = CCTK_NumGroups();
  for (int gi = 0; gi < numgroups; ++gi) {
    if (CCTK_GroupTypeI(gi) != CCTK_GF)
      continue;

    const int level = 0;
    const GHExt::LevelData &restrict leveldata = ghext->leveldata.at(level);
    const GHExt::LevelData::GroupData &restrict groupdata =
        *leveldata.groupdata.at(gi);

    const int tl = 0;
    for (int vi = 0; vi < groupdata.numvars; ++vi) {

      const reduction<CCTK_REAL, dim> red = reduce(gi, vi, tl);

      if (is_root) {
        file << cctk_iteration << sep << cctk_time << sep
             << CCTK_FullVarName(groupdata.firstvarindex + vi) << sep << red.min
             << sep << red.max << sep << red.sum << sep << red.avg() << sep
             << red.sdv() << sep << red.norm0() << sep << red.norm1() << sep
             << red.norm2() << sep << red.norm_inf();
        for (int d = 0; d < dim; ++d)
          file << sep << red.minloc[d];
        for (int d = 0; d < dim; ++d)
          file << sep << red.maxloc[d];
        file << "\n";
      }
    }
  }

  if (is_root)
    file.close();
}

////////////////////////////////////////////////////////////////////////////////

void OutputScalars(const cGH *restrict cctkGH) {
  DECLARE_CCTK_ARGUMENTS;
  DECLARE_CCTK_PARAMETERS;

  static Timer timer("OutputNorms");
  Interval interval(timer);

  const bool is_root = CCTK_MyProc(nullptr) == 0;
  if (!is_root)
    return;
  const string sep = "\t";

  const int numgroups = CCTK_NumGroups();
  for (int gi = 0; gi < numgroups; ++gi) {
    cGroup group;
    int ierr = CCTK_GroupData(gi, &group);
    assert(!ierr);

    if (group.grouptype != CCTK_SCALAR)
      continue;

    string groupname = unique_C_ptr<char>(CCTK_GroupName(gi)).get();
    groupname = regex_replace(groupname, regex("::"), "-");
    for (auto &c : groupname)
      c = tolower(c);
    ostringstream buf;
    buf << out_dir << "/" << groupname << ".tsv";
    const string filename = buf.str();

    ofstream file(filename.c_str(), std::ofstream::out | std::ofstream::app);

    // TODO: write proper header (once)
#if 0
    // Output header
    file << "#"
         << sep
         << "iteration"
         << sep
         << "time";
    for (int vi = 0; vi < groupdata.numvars; ++vi) {
      file << sep << CCTK_VarName(group.firstvarindex + vi);
    file << "\n";
#endif

    // Output data
    const GHExt::GlobalData &restrict globaldata = ghext->globaldata;
    const GHExt::GlobalData::ScalarGroupData &restrict scalargroupdata =
        *globaldata.scalargroupdata.at(gi);
    const int tl = 0;
    file << cctkGH->cctk_iteration << sep << cctkGH->cctk_time;
    for (int vi = 0; vi < scalargroupdata.numvars; ++vi) {
      file << sep << scalargroupdata.data.at(tl).at(vi);
    }
    file << "\n";
  }
}

////////////////////////////////////////////////////////////////////////////////

int OutputGH(const cGH *restrict cctkGH) {
  DECLARE_CCTK_ARGUMENTS;
  DECLARE_CCTK_PARAMETERS;

  static Timer timer("OutputGH");
  Interval interval(timer);

  const bool is_root = CCTK_MyProc(nullptr) == 0;
  if (is_root)
    cout << "OutputGH: iteration " << cctk_iteration << ", time " << cctk_time
         << ", run time " << CCTK_RunTime() << " s\n";

  if (out_every > 0 && cctk_iteration % out_every == 0) {
    OutputPlotfile(cctkGH);
#ifdef HAVE_CAPABILITY_Silo
    // TODO: Stop at paramcheck time when Silo output parameters are
    // set, but Silo is not available
    OutputSilo(cctkGH);
#endif
    OutputASCII(cctkGH);
    OutputNorms(cctkGH);
    OutputScalars(cctkGH);
  }

  // TODO: This should be the number of variables output
  return 0;
}

////////////////////////////////////////////////////////////////////////////////

#if 0

// Input

int Input(cGH *const cctkGH, const char *const basefilename,
          const int called_from) {
  DECLARE_CCTK_PARAMETERS;

  assert(called_from == CP_RECOVER_PARAMETERS ||
         called_from == CP_RECOVER_DATA || called_from == FILEREADER_DATA);
  const bool do_recover =
      called_from == CP_RECOVER_PARAMETERS || called_from == CP_RECOVER_DATA;
  const bool read_parameters = called_from == CP_RECOVER_PARAMETERS;

  static Timer timer("InputGH");
  Interval interval(timer);

  const string projectname = [&]() -> string {
    if (do_recover) {
      // basename is only passed for CP_RECOVER_PARAMETERS, and needs to
      // be remembered
      static string saved_basefilename;
      if (read_parameters)
        saved_basefilename = basefilename;
      assert(!saved_basefilename.empty());
      return saved_basefilename;
    } else {
      return basefilename;
    }
  }();

  if (do_recover)
    if (read_parameters)
      CCTK_VINFO("Recovering parameters  from file \"%s\"",
                 projectname.c_str());
    else
      CCTK_VINFO("Recovering variables  from file \"%s\"", projectname.c_str());
  else
    CCTK_VINFO("Reading variables from file \"%s\"", projectname.c_str());

  if (do_recover && !read_parameters) {
    // Set global Cactus variables
    // CCTK_SetMainLoopIndex(main_loop_index);
#warning "TODO"
    // cctkGH->cctk_iteration = iteration;
    cctkGH->cctk_iteration = 0;
  }

  // Determine which variables to read
  const auto ioUtilGH =
      static_cast<const ioGH *>(CCTK_GHExtension(cctkGH, "IO"));
  const vector<bool> groups_enabled = [&] {
    vector<bool> enabled(CCTK_NumGroups(), false);
    if (do_recover) {
      for (int gi = 0; gi < CCTK_NumGroups(); ++gi) {
        if (!CCTK_QueryGroupStorageI(cctkGH, gi))
          continue;
        cGroup gdata;
        int ierr = CCTK_GroupData(gi, &gdata);
        assert(!ierr);
        // Do not recover groups with a "checkpoint=no" tag
        const int len =
            Util_TableGetString(gdata.tagstable, 0, nullptr, "checkpoint");
        if (len > 0) {
          array<char, 10> buf;
          Util_TableGetString(gdata.tagstable, buf.size(), buf.data(),
                              "checkpoint");
          if (CCTK_EQUALS(buf.data(), "no"))
            continue;
          assert(CCTK_EQUALS(buf.data(), "yes"));
        }
        enabled.at(gi) = true;
      }
    } else {
      for (int gi = 0; gi < CCTK_NumGroups(); ++gi) {
        const int v0 = CCTK_FirstVarIndexI(gi);
        assert(v0 >= 0);
        const int nv = CCTK_NumVarsInGroupI(gi);
        assert(nv >= 0);
        for (int vi = 0; vi < nv; ++vi)
          if (!ioUtilGH->do_inVars || ioUtilGH->do_inVars[v0 + vi])
            enabled.at(gi) = true;
      }
    }
    return enabled;
  }();

#warning "CONTINUE HERE"
#if 0
  io_dir_t io_dir = do_recover ? io_dir_t::recover : io_dir_t::input;
  bool did_read_parameters = false;
  bool did_read_grid_structure = false;
  if (not input_file_hdf5_ptrs.count(projectname)) {
    static HighResTimer::HighResTimer timer1("SimulationIO::read_file_hdf5");
    auto timer1_clock = timer1.start();
    input_file_hdf5_ptrs[projectname] = make_unique<input_file_t>(
        io_dir, projectname, file_format::hdf5, iteration, -1, -1);
    timer1_clock.stop(0);
  }
  const auto &input_file_ptr = input_file_hdf5_ptrs.at(projectname);
  if (read_parameters) {
    static HighResTimer::HighResTimer timer1(
        "SimulationIO::read_parameters_hdf5");
    auto timer1_clock = timer1.start();
    input_file_ptr->read_params();
    did_read_parameters = true;
    timer1_clock.stop(0);
  } else {
    if (not did_read_grid_structure) {
      static HighResTimer::HighResTimer timer1(
          "SimulationIO::read_grid_structure_hdf5");
      auto timer1_clock = timer1.start();
      input_file_ptr->read_grid_structure(cctkGH);
      did_read_grid_structure = true;
      timer1_clock.stop(0);
    }
    static HighResTimer::HighResTimer timer1(
        "SimulationIO::read_variables_hdf5");
    auto timer1_clock = timer1.start();
    input_file_ptr->read_vars(input_vars, -1, -1);
    timer1_clock.stop(0);
  }

  if (read_parameters)
    return did_read_parameters ? 1 : 0;

  assert(did_read_grid_structure);
#endif
  return 0; // no error
}

////////////////////////////////////////////////////////////////////////////////

// Recovering

extern "C" int CarpetX_RecoverParameters() {
  DECLARE_CCTK_PARAMETERS;
  const char *const out_extension = ".silo";
  const int iret = IOUtil_RecoverParameters(Input, out_extension, "CarpetX");
  return iret;
}

#endif

} // namespace CarpetX
