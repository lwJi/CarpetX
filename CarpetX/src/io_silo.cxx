#include "io_silo.hxx"

#include "driver.hxx"
#include "io_meta.hxx"
#include "io_silo_names.hxx"
#include "mpi_types.hxx"
#include "schedule.hxx"
#include <carpetx.hxx>

#include <tuple.hxx>

#include <CactusBase/IOUtil/src/ioutil_CheckpointRecovery.h>
#include <cctk.h>
#include <cctk_Arguments.h>
#include <cctk_Parameters.h>

#ifdef HAVE_CAPABILITY_Silo

#include <AMReX.H>
#include <AMReX_IntVect.H>

#include <mpi.h>

#include <silo.hxx>

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace CarpetX {

namespace {

constexpr bool io_verbose = true;

// Slice membership + slab bounds for component `c`, shared by the data-write
// and metadata passes so they skip the same components in the same order.
// Returns the thickness-1 slab (or nullopt if `c` misses the plane), tested
// against the interior box so a box-boundary plane belongs to one component.
inline std::optional<std::pair<Arith::vect<int, 3>, Arith::vect<int, 3> > >
slice_slab(const amrex::MultiFab &mfab, const int c,
           const amrex::Real *const x0, const amrex::Real *const dx,
           const amrex::IndexType &indextype, const slice_t &slice) {
  const amrex::Box &fabbox = mfab.fabbox(c); // exterior
  const Arith::vect<int, 3> ext_lo{fabbox.smallEnd(0), fabbox.smallEnd(1),
                                   fabbox.smallEnd(2)};
  const Arith::vect<int, 3> ext_hi{fabbox.bigEnd(0) + 1, fabbox.bigEnd(1) + 1,
                                   fabbox.bigEnd(2) + 1};
  const amrex::Box &validbox = mfab.box(c); // interior (variable-typed)
  // enclosedCells maps a nodal or cell valid box to the same cell range, so
  // membership (which box owns the plane) is centering-independent.
  const amrex::Box cvalidbox = amrex::enclosedCells(validbox);
  const Arith::vect<int, 3> cval_lo{
      cvalidbox.smallEnd(0), cvalidbox.smallEnd(1), cvalidbox.smallEnd(2)};
  const Arith::vect<int, 3> cval_hi{cvalidbox.bigEnd(0) + 1,
                                    cvalidbox.bigEnd(1) + 1,
                                    cvalidbox.bigEnd(2) + 1};
  const Arith::vect<CCTK_REAL, 3> sx0{x0[0], x0[1], x0[2]};
  const Arith::vect<CCTK_REAL, 3> sdx{dx[0], dx[1], dx[2]};
  const Arith::vect<bool, 3> is_cell_centred{indextype.cellCentered(0),
                                             indextype.cellCentered(1),
                                             indextype.cellCentered(2)};
  return slice.restrict_component(ext_lo, ext_hi, cval_lo, cval_hi, sx0, sdx,
                                  is_cell_centred);
}

struct mesh_props_t {
  std::array<int, dim> nghosts;

  friend bool operator==(const mesh_props_t &p, const mesh_props_t &q) {
    return p.nghosts == q.nghosts;
  }
  friend bool operator<(const mesh_props_t &p, const mesh_props_t &q) {
    return p.nghosts < q.nghosts;
  }
};

std::string var_name(const gi_t gi, const int vi) {
  return CCTK_FullVarName(ghext->record(gi).var(vi_t{vi}).index());
}

void RegisterSiloOutput(const cGH *restrict const cctkGH,
                        const std::vector<bool> &output_group,
                        const std::string &output_dir,
                        const std::string &output_file, const slice_t &slice) {
  std::vector<std::string> variables;
  for (int gi = 0; gi < CCTK_NumGroups(); ++gi) {
    if (!output_group.at(gi))
      continue;
    if (CCTK_GroupTypeI(gi) != CCTK_GF)
      continue;
    const GHExt::GroupRecord &record = ghext->record(gi_t{gi});
    for (const vi_t vi : record.variables())
      variables.push_back(CCTK_FullVarName(record.var(vi).index()));
  }
  std::vector<int> output_directions;
  for (const int d : slice.inplane_dirs())
    output_directions.push_back(d);
  OutputMeta_RegisterOutputFile(silo_output_description(
      output_dir + "/" + make_filename(output_file, cctkGH->cctk_iteration),
      "2D CarpetX HDF5 Silo slice output", std::move(variables),
      cctkGH->cctk_iteration, std::move(output_directions)));
}

} // namespace

////////////////////////////////////////////////////////////////////////////////

void OutputSilo(const cGH *restrict const cctkGH,
                const std::vector<bool> &output_group,
                const std::string &output_dir, const std::string &output_file,
                const slice_t &slice) {
  DECLARE_CCTK_ARGUMENTS;
  DECLARE_CCTK_PARAMETERS;

  int ierr;

  // Set up timers
  static Timer timer("OutputSilo");
  Interval interval(timer);

  if (std::count(output_group.begin(), output_group.end(), true) == 0)
    return;

  if (io_verbose)
    CCTK_VINFO("OutputSilo...");

  static Timer timer_setup("OutputSilo.setup");
  auto interval_setup = std::make_unique<Interval>(timer_setup);

  const MPI_Comm mpi_comm = MPI_COMM_WORLD;
  const int myproc = CCTK_MyProc(cctkGH);
  const int nprocs = CCTK_nProcs(cctkGH);

  const int ioproc_every = [&]() {
    if (CCTK_EQUALS(out_mode, "proc"))
      return 1;
    if (CCTK_EQUALS(out_mode, "np"))
      return out_proc_every;
    if (CCTK_EQUALS(out_mode, "onefile"))
      return nprocs;
    assert(0);
  }();
  assert(ioproc_every > 0);

  const int myioproc = myproc / ioproc_every * ioproc_every;
  assert(myioproc <= myproc);
  assert(myproc < myioproc + ioproc_every);
  const bool write_file = myproc % ioproc_every == 0;
  assert((myioproc == myproc) == write_file);
  // If process 1 exists and if it is not an I/O process, then output
  // metadata there. Else output metadata on process 0.
  const int metafile_ioproc = nprocs == 1 || ioproc_every == 1 ? 0 : 1;
  const bool write_metafile = myproc == metafile_ioproc;

  // Configure Silo library
  DBShowErrors(DB_ALL_AND_DRVR, nullptr);
  DBSetCompression(out_silo_compression_options);
  DBSetEnableChecksums(1);

  // TODO: directories instead of carefully chosen names

  constexpr int ndims = dim;

  for (const auto &patchdata : ghext->patches())
    if (!patchdata.is_cartesian)
      CCTK_VERROR("2D Silo slice output is only supported for "
                  "Cartesian patches");

  interval_setup = nullptr;

  static Timer timer_data("OutputSilo.data");
  auto interval_data = std::make_unique<Interval>(timer_data);

  {
    static Timer timer_mkdir("OutputSilo.mkdir");
    auto interval_mkdir = std::make_unique<Interval>(timer_mkdir);
    const std::string subdirname = make_subdirname(output_file, cctk_iteration);
    const std::string pathname = std::string(output_dir) + "/" + subdirname;
    const int mode = 0755;
    static std::once_flag create_directory;
    call_once(create_directory, [&]() {
      const int ierr = CCTK_CreateDirectory(mode, output_dir.c_str());
      assert(ierr >= 0);
    });
    ierr = CCTK_CreateDirectory(mode, pathname.c_str());
    assert(ierr >= 0);
    interval_mkdir = nullptr;

    DB::ptr<DBfile> file;
    if (write_file) {
      const std::string filename =
          pathname + "/" +
          make_filename(output_file, cctk_iteration, myproc / ioproc_every);
      file = DB::make(DBCreate(filename.c_str(), DB_CLOBBER, DB_LOCAL,
                               output_file.c_str(), DB_HDF5));
      assert(file);
    }

    if (write_file) {
      {
        char *const data = IOUtil_GetAllParameters(cctkGH, 1 /*all*/);
        const std::string parameters(data);
        std::free(data);
        const int dims = parameters.length();
        ierr = DBWrite(file.get(), "AllParameters", parameters.data(), &dims, 1,
                       DB_CHAR);
        assert(!ierr);
      }

      { // Tell VisIt that the mesh structure may change over time
        const int dims = 1;
        const int value = 1;
        ierr = DBWrite(file.get(), "MetadataIsTimeVarying", &value, &dims, 1,
                       DB_INT);
        assert(!ierr);
      }
    }

    for (const auto &patchdata : ghext->patches()) {

      for (const auto &leveldata : patchdata.levels()) {

        const amrex::DistributionMapping &dm = leveldata.fab->DistributionMap();
        const int ncomponents = dm.size();

        std::set<mesh_props_t> have_meshes;
        for (const auto &groupdata : leveldata.groups()) {
          const gi_t gi = groupdata.record.groupindex;
          if (!output_group.at(gi.index()))
            continue;

          const int numvars = groupdata.record.numvars;
          const tl_t tl{0};
          const amrex::MultiFab &mfab = *groupdata.timelevel(tl).mfab;
          const amrex::IndexType &indextype = mfab.ixType();
          const amrex::DistributionMapping &dm = mfab.DistributionMap();

          const std::array<int, dim> &nghosts = groupdata.nghostzones;
          const mesh_props_t mesh_props{nghosts};
          const bool have_mesh = have_meshes.count(mesh_props);

          for (int component = 0; component < ncomponents; ++component) {
            const int proc = dm[component];
            const int ioproc = proc / ioproc_every * ioproc_every;
            const bool send_this_fab = proc == myproc;
            const bool write_this_fab = ioproc == myproc;
            if (!(send_this_fab || write_this_fab))
              continue;

            // TODO: Check whether data are valid
            const amrex::Box &fabbox = mfab.fabbox(component); // exterior

            // Exterior (source) half-open bounds. The variable `data` array is
            // always shaped by this box, whether the local fab pointer or an
            // MPI receive buffer.
            const Arith::vect<int, 3> ext_lo{
                fabbox.smallEnd(0), fabbox.smallEnd(1), fabbox.smallEnd(2)};
            const Arith::vect<int, 3> ext_hi{fabbox.bigEnd(0) + 1,
                                             fabbox.bigEnd(1) + 1,
                                             fabbox.bigEnd(2) + 1};

            // Number of zones in the full exterior box. Used for MPI send/recv
            // sizing and for indexing into the (un-sliced) `data` array.
            std::ptrdiff_t src_zonecount = 1;
            for (int d = 0; d < ndims; ++d)
              src_zonecount *= ext_hi[d] - ext_lo[d];
            assert(src_zonecount >= 0 && src_zonecount <= INT_MAX);

            const amrex::Geometry &geom =
                patchdata.amrcore->Geom(leveldata.level.index());
            const amrex::Real *const x0 = geom.ProbLo();
            const amrex::Real *const dx = geom.CellSize();

            const int slice_n = slice.normal_dir;
            const auto slab_box =
                slice_slab(mfab, component, x0, dx, indextype, slice);
            if (!slab_box)
              continue;
            const Arith::vect<int, 3> sub_lo = slab_box->first;
            const Arith::vect<int, 3> sub_hi = slab_box->second;

            std::array<int, ndims> dims;
            for (int d = 0; d < ndims; ++d)
              dims[d] = sub_hi[d] - sub_lo[d];
            std::ptrdiff_t zonecount = 1;
            for (int d = 0; d < ndims; ++d)
              zonecount *= dims[d];
            assert(zonecount >= 0 && zonecount <= INT_MAX);

            if (write_file && !have_mesh) {
              static Timer timer_mesh("OutputSilo.mesh");
              Interval interval_mesh(timer_mesh);

              const std::string meshname =
                  make_meshname(nghosts, leveldata.level, component);

              std::array<int, ndims> dims_vc;
              for (int d = 0; d < ndims; ++d)
                dims_vc[d] = dims[d] + int(indextype.cellCentered(d));

              std::array<std::vector<CCTK_REAL>, ndims> coords;
              std::array<const void *, ndims> coord_ptrs;
              for (int d = 0; d < ndims; ++d) {
                coords[d].resize(dims_vc[d]);
                // sub_lo[d] == fabbox.smallEnd(d) for in-plane axes; for the
                // sliced normal axis it is the plane index, so the slab's
                // single coordinate plane is placed correctly.
                for (int i = 0; i < dims_vc[d]; ++i)
                  coords[d][i] = x0[d] + (sub_lo[d] + i) * dx[d];
              }
              for (int d = 0; d < ndims; ++d)
                coord_ptrs[d] = coords[d].data();

              const DB::ptr<DBoptlist> optlist = DB::make(DBMakeOptlist(10));
              assert(optlist);

              int cartesian = DB_CARTESIAN;
              ierr = DBAddOption(optlist.get(), DBOPT_COORDSYS, &cartesian);
              assert(!ierr);

              int cycle = cctk_iteration;
              ierr = DBAddOption(optlist.get(), DBOPT_CYCLE, &cycle);
              assert(!ierr);

              std::array<int, ndims> min_index, max_index;
              for (int d = 0; d < ndims; ++d) {
                const int ng = d == slice_n ? 0 : nghosts[d];
                min_index[d] = ng;
                max_index[d] = ng;
              }
              ierr =
                  DBAddOption(optlist.get(), DBOPT_LO_OFFSET, min_index.data());
              assert(!ierr);
              ierr =
                  DBAddOption(optlist.get(), DBOPT_HI_OFFSET, max_index.data());
              assert(!ierr);

              int column_major = 0;
              ierr =
                  DBAddOption(optlist.get(), DBOPT_MAJORORDER, &column_major);
              assert(!ierr);

              double dtime = cctk_time;
              ierr = DBAddOption(optlist.get(), DBOPT_DTIME, &dtime);
              assert(!ierr);

              int hide_from_gui = 1;
              ierr = DBAddOption(optlist.get(), DBOPT_HIDE_FROM_GUI,
                                 &hide_from_gui);
              assert(!ierr);

              ierr = DBPutQuadmesh(file.get(), meshname.c_str(), nullptr,
                                   coord_ptrs.data(), dims_vc.data(), ndims,
                                   db_datatype_v<CCTK_REAL>, DB_COLLINEAR,
                                   optlist.get());
              assert(!ierr);

              have_meshes.insert(mesh_props);
            } // if write mesh

            // Communicate variable
            static Timer timer_mpi("OutputSilo.mpi");
            auto interval_mpi = std::make_unique<Interval>(timer_mpi);
            const int mpi_tag = 22900; // randomly chosen
            std::vector<CCTK_REAL> buffer;
            const CCTK_REAL *data = nullptr;
            if (send_this_fab && write_this_fab) {
              const amrex::FArrayBox &fab = mfab[component];
              data = fab.dataPtr();
            } else if (send_this_fab) {
              const amrex::FArrayBox &fab = mfab[component];
              assert(numvars * src_zonecount <= INT_MAX);
              MPI_Send(fab.dataPtr(), numvars * src_zonecount,
                       mpi_datatype_v<CCTK_REAL>, ioproc, mpi_tag, mpi_comm);
            } else {
              buffer.resize(numvars * src_zonecount);
              assert(numvars * src_zonecount <= INT_MAX);
              MPI_Recv(buffer.data(), numvars * src_zonecount,
                       mpi_datatype_v<CCTK_REAL>, proc, mpi_tag, mpi_comm,
                       MPI_STATUS_IGNORE);
              data = buffer.data();
            }
            interval_mpi = nullptr;

            if (write_file) {
              static Timer timer_var("OutputSilo.var");
              Interval interval_var(timer_var);

              const std::string meshname =
                  make_meshname(nghosts, leveldata.level, component);

              const int centering = [&]() {
                const int rank = indextype.cellCentered(0) +
                                 indextype.cellCentered(1) +
                                 indextype.cellCentered(2);
                switch (rank) {
                case 0:
                  return DB_NODECENT;
                case 1:
                  return DB_EDGECENT;
                case 2:
                  return DB_FACECENT;
                case 3:
                  return DB_ZONECENT;
                }
                assert(0);
              }();

              const DB::ptr<DBoptlist> optlist = DB::make(DBMakeOptlist(10));
              assert(optlist);

              int cartesian = DB_CARTESIAN;
              ierr = DBAddOption(optlist.get(), DBOPT_COORDSYS, &cartesian);
              assert(!ierr);

              int cycle = cctk_iteration;
              ierr = DBAddOption(optlist.get(), DBOPT_CYCLE, &cycle);
              assert(!ierr);

              int column_major = 0;
              ierr =
                  DBAddOption(optlist.get(), DBOPT_MAJORORDER, &column_major);
              assert(!ierr);

              double dtime = cctk_time;
              ierr = DBAddOption(optlist.get(), DBOPT_DTIME, &dtime);
              assert(!ierr);

              int hide_from_gui = 1;
              ierr = DBAddOption(optlist.get(), DBOPT_HIDE_FROM_GUI,
                                 &hide_from_gui);
              assert(!ierr);

              if (centering == DB_EDGECENT || centering == DB_FACECENT) {
                // Need to find the other 2 edge- or face-centered
                // variables, and output them as well. Maybe output all
                // 3 when the x- or xy-centered value is output? Maybe
                // add a "sibling" tag to such grid functions to find
                // these other components?
                assert(0);
              }

              std::vector<CCTK_REAL> slab(zonecount);
              for (int vi = 0; vi < numvars; ++vi) {
                const std::string varname =
                    make_varname(var_name(gi, vi), leveldata.level, component);

                extract_subbox(slab.data(), data + vi * src_zonecount, ext_lo,
                               ext_hi, sub_lo, sub_hi);

                ierr = DBPutQuadvar1(
                    file.get(), varname.c_str(), meshname.c_str(), slab.data(),
                    dims.data(), ndims, nullptr, 0, db_datatype_v<CCTK_REAL>,
                    centering, optlist.get());
              } // for vi
            } // if write_file

          } // for component
        } // for gi
      } // for leveldata
    } // for patchdata
  } // write data

  interval_data = nullptr;

  static Timer timer_meta("OutputSilo.meta");
  auto interval_meta = std::make_unique<Interval>(timer_meta);

  if (write_metafile) {

    const std::string metafilename = std::string(output_dir) + "/" +
                                     make_filename(output_file, cctk_iteration);
    const DB::ptr<DBfile> metafile =
        DB::make(DBCreate(metafilename.c_str(), DB_CLOBBER, DB_LOCAL,
                          output_file.c_str(), DB_HDF5));
    assert(metafile);

    {
      char *const data = IOUtil_GetAllParameters(cctkGH, 1 /*all*/);
      const std::string parameters(data);
      std::free(data);
      const int dims = parameters.length();
      ierr = DBWrite(metafile.get(), "AllParameters", parameters.data(), &dims,
                     1, DB_CHAR);
      assert(!ierr);
    }

    {
      // Tell VisIt that the mesh structure may change over time
      const int dims = 1;
      const int value = 1;
      ierr = DBWrite(metafile.get(), "MetadataIsTimeVarying", &value, &dims, 1,
                     DB_INT);
      assert(!ierr);
    }

    std::map<mesh_props_t, std::size_t> mesh_nblocks;
    for (int gi0 = 0; gi0 < CCTK_NumGroups(); ++gi0) {
      const gi_t gi{gi0};
      if (!output_group.at(gi.index()))
        continue;
      if (CCTK_GroupTypeI(gi.index()) != CCTK_GF)
        continue;

      const auto &groupdata0 = ghext->level(level_t{patch_t{0}, 0}).group(gi);
      const int numvars = groupdata0.record.numvars;
      const tl_t tl{0};
      const amrex::MultiFab &mfab0 = *groupdata0.timelevel(tl).mfab;
      const amrex::IndexType &indextype = mfab0.ixType();

      const std::array<int, dim> &nghosts = groupdata0.nghostzones;
      const mesh_props_t mesh_props{nghosts};
      const bool have_mesh = mesh_nblocks.count(mesh_props);

      if (!have_mesh) {
        const std::string multimeshname = make_meshname(nghosts);

        typedef std::array<std::array<CCTK_REAL, ndims>, 2> extent_t;
        std::vector<extent_t> extents;

        for (const auto &patchdata : ghext->patches()) {
          for (const auto &leveldata : patchdata.levels()) {
            const auto &groupdata = leveldata.group(gi);
            const tl_t tl{0};
            const amrex::MultiFab &mfab = *groupdata.timelevel(tl).mfab;
            const int ncomponents = mfab.size();
            const amrex::Geometry &geom =
                patchdata.amrcore->Geom(leveldata.level.index());
            const amrex::Real *const x0 = geom.ProbLo();
            const amrex::Real *const dx = geom.CellSize();
            for (int c = 0; c < ncomponents; ++c) {
              const auto slab_box =
                  slice_slab(mfab, c, x0, dx, indextype, slice);
              if (!slab_box)
                continue;
              const Arith::vect<int, 3> &sub_lo = slab_box->first;
              const Arith::vect<int, 3> &sub_hi = slab_box->second;
              extent_t extent;
              for (int d = 0; d < ndims; ++d) {
                extent[0][d] = x0[d] + sub_lo[d] * dx[d];
                extent[1][d] = x0[d] + (sub_hi[d] - 1) * dx[d];
              }
              extents.push_back(extent);
            }
          }
        }

        // Write multimesh

        std::vector<std::string> meshnames;
        for (const auto &patchdata : ghext->patches()) {
          for (const auto &leveldata : patchdata.levels()) {
            const auto &groupdata = leveldata.group(gi);
            const amrex::MultiFab &mfab = *groupdata.timelevel(tl).mfab;
            const amrex::DistributionMapping &dm = mfab.DistributionMap();
            const int ncomponents = dm.size();
            const amrex::Geometry &geom =
                patchdata.amrcore->Geom(leveldata.level.index());
            const amrex::Real *const x0 = geom.ProbLo();
            const amrex::Real *const dx = geom.CellSize();
            for (int c = 0; c < ncomponents; ++c) {
              if (!slice_slab(mfab, c, x0, dx, indextype, slice))
                continue; // skip components that miss the plane
              const int proc = dm[c];
              const std::string proc_filename =
                  make_subdirname(output_file, cctk_iteration) + "/" +
                  make_filename(output_file, cctk_iteration,
                                proc / ioproc_every);
              const std::string meshname =
                  proc_filename + ":" +
                  make_meshname(nghosts, leveldata.level, c);
              meshnames.push_back(meshname);
            }
          }
        }
        std::vector<const char *> meshname_ptrs;
        meshname_ptrs.reserve(meshnames.size());
        for (const auto &meshname : meshnames)
          meshname_ptrs.push_back(meshname.c_str());

        const DB::ptr<DBoptlist> optlist = DB::make(DBMakeOptlist(10));
        assert(optlist);

        int cycle = cctk_iteration;
        ierr = DBAddOption(optlist.get(), DBOPT_CYCLE, &cycle);
        assert(!ierr);

        double dtime = cctk_time;
        ierr = DBAddOption(optlist.get(), DBOPT_DTIME, &dtime);
        assert(!ierr);

        int quadmesh = DB_QUADMESH;
        ierr = DBAddOption(optlist.get(), DBOPT_MB_BLOCK_TYPE, &quadmesh);
        assert(!ierr);

        int extents_size = 2 * ndims;
        // This needs to have type `double`, even if everything else is
        // `float`
        typedef std::array<std::array<double, ndims>, 2> dextent_t;
#ifdef CCTK_REAL_PRECISION_8
        std::vector<dextent_t> &dextents = extents;
#else
        std::vector<dextent_t> dextents;
        dextents.resize(extents.size());
        for (size_t n = 0; n < dextents.size(); ++n) {
          const auto &ext = extents[n];
          dextent_t dext;
          for (int f = 0; f < 2; ++f)
            for (int d = 0; d < ndims; ++d)
              dext[f][d] = ext[f][d];
          dextents[n] = dext;
        }
#endif
        ierr = DBAddOption(optlist.get(), DBOPT_EXTENTS_SIZE, &extents_size);
        assert(!ierr);
        assert(extents.size() == meshname_ptrs.size());
        ierr = DBAddOption(optlist.get(), DBOPT_EXTENTS, dextents.data());
        assert(!ierr);

        std::vector<int> zonecounts;
        zonecounts.reserve(meshnames.size());
        for (const auto &patchdata : ghext->patches()) {
          for (const auto &leveldata : patchdata.levels()) {
            const auto &groupdata = leveldata.group(gi);
            const tl_t tl{0};
            const amrex::MultiFab &mfab = *groupdata.timelevel(tl).mfab;
            const int ncomponents = mfab.size();
            const amrex::Geometry &geom =
                patchdata.amrcore->Geom(leveldata.level.index());
            const amrex::Real *const x0 = geom.ProbLo();
            const amrex::Real *const dx = geom.CellSize();
            for (int c = 0; c < ncomponents; ++c) {
              const auto slab_box =
                  slice_slab(mfab, c, x0, dx, indextype, slice);
              if (!slab_box)
                continue;
              const Arith::vect<int, 3> &sub_lo = slab_box->first;
              const Arith::vect<int, 3> &sub_hi = slab_box->second;
              std::array<int, ndims> dims_vc;
              for (int d = 0; d < ndims; ++d)
                dims_vc[d] =
                    (sub_hi[d] - sub_lo[d]) + int(indextype.cellCentered(d));
              int zonecount = 1;
              for (int d = 0; d < ndims; ++d)
                zonecount *= dims_vc[d];
              zonecounts.push_back(zonecount);
            }
          }
        }
        assert(zonecounts.size() == meshname_ptrs.size());
        ierr = DBAddOption(optlist.get(), DBOPT_ZONECOUNTS, zonecounts.data());
        assert(!ierr);

        ierr = DBPutMultimesh(metafile.get(), multimeshname.c_str(),
                              meshname_ptrs.size(), meshname_ptrs.data(),
                              nullptr, optlist.get());
        assert(!ierr);

        mesh_nblocks[mesh_props] = meshname_ptrs.size();
      } // if write multimesh

      {
        const std::string multimeshname = make_meshname(nghosts);

        const DB::ptr<DBoptlist> optlist = DB::make(DBMakeOptlist(10));
        assert(optlist);

        int cycle = cctk_iteration;
        ierr = DBAddOption(optlist.get(), DBOPT_CYCLE, &cycle);
        assert(!ierr);

        double dtime = cctk_time;
        ierr = DBAddOption(optlist.get(), DBOPT_DTIME, &dtime);
        assert(!ierr);

        ierr = DBAddOption(optlist.get(), DBOPT_MMESH_NAME,
                           const_cast<char *>(multimeshname.c_str()));
        assert(!ierr);

        int vartype_scalar = DB_VARTYPE_SCALAR;
        ierr = DBAddOption(optlist.get(), DBOPT_TENSOR_RANK, &vartype_scalar);
        assert(!ierr);

        int quadvar = DB_QUADVAR;
        ierr = DBAddOption(optlist.get(), DBOPT_MB_BLOCK_TYPE, &quadvar);
        assert(!ierr);

        for (int vi = 0; vi < numvars; ++vi) {
          const std::string multivarname = make_varname(var_name(gi, vi));

          std::vector<std::string> varnames;
          for (const auto &patchdata : ghext->patches()) {
            for (const auto &leveldata : patchdata.levels()) {
              const auto &groupdata = leveldata.group(gi);
              const tl_t tl{0};
              const amrex::MultiFab &mfab = *groupdata.timelevel(tl).mfab;
              const amrex::DistributionMapping &dm = mfab.DistributionMap();
              const int ncomponents = dm.size();
              const amrex::Geometry &geom =
                  patchdata.amrcore->Geom(leveldata.level.index());
              const amrex::Real *const x0 = geom.ProbLo();
              const amrex::Real *const dx = geom.CellSize();
              for (int c = 0; c < ncomponents; ++c) {
                if (!slice_slab(mfab, c, x0, dx, indextype, slice))
                  continue; // skip components that miss the plane
                const int proc = dm[c];
                const std::string proc_filename =
                    make_subdirname(output_file, cctk_iteration) + "/" +
                    make_filename(output_file, cctk_iteration,
                                  proc / ioproc_every);
                const std::string varname =
                    proc_filename + ":" +
                    make_varname(var_name(gi, vi), leveldata.level, c);
                varnames.push_back(varname);
              }
            }
          }
          std::vector<const char *> varname_ptrs;
          varname_ptrs.reserve(varnames.size());
          for (const auto &varname : varnames)
            varname_ptrs.push_back(varname.c_str());

          const std::size_t nmesh = mesh_nblocks.at(mesh_props);
          if (varname_ptrs.size() != nmesh)
            CCTK_VERROR(
                "2D Silo slice multivar \"%s\" enumerates %zu blocks but its "
                "multimesh \"%s\" enumerates %zu. The slice plane likely "
                "lies "
                "on an inter-box boundary and block membership has become "
                "centering-dependent (see restrict_box_interior in "
                "io_slice.hxx / slice_slab in io_silo.cxx).",
                multivarname.c_str(), varname_ptrs.size(),
                multimeshname.c_str(), nmesh);

          ierr = DBPutMultivar(metafile.get(), multivarname.c_str(),
                               varname_ptrs.size(), varname_ptrs.data(),
                               nullptr, optlist.get());
          assert(!ierr);
        } // for vi
      } // write multivar

    } // for gi

    {
      const std::string dirname = DB::legalize_name(driver_name);
      ierr = DBMkDir(metafile.get(), dirname.c_str());
      assert(!ierr);

      {
        const auto num_patches = ghext->num_patches();

        std::vector<int> values(num_patches);

        for (const auto &patchdata : ghext->patches()) {
          values.at(patchdata.patch.index()) = patchdata.num_levels();
        }

        const std::string varname =
            dirname + "/" + DB::legalize_name("nlevels");

        ierr = DBWrite(metafile.get(), varname.c_str(), values.data(),
                       &num_patches, 1, DB_INT);
        assert(!ierr);
      }

      for (const auto &patchdata : ghext->patches()) {
        for (const auto &leveldata : patchdata.levels()) {
          const amrex::FabArrayBase &fab = *leveldata.fab;
          const int ncomponents = fab.size();
          std::vector<int> boxes(2 * ndims * ncomponents);
          for (int component = 0; component < ncomponents; ++component) {
            const amrex::Box &fabbox = fab.box(component); // valid region
            for (int d = 0; d < ndims; ++d)
              boxes[d + 2 * ndims * component] = fabbox.smallEnd(d);
            for (int d = 0; d < ndims; ++d)
              boxes[d + ndims + 2 * ndims * component] = fabbox.bigEnd(d);
          }
          const int dims[2] = {ncomponents, 2 * ndims};
          const std::string varname =
              dirname + "/" + make_fabarraybasename(leveldata.level);
          ierr = DBWrite(metafile.get(), varname.c_str(), boxes.data(), dims, 2,
                         DB_INT);
          assert(!ierr);
        }
      }

      for (const auto &patchdata : ghext->patches()) {
        for (const auto &leveldata : patchdata.levels()) {
          {
            const std::string varname =
                dirname + "/" +
                DB::legalize_name("iteration_num.m" +
                                  std::to_string(patchdata.patch.index()) +
                                  ".rl" +
                                  std::to_string(leveldata.level.index()));
            long long val = leveldata.iteration.num;
            int dims = 1;
            ierr = DBWrite(metafile.get(), varname.c_str(), &val, &dims, 1,
                           DB_LONG_LONG);
            assert(!ierr);
          }
          {
            const std::string varname =
                dirname + "/" +
                DB::legalize_name("iteration_den.m" +
                                  std::to_string(patchdata.patch.index()) +
                                  ".rl" +
                                  std::to_string(leveldata.level.index()));
            long long val = leveldata.iteration.den;
            int dims = 1;
            ierr = DBWrite(metafile.get(), varname.c_str(), &val, &dims, 1,
                           DB_LONG_LONG);
            assert(!ierr);
          }
          {
            const std::string varname =
                dirname + "/" +
                DB::legalize_name("last_step_iteration.m" +
                                  std::to_string(patchdata.patch.index()) +
                                  ".rl" +
                                  std::to_string(leveldata.level.index()));
            long long val = leveldata.last_step_iteration;
            int dims = 1;
            ierr = DBWrite(metafile.get(), varname.c_str(), &val, &dims, 1,
                           DB_LONG_LONG);
            assert(!ierr);
          }
        }
      }
    }

    {
      const std::string visitname = [&]() {
        std::ostringstream buf;
        buf << output_dir << "/" << output_file << ".silo.visit";
        return buf.str();
      }();
      std::ofstream visit(visitname, std::ios::app);
      assert(visit.good());
      visit << make_filename(output_file, cctk_iteration) << "\n";
    }

    RegisterSiloOutput(cctkGH, output_group, output_dir, output_file, slice);

  } // if write metadata

  interval_meta = nullptr;

  if (io_verbose)
    timer.print();
}

} // namespace CarpetX

#else

namespace CarpetX {
void OutputSilo(const cGH *, const std::vector<bool> &, const std::string &,
                const std::string &, const slice_t &) {
  CCTK_ERROR("Silo is not enabled. The parameter "
             "CarpetX::out_silo_2d_vars must be empty.");
}
} // namespace CarpetX

#endif
