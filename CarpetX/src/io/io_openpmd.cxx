#include "io/io_openpmd.hxx"

#include "hierarchy/driver.hxx"
#include "io/io.hxx"
#include "io/io_meta.hxx"
#include "io/io_openpmd_names.hxx"
#include "io/io_slice.hxx"
#include "io/selection.hxx"
#include "util/refuse.hxx"
#include "util/var_type.hxx"
#include <carpetx.hxx>

#include <div.hxx>
#include <vect.hxx>

#include <CactusBase/IOUtil/src/ioutil_CheckpointRecovery.h>
#include <cctk.h>
#include <cctk_Arguments.h>
#include <cctk_Parameters.h>
#include <util_Network.h>

#include <source_location>

#ifdef HAVE_CAPABILITY_openPMD_api

#include <openPMD/openPMD.hpp>

#ifdef HAVE_CAPABILITY_ADIOS2
#include <adios2.h>
#endif

#if defined _OPENMP
#include <omp.h>
#elif defined __HIPCC__
#define omp_get_max_threads() 1
#define omp_get_num_threads() 1
#define omp_get_thread_num() 0
#define omp_in_parallel() 0
#else
static inline int omp_get_max_threads() { return 1; }
static inline int omp_get_num_threads() { return 1; }
static inline int omp_get_thread_num() { return 0; }
static inline int omp_in_parallel() { return 0; }
#endif

#include <mpi.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <ios>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#if !openPMD_HAVE_MPI
#error                                                                         \
    "CarpetX requires openPMD_api with MPI support. Please use -DopenPMD_USE_MPI=ON when building openPMD."
#endif

namespace CarpetX {

constexpr bool io_verbose = true;

constexpr bool output_ghosts = false;

////////////////////////////////////////////////////////////////////////////////

namespace {

template <typename T, std::size_t N>
constexpr std::array<T, N> reversed(const std::array<T, N> &arr) {
  std::array<T, N> res;
  for (std::size_t n = 0; n < N; ++n)
    res[n] = arr[N - 1 - n];
  return res;
}

template <typename T, int D>
constexpr Arith::vect<T, D> reversed(const Arith::vect<T, D> &arr) {
  Arith::vect<T, D> res;
  for (std::size_t d = 0; d < D; ++d)
    res[d] = arr[D - 1 - d];
  return res;
}

template <typename T> std::vector<T> reversed(std::vector<T> &&vec) {
  std::vector<T> res(vec);
  std::reverse(res.begin(), res.end());
  return res;
}

template <typename T = std::uint64_t, typename C>
std::vector<T> to_vector(const C &source) {
  std::vector<T> res(source.size());
  for (std::size_t n = 0; n < res.size(); ++n)
    res[n] = source[n];
  return res;
}

} // namespace

////////////////////////////////////////////////////////////////////////////////

struct carpetx_openpmd_t {
  carpetx_openpmd_t() = default;

  carpetx_openpmd_t(const carpetx_openpmd_t &) = delete;
  carpetx_openpmd_t(carpetx_openpmd_t &&) = default;
  carpetx_openpmd_t &operator=(const carpetx_openpmd_t &) = delete;
  carpetx_openpmd_t &operator=(carpetx_openpmd_t &&) = default;

  static std::optional<carpetx_openpmd_t> self;

  ////////////////////////////////////////////////////////////////////////////////

  template <typename T, std::size_t D> struct box_t {
    Arith::vect<T, D> lo, hi;
    constexpr friend bool operator==(const box_t &x, const box_t &y) {
      if (x.empty() && y.empty())
        return true;
      if (x.empty() || y.empty())
        return false;
      return all(x.lo == y.lo) && all(x.hi == y.hi);
    }
    constexpr friend bool operator!=(const box_t &x, const box_t &y) {
      return !(x == y);
    }
    constexpr bool empty() const { return any(hi < lo); }
    constexpr Arith::vect<T, D> shape() const {
      Arith::vect<T, D> sh;
      for (std::size_t d = 0; d < D; ++d)
        sh[d] = hi[d] < lo[d] ? T{0} : hi[d] - lo[d];
      return sh;
    }
    constexpr T size() const {
      const auto sh = shape();
      T sz{1};
      for (std::size_t d = 0; d < D; ++d)
        sz *= sh[d];
      return sz;
    }
    constexpr Arith::vect<T, D> stride() const {
      const Arith::vect<T, D> sh = shape();
      Arith::vect<T, D> str;
      T np{1};
      for (std::size_t d = 0; d < D; ++d)
        str[d] = (np *= sh[d]);
      return str;
    }
    constexpr T linear(const Arith::vect<T, D> &index) const {
      const Arith::vect<T, D> str = stride();
      T lin{0};
      if (D > 0) {
        lin = index[0];
        for (std::size_t d = 1; d < D; ++d)
          lin += index[d] * str[d - 1];
      }
      return lin;
    }
    friend std::ostream &operator<<(std::ostream &os, const box_t<T, D> &box) {
      return os << "box_t{lo:" << box.lo << ",hi:" << box.hi << "}";
    }
  };

  template <typename T, typename I, std::size_t D> struct level_layout_t {
    box_t<T, D> rdomain;
    Arith::vect<bool, D> is_cell_centred;
    box_t<I, D> idomain;
    constexpr Arith::vect<T, D> rcoord(const Arith::vect<I, D> &icoord) const {
      Arith::vect<T, D> r;
      for (std::size_t d = 0; d < D; ++d) {
        const T rlo = rdomain.lo[d];
        const T rhi = rdomain.hi[d];
        const I ilo2 = 2 * idomain.lo[d] - is_cell_centred;
        const I ihi2 = 2 * idomain.hi[d] + is_cell_centred;
        const I i2 = 2 * icoord[d];
        // The expression below has been carefully constructed to
        // avoid round-off errors at i=ilo and i=ihi. Do not rearrange
        // the terms without preserving this property.
        r[d] = (T(ihi2 - i2) / T(ihi2 - ilo2)) * rlo[d] +
               (T(i2 - ilo2) / T(ihi2 - ilo2)) * rhi[d];
      }
      return r;
    }
    std::vector<box_t<I, D> > grids;
    Arith::vect<std::vector<I>, 2> offsets_sizes() const {
      std::vector<I> offsets(grids.size() + 1), sizes(grids.size());
      I offset{0};
      for (std::size_t n = 0; n < grids.size(); ++n) {
        const T size = grids[n].size();
        offsets[n] = offset;
        sizes[n] = size;
        offset += size;
      }
      offsets[grids.size()] = offset;
      return {offsets, sizes};
    }
  };

  template <typename T, typename I, std::size_t D> struct grid_structure_t {
    box_t<T, D> rdomain;
    std::vector<level_layout_t<T, I, D> > levels;
  };

  ////////////////////////////////////////////////////////////////////////////////

  static std::string make_meshname(const GHExt::GroupRecord &group,
                                   const level_t level, const tl_t tl) {
    return CarpetX::make_meshname(group.groupname, level, tl);
  }

  static std::string make_componentname(const GHExt::GroupRecord &group,
                                        const vi_t vi) {
    return CarpetX::make_componentname(CCTK_FullVarName(group.var(vi).index()));
  }

  ////////////////////////////////////////////////////////////////////////////////

  std::optional<std::string> filename;
  std::optional<openPMD::Series> series;
  std::optional<openPMD::WriteIterations> write_iters;

  void write_openpmd_slice(const cGH *const cctkGH,
                           const group_selection_t &selection,
                           const std::string &output_dir,
                           const std::string &output_file,
                           const slice_t &slice);
};

////////////////////////////////////////////////////////////////////////////////

std::optional<carpetx_openpmd_t> carpetx_openpmd_t::self;

void OutputOpenPMD(const cGH *const cctkGH) {
  const operation_scope_t op("CarpetX::OutputOpenPMD");
  DECLARE_CCTK_PARAMETERS;

  const group_selection_t selection = group_selection_t::select_groups(
      "out_openpmd_2d_vars", out_openpmd_2d_vars);
  if (selection.empty())
    return;
  selection.log("openPMD-2D", group_kind::grid_function);

  for (const GHExt::GroupRecord &group :
       selection.records(group_kind::grid_function))
    switch (group.var_type) {
    case var_type::real:
      break;
    case var_type::integer:
    case var_type::complex:
      refuse(std::source_location::current(),
             "openPMD-2D writes only real grid functions; group %s is %s",
             group.groupname.c_str(),
             CCTK_VarTypeName(cctk_code(group.var_type)));
    }

  if (!carpetx_openpmd_t::self)
    carpetx_openpmd_t::self = std::make_optional<carpetx_openpmd_t>();
  const std::string simulation_name = get_simulation_name();
  carpetx_openpmd_t::self->write_openpmd_slice(cctkGH, selection, out_dir,
                                               simulation_name + ".xy",
                                               slice_t{2, out_xyplane_z});
  carpetx_openpmd_t::self->write_openpmd_slice(cctkGH, selection, out_dir,
                                               simulation_name + ".xz",
                                               slice_t{1, out_xzplane_y});
  carpetx_openpmd_t::self->write_openpmd_slice(cctkGH, selection, out_dir,
                                               simulation_name + ".yz",
                                               slice_t{0, out_yzplane_x});
}

void ShutdownOpenPMD() { carpetx_openpmd_t::self.reset(); }

////////////////////////////////////////////////////////////////////////////////

void carpetx_openpmd_t::write_openpmd_slice(const cGH *const cctkGH,
                                            const group_selection_t &selection,
                                            const std::string &output_dir,
                                            const std::string &output_file,
                                            const slice_t &slice) {
  DECLARE_CCTK_ARGUMENTS;
  DECLARE_CCTK_PARAMETERS;

  // Set up timers
  static Timer timer("OutputOpenPMD");
  Interval interval(timer);

  if (io_verbose)
    CCTK_VINFO("OutputOpenPMD...");

  const openPMD::Format format = get_format();

  if (!series) {

    if (io_verbose)
      CCTK_VINFO("Creating openPMD object...");
    const int mode = 0755;
    static std::once_flag create_directory;
    call_once(create_directory, [&]() {
      const int ierr = CCTK_CreateDirectory(mode, output_dir.c_str());
      assert(ierr >= 0);
    });
    std::ostringstream buf;
    switch (iterationEncoding) {
    case openPMD::IterationEncoding::fileBased:
      buf << output_dir << "/" << output_file << ".it%08T"
          << openPMD::suffix(format);
      break;
    case openPMD::IterationEncoding::variableBased:
      buf << output_dir << "/" << output_file << openPMD::suffix(format);
      break;
    default:
      abort();
    }
    filename = std::make_optional<std::string>(buf.str());
    static bool is_first_output = true;
    const openPMD::Access access =
        iterationEncoding == openPMD::IterationEncoding::fileBased
            ? openPMD::Access::CREATE
        : iterationEncoding == openPMD::IterationEncoding::variableBased
            ? (is_first_output ? openPMD::Access::CREATE
                               : openPMD::Access::READ_WRITE)
            : openPMD::Access::READ_ONLY /*error*/;
    is_first_output = false;
    CCTK_VINFO("  options: %s", options.c_str());
    series = std::make_optional<openPMD::Series>(*filename, access,
                                                 MPI_COMM_WORLD, options);
    series->setIterationEncoding(iterationEncoding);

    // Series metadata must be rank-identical (HDF5 metadata writes are
    // collective): broadcast rank 0's author and hostname.
    {
      char author[1000] = {0};
      char const *const user = getenv("USER");
      if (user)
        std::snprintf(author, sizeof author, "%s", user);
      MPI_Bcast(author, sizeof author, MPI_CHAR, 0, MPI_COMM_WORLD);
      if (author[0])
        series->setAuthor(author);
    }
    {
      char hostname[1000];
      Util_GetHostName(hostname, sizeof hostname);
      MPI_Bcast(hostname, sizeof hostname, MPI_CHAR, 0, MPI_COMM_WORLD);
      series->setMachine(hostname);
    }

    write_iters =
        std::make_optional<openPMD::WriteIterations>(series->writeIterations());
  }
  assert(filename);
  assert(series);
  assert(write_iters);

  if (io_verbose)
    CCTK_VINFO("Creating iteration %d...", cctk_iteration);
  openPMD::Iteration write_iter = (*write_iters)[cctk_iteration];
  write_iter.setTime(cctk_time);
  write_iter.setDt(cctk_delta_time);
  write_iter.setTimeUnitSI(Unit::time);

  // Iteration attributes are set on EVERY rank (all inputs are replicated):
  // rank-asymmetric writes deadlock the HDF5 backend's collective metadata.

  // Write parameters
  {
    char *const data = IOUtil_GetAllParameters(cctkGH, 1 /*all*/);
    const std::string parameters(data);
    std::free(data);
    write_iter.setAttribute("AllParameters", parameters);
  }

  const GHExt &ext = driver();
  for (const auto &patchdata : ext.patches()) {
    for (const auto &leveldata : patchdata.levels()) {
      if (io_verbose)
        CCTK_VINFO("Writing patch %d level %d mesh...", patchdata.patch.index(),
                   leveldata.level.index());

      // Determine grid structure

      const int *const nghosts = cctkGH->cctk_nghostzones;
      const amrex::Geometry &geom =
          patchdata.amrcore->Geom(leveldata.level.index());
      const amrex::Real *const xlo = geom.ProbLo();
      const amrex::Real *const xhi = geom.ProbHi();
      const amrex::Real *const dx = geom.CellSize();
      const box_t<CCTK_REAL, 3> rdomain{
          .lo = {xlo[0] - output_ghosts * nghosts[0] * dx[0],
                 xlo[1] - output_ghosts * nghosts[1] * dx[1],
                 xlo[2] - output_ghosts * nghosts[2] * dx[2]},
          .hi = {xhi[0] + output_ghosts * nghosts[0] * dx[0],
                 xhi[1] + output_ghosts * nghosts[1] * dx[1],
                 xhi[2] + output_ghosts * nghosts[2] * dx[2]}};
      const amrex::Box &dom = geom.Domain();
      const amrex::IntVect &ilo = dom.smallEnd();
      const amrex::IntVect &ihi = dom.bigEnd();
      box_t<int, 3> idomain{
          .lo = {ilo[0] - output_ghosts * nghosts[0],
                 ilo[1] - output_ghosts * nghosts[1],
                 ilo[2] - output_ghosts * nghosts[2]},
          .hi = {ihi[0] + output_ghosts * nghosts[0] + 1 + 1,
                 ihi[1] + output_ghosts * nghosts[1] + 1 + 1,
                 ihi[2] + output_ghosts * nghosts[2] + 1 + 1}};
      const Arith::vect<CCTK_REAL, 3> slice_x0{xlo[0], xlo[1], xlo[2]};
      const Arith::vect<CCTK_REAL, 3> slice_dx{dx[0], dx[1], dx[2]};
      const auto r =
          slice.restrict_box(idomain.lo, idomain.hi, slice_x0, slice_dx);
      if (!r)
        continue; // plane misses this (patch, level) entirely
      idomain.lo = r->first;
      idomain.hi = r->second;
      if (io_verbose) {
        CCTK_VINFO("Patch: %d, Level: %d", patchdata.patch.index(),
                   leveldata.level.index());
        CCTK_VINFO("  xmin: [%f,%f,%f]", double(rdomain.lo[0]),
                   double(rdomain.lo[1]), double(rdomain.lo[2]));
        CCTK_VINFO("  xmax: [%f,%f,%f]", double(rdomain.hi[0]),
                   double(rdomain.hi[1]), double(rdomain.hi[2]));
        CCTK_VINFO("  imin: [%d,%d,%d]", int(idomain.lo[0]), int(idomain.lo[1]),
                   int(idomain.lo[2]));
        CCTK_VINFO("  imax: [%d,%d,%d]", int(idomain.hi[0]), int(idomain.hi[1]),
                   int(idomain.hi[2]));
      }

      // Create dataset

      const openPMD::Datatype datatype =
          openPMD::determineDatatype<CCTK_REAL>();
      const openPMD::Extent extent = to_vector(reversed(idomain.shape()));
      const openPMD::Dataset dataset(datatype, extent);

      for (const auto &groupdata : leveldata.groups()) {
        if (selection.contains(groupdata.record.groupindex)) {

          if (io_verbose)
            CCTK_VINFO("Writing group %d %s...",
                       groupdata.record.groupindex.index(),
                       groupdata.record.groupname.c_str());

          const int numvars = groupdata.record.numvars;

          const tl_t tl{0};
          const amrex::MultiFab &mfab = *groupdata.timelevel(tl).mfab;
          const amrex::IndexType &indextype = mfab.ixType();
          const Arith::vect<bool, 3> is_cell_centred{indextype.cellCentered(0),
                                                     indextype.cellCentered(1),
                                                     indextype.cellCentered(2)};

          const int num_local_components = mfab.local_size();

          // Create mesh

          const std::string meshname =
              make_meshname(groupdata.record, leveldata.level, tl);
          if (io_verbose)
            CCTK_VINFO("Defining mesh %s...", meshname.c_str());
          assert(!write_iter.meshes.contains(meshname));
          openPMD::Mesh mesh = write_iter.meshes[meshname];

          mesh.setGeometry(openPMD::Mesh::Geometry::cartesian);
          mesh.setAxisLabels(reversed(std::vector<std::string>{"x", "y", "z"}));
          mesh.setGridSpacing(to_vector<CCTK_REAL>(
              reversed(fmap([](auto x, auto y) { return x / CCTK_REAL(y); },
                            rdomain.hi - rdomain.lo, idomain.shape() - 1))));
          mesh.setGridGlobalOffset(to_vector<double>(reversed(rdomain.lo)));
          mesh.setGridUnitSI(Unit::length);
          mesh.setTimeOffset(CCTK_REAL(0)); // TODO: check interface.ccl

          // Cell centred grids are offset by 1/2
          const Arith::vect<double, 3> position =
              fmap([](auto c) { return 0.5 * c; }, is_cell_centred);

          // Define tensor components

          std::vector<openPMD::MeshRecordComponent> record_components;
          record_components.reserve(numvars);
          for (const vi_t vi : groupdata.record.variables()) {
            const std::string componentname =
                make_componentname(groupdata.record, vi);
            record_components.push_back(mesh[componentname]);
            auto &record_component = record_components.back();
            record_component.setPosition(to_vector<double>(reversed(position)));
          }
          assert(int(record_components.size()) == numvars);

          // Write data

          if (io_verbose)
            CCTK_VINFO("Writing %d variables with %d components...", numvars,
                       num_local_components);

          for (const vi_t vi : groupdata.record.variables())
            record_components.at(vi.index()).resetDataset(dataset);

          for (int local_component = 0; local_component < num_local_components;
               ++local_component) {
            const int component = mfab.IndexArray().at(local_component);

            const amrex::Box &fabbox =
                mfab.fabbox(component); // exterior (with ghosts)
            const box_t<int, 3> extbox{
                .lo = {fabbox.smallEnd(0), fabbox.smallEnd(1),
                       fabbox.smallEnd(2)},
                .hi = {fabbox.bigEnd(0) + 1, fabbox.bigEnd(1) + 1,
                       fabbox.bigEnd(2) + 1}};
            const amrex::Box &validbox =
                mfab.box(component); // interior (without ghosts)
            const box_t<int, 3> intbox{
                .lo = {validbox.smallEnd(0), validbox.smallEnd(1),
                       validbox.smallEnd(2)},
                .hi = {validbox.bigEnd(0) + 1, validbox.bigEnd(1) + 1,
                       validbox.bigEnd(2) + 1}};
            // It seems that openPMD assumes that chunks do not have
            // ghost zones
            assert(!output_ghosts);
            box_t<int, 3> box = output_ghosts ? extbox : intbox;

            const amrex::Box cvalidbox = amrex::enclosedCells(validbox);
            const Arith::vect<int, 3> cval_lo{cvalidbox.smallEnd(0),
                                              cvalidbox.smallEnd(1),
                                              cvalidbox.smallEnd(2)};
            const Arith::vect<int, 3> cval_hi{cvalidbox.bigEnd(0) + 1,
                                              cvalidbox.bigEnd(1) + 1,
                                              cvalidbox.bigEnd(2) + 1};
            const auto slab =
                slice.restrict_component(box.lo, box.hi, cval_lo, cval_hi,
                                         slice_x0, slice_dx, is_cell_centred);
            if (!slab)
              continue; // this component does not intersect the plane
            box.lo = slab->first;
            box.hi = slab->second;

            Arith::vect<int, 3> start_vec = box.lo - idomain.lo;
            const int n = slice.normal_dir;
            // The slab must fall inside this component's exterior extent.
            assert(box.lo[n] >= extbox.lo[n] && box.hi[n] <= extbox.hi[n]);
            // idomain is vertex-framed, so box.lo - idomain.lo is wrong for
            // the thickness-1 slab; force the normal offset to 0.
            start_vec[n] = 0;

            const openPMD::Offset start = to_vector(reversed(start_vec));
            const openPMD::Extent count = to_vector(reversed(box.shape()));
            const int np = box.size();
            assert(int(count.at(0) * count.at(1) * count.at(2)) == np);
            for (int d = 0; d < 3; ++d)
              assert(
                  start.at(d) <
                  std::numeric_limits<
                      std::remove_reference_t<decltype(start.at(d))> >::max() /
                      2);
            for (int d = 0; d < 3; ++d)
              assert(start.at(d) + count.at(d) <= extent.at(d));

            const amrex::FArrayBox &fab = mfab[component];
            for (const vi_t vi : groupdata.record.variables()) {
              openPMD::MeshRecordComponent &record_component =
                  record_components.at(vi.index());
              if (box == extbox) {
                const CCTK_REAL *const ptr = fab.dataPtr() + vi.index() * np;
#if OPENPMDAPI_VERSION_GE(0, 15, 0)
                record_component.storeChunkRaw(ptr, start, count);
#else
                record_component.storeChunk(openPMD::shareRaw(ptr), start,
                                            count);
#endif
              } else {
                std::shared_ptr<CCTK_REAL> ptr(
                    new CCTK_REAL[np], std::default_delete<CCTK_REAL[]>());
                extract_subbox(ptr.get(),
                               fab.dataPtr() + vi.index() * extbox.size(),
                               extbox.lo, extbox.hi, box.lo, box.hi);
                record_component.storeChunk(std::move(ptr), start, count);
              }
            } // for vi
          } // for local_component
        }
      } // for gi

    } // for leveldata
  } // for patchdata

  if (CCTK_MyProc(nullptr) == 0) {
    output_file_description_t ofd;
    ofd.filename = *filename;
    ofd.description = "2D CarpetX openPMD slice output";
    ofd.writer_thorn = CCTK_THORNSTRING;
    for (const GHExt::GroupRecord &group :
         selection.records(group_kind::grid_function))
      for (const vi_t vi : group.variables())
        ofd.variables.push_back(CCTK_FullVarName(group.var(vi).index()));
    ofd.iterations = {cctk_iteration};
    for (const int d : slice.inplane_dirs())
      ofd.output_directions.push_back(d);
    ofd.format_name = "CarpetX/openPMD";
    ofd.format_version = {1, 0, 0};
    OutputMeta_RegisterOutputFile(std::move(ofd));
  }

  if (io_verbose)
    CCTK_VINFO("OutputOpenPMD: Performing all writes...");
  series->flush();

  if (io_verbose)
    CCTK_VINFO("OutputOpenPMD: Closing iteration...");
  write_iter.close();

  if (CCTK_MyProc(nullptr) == 0) {
    std::ostringstream buf;
    buf << output_dir << "/" << output_file << ".openpmd.visit";
    const std::string visitname = buf.str();
    std::ofstream visit(visitname, std::ios::app);
    assert(visit.good());
    switch (iterationEncoding) {
    case openPMD::IterationEncoding::fileBased:
      visit << output_file << ".it" << std::setw(8) << std::setfill('0')
            << cctk_iteration << openPMD::suffix(format) << "\n";
      break;
    case openPMD::IterationEncoding::variableBased:
      visit << output_file << openPMD::suffix(format) << "\n";
      break;
    default:
      abort();
    }
  }

  switch (iterationEncoding) {
  case openPMD::IterationEncoding::fileBased:
    write_iters.reset();
    series.reset();
    filename.reset();
    break;
  case openPMD::IterationEncoding::variableBased:
    // do nothing
    break;
  default:
    abort();
  }

  if (io_verbose)
    CCTK_VINFO("OutputOpenPMD done.");

  if (io_verbose)
    timer.print();
}

} // namespace CarpetX

#else

namespace CarpetX {
void OutputOpenPMD(const cGH *) {
  const operation_scope_t op("CarpetX::OutputOpenPMD");
  DECLARE_CCTK_PARAMETERS;
  if (group_selection_t::select_groups("out_openpmd_2d_vars",
                                       out_openpmd_2d_vars)
          .empty())
    return;
  refuse(std::source_location::current(),
         "CarpetX::out_openpmd_2d_vars is set, but this build has no openPMD");
}

void ShutdownOpenPMD() {}
} // namespace CarpetX

#endif // #ifdef HAVE_CAPABILITY_OPENPMD
