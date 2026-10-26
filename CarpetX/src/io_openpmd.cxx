#include "io_openpmd.hxx"

#include "driver.hxx"
#include "io_meta.hxx"
#include "io_openpmd_names.hxx"
#include "io_slice.hxx"
#include "schedule.hxx"
#include "valid.hxx"
#include <carpetx.hxx>

#include <div.hxx>
#include <vect.hxx>

#include <CactusBase/IOUtil/src/ioutil_CheckpointRecovery.h>
#include <cctk.h>
#include <cctk_Arguments.h>
#include <cctk_Parameters.h>
#include <util_Network.h>

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

constexpr bool input_ghosts = false;
constexpr bool output_ghosts = false;

////////////////////////////////////////////////////////////////////////////////

namespace {

// TODO: Get this from `valid.cxx`
inline CCTK_REAL get_poison() {
#if defined CCTK_REAL_PRECISION_4
  constexpr std::uint32_t ipoison = 0xffc00000UL + 0xdead;
#elif defined CCTK_REAL_PRECISION_8
  constexpr std::uint64_t ipoison = 0xfff8000000000000ULL + 0xdeadbeef;
#endif
  static_assert(sizeof ipoison == sizeof(CCTK_REAL));
  CCTK_REAL poison;
  std::memcpy(&poison, &ipoison, sizeof poison);
  return poison;
}

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

template <typename T> std::vector<T> reversed(const std::vector<T> &vec) {
  std::vector<T> res(vec.size());
  std::reverse_copy(vec.begin(), vec.end(), res.begin());
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

template <typename T> std::string vector_string(const std::vector<T> &vec) {
  std::ostringstream buf;
  buf << "[";
  for (std::size_t n = 0; n < vec.size(); ++n) {
    if (n != 0)
      buf << ",";
    buf << vec[n];
  }
  buf << "]";
  return buf.str();
}

} // namespace

////////////////////////////////////////////////////////////////////////////////

std::string Geometry_string(const openPMD::Mesh::Geometry &geometry) {
  std::ostringstream buf;
  buf << geometry;
  return buf.str();
}

std::string UnitDimension_string(const std::array<double, 7> &unitDimension) {
  std::ostringstream buf;
  buf << "L:" << unitDimension[uint8_t(openPMD::UnitDimension::L)] << " "
      << "M:" << unitDimension[uint8_t(openPMD::UnitDimension::M)] << " "
      << "T:" << unitDimension[uint8_t(openPMD::UnitDimension::T)] << " "
      << "I:" << unitDimension[uint8_t(openPMD::UnitDimension::I)] << " "
      << "θ:" << unitDimension[uint8_t(openPMD::UnitDimension::theta)] << " "
      << "N:" << unitDimension[uint8_t(openPMD::UnitDimension::N)] << " "
      << "J:" << unitDimension[uint8_t(openPMD::UnitDimension::J)];
  return buf.str();
}

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

  static std::string make_meshname(const gi_t gi, const int patch,
                                   const int level, const tl_t tl = tl_t{0},
                                   const std::string &band = std::string()) {
    assert((patch == -1) == (level == -1));
    const std::optional<level_t> where =
        level == -1 ? std::nullopt
                    : std::optional<level_t>(level_t{patch_t{patch}, level});
    return CarpetX::make_meshname(CCTK_FullGroupName(gi.index()), where, tl,
                                  band);
  }

  static std::string make_componentname(const gi_t gi, const int vi) {
    return CarpetX::make_componentname(
        CCTK_FullVarName(ghext->record(gi).var(vi_t{vi}).index()));
  }

  ////////////////////////////////////////////////////////////////////////////////

  std::optional<std::string> filename;
  std::optional<openPMD::Series> series;
  std::optional<openPMD::Iteration> read_iter;
  std::optional<openPMD::WriteIterations> write_iters;

  void InputOpenPMD(const cGH *const cctkGH,
                    const std::vector<bool> &input_group,
                    const std::string &input_dir, const std::string &input_file,
                    TimeLevelMode tl_mode,
                    const checkpoint_record_t *recovering);

  void OutputOpenPMD(const cGH *const cctkGH,
                     const std::vector<bool> &output_group,
                     const std::string &output_dir,
                     const std::string &output_file,
                     TimeLevelMode tl_mode = TimeLevelMode::Current,
                     std::optional<slice_t> slice = std::nullopt);
};

////////////////////////////////////////////////////////////////////////////////

std::optional<carpetx_openpmd_t> carpetx_openpmd_t::self;

void InputOpenPMD(const cGH *cctkGH, const std::vector<bool> &input_group,
                  const std::string &input_dir, const std::string &input_file,
                  TimeLevelMode tl_mode,
                  const checkpoint_record_t *const recovering) {
  if (!carpetx_openpmd_t::self)
    carpetx_openpmd_t::self = std::make_optional<carpetx_openpmd_t>();
  carpetx_openpmd_t::self->InputOpenPMD(cctkGH, input_group, input_dir,
                                        input_file, tl_mode, recovering);
}

void OutputOpenPMD(const cGH *const cctkGH,
                   const std::vector<bool> &output_group,
                   const std::string &output_dir,
                   const std::string &output_file, TimeLevelMode tl_mode,
                   std::optional<slice_t> slice) {
  if (!carpetx_openpmd_t::self)
    carpetx_openpmd_t::self = std::make_optional<carpetx_openpmd_t>();
  carpetx_openpmd_t::self->OutputOpenPMD(cctkGH, output_group, output_dir,
                                         output_file, tl_mode, slice);
}

void ShutdownOpenPMD() { carpetx_openpmd_t::self.reset(); }

////////////////////////////////////////////////////////////////////////////////

void carpetx_openpmd_t::InputOpenPMD(
    const cGH *const cctkGH, const std::vector<bool> &input_group,
    const std::string &input_dir, const std::string &input_file,
    TimeLevelMode tl_mode, const checkpoint_record_t *const recovering) {
  DECLARE_CCTK_ARGUMENTS;
  DECLARE_CCTK_PARAMETERS;

  // Set up timers
  static Timer timer("InputOpenPMD");
  Interval interval(timer);

  if (std::count(input_group.begin(), input_group.end(), true) == 0)
    return;

  if (io_verbose)
    CCTK_VINFO("InputOpenPMD...");

  const bool is_root = CCTK_MyProc(nullptr) == 0;
  if (is_root) {
    CCTK_VINFO("  openPMD input for groups:");
    for (int gi = 0; gi < CCTK_NumGroups(); ++gi)
      if (input_group.at(gi))
        CCTK_VINFO("    %s", CCTK_FullGroupName(gi));
  }

  if (!series) {
    if (io_verbose)
      CCTK_VINFO("Creating openPMD object...");
    const openPMD::Format format = get_format();
    std::ostringstream buf;
    switch (iterationEncoding) {
    case openPMD::IterationEncoding::fileBased:
      buf << input_dir << "/" << input_file << ".it%08T"
          << openPMD::suffix(format);
      break;
    case openPMD::IterationEncoding::variableBased:
      buf << input_dir << "/" << input_file << openPMD::suffix(format);
      break;
    default:
      abort();
    }
    filename = std::make_optional<std::string>(buf.str());
    series = std::make_optional<openPMD::Series>(
        *filename, openPMD::Access::READ_ONLY, MPI_COMM_WORLD, options);
    assert(series->iterations.count(cctkGH->cctk_iteration));
    read_iter = std::make_optional<openPMD::Iteration>(
        series->iterations[cctkGH->cctk_iteration]);
  }
  assert(filename);
  assert(series);
  assert(read_iter);

  CCTK_VINFO("  iteration: %d", cctkGH->cctk_iteration);

  const CCTK_REAL time = read_iter->time<CCTK_REAL>();
  const CCTK_REAL dt = read_iter->dt<CCTK_REAL>();
  const double timeUnitSI = read_iter->timeUnitSI();
  CCTK_VINFO("  time: %f", double(time));
  CCTK_VINFO("  dt: %f", double(dt));
  CCTK_VINFO("  time unit SI: %f", timeUnitSI);

  openPMD::Container<openPMD::Mesh> &meshes = read_iter->meshes;
  CCTK_VINFO("  found %d meshes", int(meshes.size()));

  if (io_verbose) {
    for (auto mesh_iter = meshes.begin(); mesh_iter != meshes.end();
         ++mesh_iter) {
      const std::string &mesh_name = mesh_iter->first;
      openPMD::Mesh &mesh = mesh_iter->second;
      CCTK_VINFO("    mesh: %s", mesh_name.c_str());

      const openPMD::Mesh::Geometry geometry = mesh.geometry();
      const std::vector<std::string> axisLabels = mesh.axisLabels();
      const std::vector<CCTK_REAL> gridSpacing = mesh.gridSpacing<CCTK_REAL>();
      const std::vector<double> gridGlobalOffset = mesh.gridGlobalOffset();
      const double gridUnitSI = mesh.gridUnitSI();
      const std::array<double, 7> unitDimension = mesh.unitDimension();
      const CCTK_REAL timeOffset = mesh.timeOffset<CCTK_REAL>();
      CCTK_VINFO("      geometry: %s", Geometry_string(geometry).c_str());
      CCTK_VINFO("      axis labels: %s",
                 vector_string(reversed(axisLabels)).c_str());
      CCTK_VINFO("      grid spacing: %s",
                 vector_string(reversed(gridSpacing)).c_str());
      CCTK_VINFO("      grid global offset: %s",
                 vector_string(reversed(gridGlobalOffset)).c_str());
      CCTK_VINFO("      grid unit SI: %f", gridUnitSI);
      CCTK_VINFO("      unit dimension: %s",
                 UnitDimension_string(unitDimension).c_str());
      CCTK_VINFO("      time offset: %f", double(timeOffset));

      CCTK_VINFO("      found %d components", int(mesh.size()));
      openPMD::Extent extent;
      for (auto record_component_iter = mesh.begin();
           record_component_iter != mesh.end(); ++record_component_iter) {
        const std::string &record_component_name = record_component_iter->first;
        openPMD::MeshRecordComponent &record_component =
            record_component_iter->second;
        CCTK_VINFO("        component: %s", record_component_name.c_str());

        const std::vector<CCTK_REAL> position =
            record_component.position<CCTK_REAL>();
        CCTK_VINFO("          position: %s",
                   vector_string(reversed(position)).c_str());

        const int ndims = record_component.getDimensionality();
        if (extent.empty())
          extent = record_component.getExtent();
        else
          assert(extent == record_component.getExtent());
        CCTK_VINFO("          ndims: %d", ndims);
        CCTK_VINFO("          extent: %s",
                   vector_string(reversed(extent)).c_str());

        const std::vector<openPMD::WrittenChunkInfo> chunks =
            record_component.availableChunks();
        CCTK_VINFO("          found %d chunks", int(chunks.size()));
        if (mesh_iter == meshes.begin() &&
            record_component_iter == mesh.begin()) {
          for (std::size_t n = 0; n < chunks.size(); ++n) {
            const openPMD::WrittenChunkInfo &chunk = chunks.at(n);
            CCTK_VINFO("            chunk: %d   start: %s   count: %s", int(n),
                       vector_string(reversed(chunk.offset)).c_str(),
                       vector_string(reversed(chunk.extent)).c_str());
          }
        }

      } // for record_component
    } // for mesh
  }

  // Post-read tasks
  std::vector<std::function<void()> > tasks;

  std::vector<write_scope_t> scopes;
  std::vector<array_write_scope_t> array_scopes;

  // First read grid functions in a loop over patches and levels

  for (auto &patchdata : ghext->patches()) {
    for (auto &leveldata : patchdata.levels()) {
      if (io_verbose)
        CCTK_VINFO("Reading patch %d level %d", patchdata.patch.index(),
                   leveldata.level.index());

      // Determine grid structure

      const int *const nghosts = cctkGH->cctk_nghostzones;
      const amrex::Geometry &geom =
          patchdata.amrcore->Geom(leveldata.level.index());
      const amrex::Real *const xlo = geom.ProbLo();
      const amrex::Real *const xhi = geom.ProbHi();
      const amrex::Real *const dx = geom.CellSize();
      const box_t<CCTK_REAL, 3> rdomain{
          .lo = {xlo[0] - input_ghosts * nghosts[0] * dx[0],
                 xlo[1] - input_ghosts * nghosts[1] * dx[1],
                 xlo[2] - input_ghosts * nghosts[2] * dx[2]},
          .hi = {xhi[0] + input_ghosts * nghosts[0] * dx[0],
                 xhi[1] + input_ghosts * nghosts[1] * dx[1],
                 xhi[2] + input_ghosts * nghosts[2] * dx[2]}};
      const amrex::Box &dom = geom.Domain();
      const amrex::IntVect &ilo = dom.smallEnd();
      const amrex::IntVect &ihi = dom.bigEnd();
      const box_t<int, 3> idomain{
          .lo = {ilo[0] - input_ghosts * nghosts[0],
                 ilo[1] - input_ghosts * nghosts[1],
                 ilo[2] - input_ghosts * nghosts[2]},
          .hi = {ihi[0] + input_ghosts * nghosts[0] + 1 + 1,
                 ihi[1] + input_ghosts * nghosts[1] + 1 + 1,
                 ihi[2] + input_ghosts * nghosts[2] + 1 + 1}};
      if (io_verbose) {
        CCTK_VINFO("Level: %d", leveldata.level.index());
        CCTK_VINFO("  xmin: [%f,%f,%f]", double(rdomain.lo[0]),
                   double(rdomain.lo[1]), double(rdomain.lo[2]));
        CCTK_VINFO("  xmax: [%f,%f,%f]", double(rdomain.hi[0]),
                   double(rdomain.hi[1]), double(rdomain.hi[2]));
        CCTK_VINFO("  imin: [%d,%d,%d]", int(idomain.lo[0]), int(idomain.lo[1]),
                   int(idomain.lo[2]));
        CCTK_VINFO("  imax: [%d,%d,%d]", int(idomain.hi[0]), int(idomain.hi[1]),
                   int(idomain.hi[2]));
      }

      for (auto &groupdata : leveldata.groups()) {
        const gi_t gi = groupdata.record.groupindex;
        if (input_group.at(gi.index())) {

          cGroup cgroup;
          const int ierr = CCTK_GroupData(gi.index(), &cgroup);
          assert(!ierr);
          assert(cgroup.vartype == CCTK_VARIABLE_REAL);
          assert(cgroup.dim == 3);

          if (io_verbose)
            CCTK_VINFO("Reading group %s...", CCTK_FullGroupName(gi.index()));

          const int numvars = groupdata.record.numvars;

          const tl_range tls = tl_mode == TimeLevelMode::All
                                   ? groupdata.record.timelevels()
                                   : tl_range{tl_t{0}, tl_t{1}};
          for (const tl_t tl : tls) {
            amrex::MultiFab &mfab = *groupdata.timelevel(tl).mfab;
            const amrex::IndexType &indextype = mfab.ixType();
            const Arith::vect<bool, 3> is_cell_centred{
                indextype.cellCentered(0), indextype.cellCentered(1),
                indextype.cellCentered(2)};

            const int num_local_components = mfab.local_size();

            // Read mesh

            const std::string meshname =
                make_meshname(gi, leveldata.level.patch().index(),
                              leveldata.level.index(), tl);
            if (io_verbose)
              CCTK_VINFO("Reading mesh %s...", meshname.c_str());
            assert(read_iter->meshes.count(meshname));
            const openPMD::Mesh &mesh = read_iter->meshes.at(meshname);
            // TODO: The openPMD standard says to add an attribute
            // `refinementRatio`, which is a vector of integers

            // Define tensor components

            // TODO: Set component names according to the tensor type
            std::vector<openPMD::MeshRecordComponent> record_components;
            record_components.reserve(numvars);
            openPMD::Extent extent;
            for (int vi = 0; vi < numvars; ++vi) {
              const std::string componentname = make_componentname(gi, vi);
              assert(mesh.count(componentname));
              record_components.push_back(mesh.at(componentname));
              const openPMD::MeshRecordComponent &record_component =
                  record_components.back();
              if (vi == 0)
                extent = record_component.getExtent();
              else
                assert(extent == record_component.getExtent());
            }
            assert(int(record_components.size()) == numvars);

            // Read data

            if (io_verbose)
              CCTK_VINFO("Reading %d variables with %d components...", numvars,
                         num_local_components);

            const bool group_has_no_ghosts = groupdata.nghostzones[0] == 0 &&
                                             groupdata.nghostzones[1] == 0 &&
                                             groupdata.nghostzones[2] == 0;

            scopes.push_back(
                open_write(window_t::one_level(leveldata.level),
                           {.gi = gi,
                            .tl = tl,
                            .fills = input_ghosts || group_has_no_ghosts
                                         ? make_valid_all()
                                         : make_valid_int(),
                            .keeps = valid_t(),
                            .event = "InputOpenPMD"}));

            // Loop over components (AMReX boxes)
            for (int local_component = 0;
                 local_component < num_local_components; ++local_component) {
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
              const box_t<int, 3> &box = input_ghosts ? extbox : intbox;

              const openPMD::Offset start =
                  to_vector(reversed(box.lo - idomain.lo));
              const openPMD::Extent count = to_vector(reversed(box.shape()));
              const int np = box.size();
              assert(int(count.at(0) * count.at(1) * count.at(2)) == np);
              for (int d = 0; d < 3; ++d)
                assert(start.at(d) <
                       std::numeric_limits<std::remove_reference_t<decltype(
                               start.at(d))> >::max() /
                           2);
              for (int d = 0; d < 3; ++d)
                assert(start.at(d) + count.at(d) <= extent.at(d));

              amrex::FArrayBox &fab = mfab[component];
              for (int vi = 0; vi < numvars; ++vi) {

                if (input_ghosts || intbox == extbox) {
                  CCTK_REAL *const ptr = fab.dataPtr() + vi * np;
#if OPENPMDAPI_VERSION_GE(0, 15, 0)
                  record_components.at(vi).loadChunkRaw(ptr, start, count);
#else
                  record_components.at(vi).loadChunk(openPMD::shareRaw(ptr),
                                                     start, count);
#endif

                } else {
                  const int amrex_size = extbox.size();
                  CCTK_REAL *const amrex_var_ptr =
                      fab.dataPtr() + vi * amrex_size;
                  const Arith::vect<int, 3> amrex_shape = extbox.shape();
                  const Arith::vect<int, 3> amrex_offset = box.lo - extbox.lo;
                  constexpr int amrex_di = 1;
                  const int amrex_dj = amrex_di * amrex_shape[0];
                  const int amrex_dk = amrex_dj * amrex_shape[1];
                  CCTK_REAL *const amrex_ptr =
                      amrex_var_ptr + amrex_di * amrex_offset[0] +
                      amrex_dj * amrex_offset[1] + amrex_dk * amrex_offset[2];
                  const Arith::vect<int, 3> contig_shape = box.shape();
                  constexpr int contig_di = 1;
                  const int contig_dj = contig_di * contig_shape[0];
                  const int contig_dk = contig_dj * contig_shape[1];
                  const int contig_np = contig_dk * contig_shape[2];
                  assert(contig_np == np);
#if 1
                  CCTK_REAL *const contig_ptr =
                      amrex_var_ptr + extbox.size() - box.size();
                  // TODO: optimize memory layout
#if 0
                CCTK_REAL *const contig_ptr =
                    ptr + contig_di * (contig_shape[0] - 1) +
                    contig_dj * (contig_shape[1] - 1) +
                    contig_dk * (contig_shape[2] - 1) + 1 - contig_np;
                assert(&amrex_ptr[amrex_di * (amrex_shape[0] - 1) +
                                  amrex_dj * (amrex_shape[1] - 1) +
                                  amrex_dk * (amrex_shape[2] - 1)] ==
                       &contig_ptr[contig_di * (contig_shape[0] - 1) +
                                   contig_dj * (contig_shape[1] - 1) +
                                   contig_dk * (contig_shape[2] - 1)]);
#endif
                  if (poison_undefined_values) {
                    const CCTK_REAL poison = get_poison();
#pragma omp simd
                    for (int n = 0; n < np; ++n)
                      contig_ptr[n] = poison;
                  }
                  record_components.at(vi).loadChunkRaw(contig_ptr, start,
                                                        count);
                  tasks.emplace_back([=]() {
                    for (int k = 0; k < contig_shape[2]; ++k)
                      for (int j = 0; j < contig_shape[1]; ++j)
                        for (int i = 0; i < contig_shape[0]; ++i)
                          amrex_ptr[amrex_di * i + amrex_dj * j +
                                    amrex_dk * k] =
                              contig_ptr[contig_di * i + contig_dj * j +
                                         contig_dk * k];
                  });
#else
                  CCTK_REAL *const contig_ptr = new CCTK_REAL[np];
                  if (poison_undefined_values) {
                    const CCTK_REAL poison = get_poison();
#pragma omp simd
                    for (int n = 0; n < np; ++n)
                      contig_ptr[n] = poison;
                  }
                  record_components.at(vi).loadChunkRaw(contig_ptr, start,
                                                        count);
                  tasks.emplace_back([=]() {
                    for (int k = 0; k < contig_shape[2]; ++k)
                      for (int j = 0; j < contig_shape[1]; ++j)
#pragma omp simd
                        for (int i = 0; i < contig_shape[0]; ++i)
                          amrex_ptr[amrex_di * i + amrex_dj * j +
                                    amrex_dk * k] =
                              contig_ptr[contig_di * i + contig_dj * j +
                                         contig_dk * k];
                    delete[] contig_ptr;
                  });
#endif
                }

              } // for vi
            } // for local_component
          } // for tl

          {
            const bool need_bands =
                recovering &&
                finer_level_is_behind(*recovering, leveldata.level);
            const auto read_band = [&](amrex::MultiFab *const band,
                                       const band_kind kind, const int stage) {
              if (!band || band->empty())
                return;
              assert(band->nGrowVect() == 0);
              const std::string band_tag = subcycling_band_tag(kind, stage);
              const std::string meshname =
                  make_meshname(gi, leveldata.level.patch().index(),
                                leveldata.level.index(), tl_t{0}, band_tag);
              if (!read_iter->meshes.count(meshname)) {
                if (need_bands)
                  CCTK_VERROR(
                      "Mid-cycle checkpoint lacks %s: mesh \"%s\" (band %s) "
                      "for group %s on patch %d level %d is missing. The "
                      "checkpoint was written by %s or is incomplete. "
                      "Restart from a time-aligned checkpoint.",
                      kind == band_kind::flux_register ? "flux-register data"
                                                       : "coarse RK step data",
                      meshname.c_str(), band_tag.c_str(),
                      CCTK_FullGroupName(gi.index()),
                      leveldata.level.patch().index(), leveldata.level.index(),
                      kind == band_kind::flux_register
                          ? "a version without flux-register checkpointing"
                          : "the derivative-band scheme");
                return; // not needed: time-aligned, or a register the
                        // parent resets before its next reflux
              }
              if (io_verbose)
                CCTK_VINFO("Reading band mesh %s...", meshname.c_str());
              const openPMD::Mesh &mesh = read_iter->meshes.at(meshname);

              std::vector<openPMD::MeshRecordComponent> record_components;
              record_components.reserve(numvars);
              openPMD::Extent extent;
              for (int vi = 0; vi < numvars; ++vi) {
                const std::string componentname = make_componentname(gi, vi);
                assert(mesh.count(componentname));
                record_components.push_back(mesh.at(componentname));
                if (vi == 0)
                  extent = record_components.back().getExtent();
              }
              assert(int(record_components.size()) == numvars);

              const int num_local_components = band->local_size();
              for (int local_component = 0;
                   local_component < num_local_components; ++local_component) {
                const int component = band->IndexArray().at(local_component);

                const amrex::Box &fabbox =
                    band->fabbox(component); // zero ghost
                const box_t<int, 3> box{
                    .lo = {fabbox.smallEnd(0), fabbox.smallEnd(1),
                           fabbox.smallEnd(2)},
                    .hi = {fabbox.bigEnd(0) + 1, fabbox.bigEnd(1) + 1,
                           fabbox.bigEnd(2) + 1}};

                const openPMD::Offset start =
                    to_vector(reversed(box.lo - idomain.lo));
                const openPMD::Extent count = to_vector(reversed(box.shape()));
                const int np = box.size();
                assert(int(count.at(0) * count.at(1) * count.at(2)) == np);
                for (int d = 0; d < 3; ++d)
                  assert(start.at(d) + count.at(d) <= extent.at(d));

                amrex::FArrayBox &fab = (*band)[component];
                for (int vi = 0; vi < numvars; ++vi) {
                  CCTK_REAL *const ptr = fab.dataPtr() + vi * np;
#if OPENPMDAPI_VERSION_GE(0, 15, 0)
                  record_components.at(vi).loadChunkRaw(ptr, start, count);
#else
                  record_components.at(vi).loadChunk(openPMD::shareRaw(ptr),
                                                     start, count);
#endif
                } // for vi
              } // for local_component
            };

            for (int s = 0; s < max_num_rk_stages; ++s)
              read_band(
                  rk_source_band(leveldata.level, gi, band_kind::ks_source, s),
                  band_kind::ks_source, s);
            read_band(
                rk_source_band(leveldata.level, gi, band_kind::old_source),
                band_kind::old_source, -1);
            for (int f = 0; f < 2 * dim; ++f)
              read_band(rk_source_band(leveldata.level, gi,
                                       band_kind::flux_register, f),
                        band_kind::flux_register, f);
          }
        }
      } // for gi

    } // for leveldata
  } // for patchdata

  // Next read grid scalars and grid arrays

  {
    const int numgroups = CCTK_NumGroups();
    for (int gi0 = 0; gi0 < numgroups; ++gi0) {
      const gi_t gi{gi0};
      if (input_group.at(gi.index())) {

        // Check group properties

        cGroup cgroup;
        const int ierr = CCTK_GroupData(gi.index(), &cgroup);
        assert(!ierr);
        if (cgroup.grouptype == CCTK_GF)
          continue;
        assert(cgroup.disttype == CCTK_DISTRIB_CONSTANT);
        assert(cgroup.dim >= 0);
        assert(cgroup.dim <= 3);

        if (io_verbose)
          CCTK_VINFO("Reading group %d %s...", gi.index(),
                     CCTK_FullGroupName(gi.index()));

        auto &groupdata = ghext->record(gi).array();
        if (!groupdata.record.has_storage())
          continue;
        const int numvars = groupdata.record.numvars;
        const tl_t tl{0};

        // Determine grid structure

        using ivect = Arith::vect<int, dim>;

        const box_t<int, 3> idomain{.lo = ivect{0, 0, 0},
                                    .hi = ivect(groupdata.gsh)};

        // Read mesh

        const std::string meshname = make_meshname(gi, -1, -1);
        assert(read_iter->meshes.count(meshname));
        const openPMD::Mesh &mesh = read_iter->meshes.at(meshname);
        // TODO: The openPMD standard says to add an attribute
        // `refinementRatio`, which is a vector of integers

        // Define tensor components

        // TODO: Set component names according to the tensor type
        std::vector<openPMD::MeshRecordComponent> record_components;
        record_components.reserve(numvars);
        openPMD::Extent extent;
        for (int vi = 0; vi < numvars; ++vi) {
          const std::string componentname = make_componentname(gi, vi);
          assert(mesh.count(componentname));
          record_components.push_back(mesh.at(componentname));
          const openPMD::MeshRecordComponent &record_component =
              record_components.back();
          if (vi == 0)
            extent = record_component.getExtent();
          else
            assert(extent == record_component.getExtent());
        }
        assert(int(record_components.size()) == numvars);

        // Read data

        if (io_verbose)
          CCTK_VINFO("Reading %d variables...", numvars);

        const bool group_has_no_ghosts = groupdata.nghostzones[0] == 0 &&
                                         groupdata.nghostzones[1] == 0 &&
                                         groupdata.nghostzones[2] == 0;

        array_scopes.push_back(open_array_write(
            {.gi = gi,
             .tl = tl,
             .fills = input_ghosts || group_has_no_ghosts ? make_valid_all()
                                                          : make_valid_int(),
             .keeps = valid_t(),
             .event = "InputOpenPMD"}));

        // exterior (with ghosts)
        for (int d = 0; d < dim; ++d)
          assert(groupdata.lsh[d] == groupdata.gsh[d]);
        assert(all(Arith::vect<int, dim>(groupdata.lsh) ==
                   Arith::vect<int, dim>(groupdata.gsh)));
        const box_t<int, 3> extbox{.lo = ivect{0, 0, 0},
                                   .hi = ivect(groupdata.lsh)};
        // interior (without ghosts)
        const box_t<int, 3> intbox{.lo = ivect(groupdata.nghostzones),
                                   .hi = ivect(groupdata.lsh) -
                                         ivect(groupdata.nghostzones)};
        // It seems that openPMD assumes that chunks do not have ghost zones
        assert(!input_ghosts);
        const box_t<int, 3> &box = input_ghosts ? extbox : intbox;

        const openPMD::Offset start = to_vector(reversed(box.lo - idomain.lo));
        const openPMD::Extent count = to_vector(reversed(box.shape()));
        const int np = box.size();
        assert(int(count.at(0) * count.at(1) * count.at(2)) == np);
        for (int d = 0; d < 3; ++d)
          assert(start.at(d) <
                 std::numeric_limits<
                     std::remove_reference_t<decltype(start.at(d))> >::max() /
                     2);
        for (int d = 0; d < 3; ++d)
          assert(start.at(d) + count.at(d) <= extent.at(d));

        const Arith::vect<int, 3> cactus_shape = extbox.shape();
        constexpr int cactus_di = 1;
        const int cactus_dj = cactus_di * cactus_shape[0];
        const int cactus_dk = cactus_dj * cactus_shape[1];
        const int cactus_np = cactus_dk * cactus_shape[2];
        assert(cactus_di > 0);
        assert(cactus_dj > 0);
        assert(cactus_dk > 0);
        assert(cactus_np > 0);
        assert(int(groupdata.timelevel(tl).data.size()) == numvars * cactus_np);
        for (int vi = 0; vi < numvars; ++vi) {
          void *const cactus_var_ptr =
              groupdata.timelevel(tl).data.data_at(vi * cactus_np);
          if (input_ghosts || intbox == extbox) {
#if !OPENPMDAPI_VERSION_GE(0, 15, 0)
#define loadChunkRaw(ptr, start, count)                                        \
  loadChunk(openPMD::shareRaw(ptr), start, count)
#endif
            switch (cgroup.vartype) {
            case CCTK_VARIABLE_REAL:
              record_components.at(vi).loadChunkRaw(
                  static_cast<CCTK_REAL *>(cactus_var_ptr), start, count);
              break;
            case CCTK_VARIABLE_INT:
              record_components.at(vi).loadChunkRaw(
                  static_cast<CCTK_INT *>(cactus_var_ptr), start, count);
              break;
            case CCTK_VARIABLE_COMPLEX:
              record_components.at(vi).loadChunkRaw(
                  static_cast<CCTK_COMPLEX *>(cactus_var_ptr), start, count);
              break;
            default:
              assert(0 && "Unexpected variable type");
            }
          } else {
            auto cactus_ptr = &groupdata.timelevel(tl).data;
            char *const cactus_var_ptr =
                static_cast<char *>(cactus_ptr->data_at(vi * cactus_np));
            const int vartypesize = CCTK_VarTypeSize(cgroup.vartype);
            const Arith::vect<int, 3> contig_shape = box.shape();
            constexpr int contig_di = 1;
            const int contig_dj = contig_di * contig_shape[0];
            const int contig_dk = contig_dj * contig_shape[1];
            const int contig_np = contig_dk * contig_shape[2];
            assert(contig_np == np);
            const int contig_offset = extbox.size() - box.size();
            // TODO: optimize memory layout
            const auto expand_box = [=](void *const contig_ptr) {
              for (int k = 0; k < contig_shape[2]; ++k)
                for (int j = 0; j < contig_shape[1]; ++j)
                  for (int i = 0; i < contig_shape[0]; ++i)
                    // TODO: copy whole contiguous strip at once
                    memcpy(cactus_var_ptr +
                               (cactus_di * i + cactus_dj * j + cactus_dk * k) *
                                   vartypesize,
                           cactus_var_ptr + (contig_di * i + contig_dj * j +
                                             contig_dk * k + contig_offset) *
                                                vartypesize,
                           vartypesize);
              if (poison_undefined_values) {
                const size_t typesize =
                    size_t(CCTK_VarTypeSize(cgroup.vartype));
                // TODO: Use AnyTypeScalarRef for this?
                std::vector<char> poison(typesize);
                assert(cgroup.vartype == CCTK_VARIABLE_REAL ||
                       cgroup.vartype == CCTK_VARIABLE_INT ||
                       cgroup.vartype == CCTK_VARIABLE_COMPLEX);
                switch (cgroup.vartype) {
                case CCTK_VARIABLE_REAL: {
                  poison_value_t<CCTK_REAL> poison_value;
                  poison_value.set_to_poison(poison.data(), 1);
                } break;
                case CCTK_VARIABLE_INT: {
                  poison_value_t<CCTK_INT> poison_value;
                  poison_value.set_to_poison(poison.data(), 1);
                } break;
                case CCTK_VARIABLE_COMPLEX: {
                  poison_value_t<CCTK_COMPLEX> poison_value;
                  poison_value.set_to_poison(poison.data(), 1);
                } break;
                }
                for (int k = extbox.lo[2]; k < extbox.hi[2]; ++k) {
                  for (int j = extbox.lo[1]; j < extbox.hi[1]; ++j) {
                    for (int i = extbox.lo[0]; i < extbox.hi[0]; ++i) {
                      const Arith::vect<int, dim> I{i, j, k};
                      if (any(I < box.lo || I >= box.hi))
                        memcpy(cactus_var_ptr + (cactus_di * i + cactus_dj * j +
                                                 cactus_dk * k) *
                                                    vartypesize,
                               poison.data(), poison.size());
                    }
                  }
                }
              }
            };
            assert(cgroup.vartype == CCTK_VARIABLE_REAL ||
                   cgroup.vartype == CCTK_VARIABLE_INT ||
                   cgroup.vartype == CCTK_VARIABLE_COMPLEX);
            switch (cgroup.vartype) {
            case CCTK_VARIABLE_REAL:
              record_components.at(vi).loadChunk(
                  std::shared_ptr<CCTK_REAL>(
                      static_cast<CCTK_REAL *>(
                          cactus_ptr->data_at(contig_offset + cactus_np * vi)),
                      [=](CCTK_REAL *const ptr) {
                        expand_box(static_cast<void *>(ptr));
                      }),
                  start, count);
              break;
            case CCTK_VARIABLE_INT:
              record_components.at(vi).loadChunk(
                  std::shared_ptr<CCTK_INT>(
                      static_cast<CCTK_INT *>(
                          cactus_ptr->data_at(contig_offset + cactus_np * vi)),
                      [=](CCTK_INT *const ptr) {
                        expand_box(static_cast<void *>(ptr));
                      }),
                  start, count);
              break;
            case CCTK_VARIABLE_COMPLEX:
              record_components.at(vi).loadChunk(
                  std::shared_ptr<CCTK_COMPLEX>(
                      static_cast<CCTK_COMPLEX *>(
                          cactus_ptr->data_at(contig_offset + cactus_np * vi)),
                      [=](CCTK_COMPLEX *const ptr) {
                        expand_box(static_cast<void *>(ptr));
                      }),
                  start, count);
              break;
            }
          }
        } // for vi
      }
    } // for gi
  }

  if (io_verbose)
    CCTK_VINFO("InputOpenPMD: Performing all reads...");
  series->flush();

  if (io_verbose)
    CCTK_VINFO("InputOpenPMD: Post-processing data...");
  for (auto &task : tasks)
    std::move(task)();
  tasks.clear();
  scopes.clear();
  array_scopes.clear();

  if (io_verbose)
    CCTK_VINFO("InputOpenPMD: Closing iteration...");
  read_iter->close();

  if (io_verbose)
    CCTK_VINFO("InputOpenPMD: Deallocating objects...");
  read_iter.reset();
  series.reset();
  filename.reset();

  if (io_verbose)
    CCTK_VINFO("InputOpenPMD done.");

  if (io_verbose)
    timer.print();
}

////////////////////////////////////////////////////////////////////////////////

void carpetx_openpmd_t::OutputOpenPMD(const cGH *const cctkGH,
                                      const std::vector<bool> &output_group,
                                      const std::string &output_dir,
                                      const std::string &output_file,
                                      TimeLevelMode tl_mode,
                                      std::optional<slice_t> slice) {
  DECLARE_CCTK_ARGUMENTS;
  DECLARE_CCTK_PARAMETERS;

  // Set up timers
  static Timer timer("OutputOpenPMD");
  Interval interval(timer);

  if (std::count(output_group.begin(), output_group.end(), true) == 0)
    return;

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

  const int myproc = CCTK_MyProc(cctkGH);
  const int ioproc = 0;

  // At an unsynchronized subcycling checkpoint the coarse source bands hold
  // the in-progress coarse step (u(t_n) and the stage derivatives), which
  // exists nowhere else and must be serialized. Otherwise the on-disk format
  // is unchanged.
  const bool write_bands = !all_levels_synchronized();

  // Iteration attributes are set on EVERY rank (all inputs are replicated):
  // rank-asymmetric writes deadlock the HDF5 backend's collective metadata.

  // Write parameters
  {
    char *const data = IOUtil_GetAllParameters(cctkGH, 1 /*all*/);
    const std::string parameters(data);
    std::free(data);
    write_iter.setAttribute("AllParameters", parameters);
  }

  if (!slice) {
    const int ndims = Loop::dim;
    write_iter.setAttribute<std::int64_t>("numDims", ndims);
    const int npatches = ghext->num_patches();
    write_iter.setAttribute<std::int64_t>("numPatches", npatches);
    std::vector<std::string> patch_suffixes(npatches);
    for (const auto &patchdata : ghext->patches()) {
      const int patch = patchdata.patch.index();
      std::ostringstream buf;
      // String attributes cannot be empty; we thus always need to use a patch
      // suffix if (ghext->num_patches() > 1)
      buf << "_patch" << std::setw(2) << std::setfill('0') << patch;
      patch_suffixes.at(patch) = buf.str();
    }
    write_iter.setAttribute("patchSuffixes", patch_suffixes);
    for (const auto &patchdata : ghext->patches()) {
      const int patch = patchdata.patch.index();
      const int nlevels = patchdata.num_levels();
      write_iter.setAttribute<std::int64_t>(
          "numLevels" + patch_suffixes.at(patch), nlevels);
      std::vector<std::string> level_suffixes(nlevels);
      for (const auto &leveldata : patchdata.levels()) {
        const int level = leveldata.level.index();
        std::ostringstream buf;
        buf << patch_suffixes.at(patch) << "_lev" << std::setw(2)
            << std::setfill('0') << level;
        level_suffixes.at(level) = buf.str();
      }
      write_iter.setAttribute("levelSuffixes" + patch_suffixes.at(patch),
                              level_suffixes);
      for (const auto &leveldata : patchdata.levels()) {
        const int level = leveldata.level.index();
        const amrex::FabArrayBase &mfab = *leveldata.fab;
        const int nchunks = mfab.size();
        std::vector<std::int64_t> chunk_infos(2 * ndims * nchunks);
        for (int component = 0; component < nchunks; ++component) {
          const amrex::Box &box = mfab.box(component);
          for (int d = 0; d < ndims; ++d) {
            chunk_infos.at(2 * ndims * component + 0 * ndims + ndims - 1 - d) =
                box.smallEnd()[d];
            chunk_infos.at(2 * ndims * component + 1 * ndims + ndims - 1 - d) =
                box.bigEnd()[d] + 1;
          }
        }
        write_iter.setAttribute("chunkInfo" + level_suffixes.at(level),
                                chunk_infos);

        // Write per-level iteration as rational (numerator/denominator)
        write_iter.setAttribute(
            "iteration_num" + level_suffixes.at(level),
            static_cast<std::int64_t>(leveldata.iteration.num));
        write_iter.setAttribute(
            "iteration_den" + level_suffixes.at(level),
            static_cast<std::int64_t>(leveldata.iteration.den));
        // Write the per-level last-step iteration
        write_iter.setAttribute(
            "last_step_iteration" + level_suffixes.at(level),
            static_cast<std::int64_t>(leveldata.last_step_iteration));
      }
    }
  } // if !slice

  // First write grid functions in a loop over patches and levels

  for (const auto &patchdata : ghext->patches()) {
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
      // The domain is always vertex centred. The tensor components are
      // then staggered if necessary.
      box_t<int, 3> idomain{
          .lo = {ilo[0] - output_ghosts * nghosts[0],
                 ilo[1] - output_ghosts * nghosts[1],
                 ilo[2] - output_ghosts * nghosts[2]},
          .hi = {ihi[0] + output_ghosts * nghosts[0] + 1 + 1,
                 ihi[1] + output_ghosts * nghosts[1] + 1 + 1,
                 ihi[2] + output_ghosts * nghosts[2] + 1 + 1}};
      const Arith::vect<CCTK_REAL, 3> slice_x0{xlo[0], xlo[1], xlo[2]};
      const Arith::vect<CCTK_REAL, 3> slice_dx{dx[0], dx[1], dx[2]};
      if (slice) {
        const auto r =
            slice->restrict_box(idomain.lo, idomain.hi, slice_x0, slice_dx);
        if (!r)
          continue; // plane misses this (patch, level) entirely
        idomain.lo = r->first;
        idomain.hi = r->second;
      }
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
        const gi_t gi = groupdata.record.groupindex;
        if (output_group.at(gi.index())) {

          cGroup cgroup;
          const int ierr = CCTK_GroupData(gi.index(), &cgroup);
          assert(!ierr);
          assert(cgroup.vartype == CCTK_VARIABLE_REAL);
          assert(cgroup.dim == 3);

          if (io_verbose)
            CCTK_VINFO("Writing group %d %s...", gi.index(),
                       CCTK_FullGroupName(gi.index()));

          const int numvars = groupdata.record.numvars;

          const tl_range tls = tl_mode == TimeLevelMode::All
                                   ? groupdata.record.timelevels()
                                   : tl_range{tl_t{0}, tl_t{1}};
          for (const tl_t tl : tls) {
            const amrex::MultiFab &mfab = *groupdata.timelevel(tl).mfab;
            const amrex::IndexType &indextype = mfab.ixType();
            const Arith::vect<bool, 3> is_cell_centred{
                indextype.cellCentered(0), indextype.cellCentered(1),
                indextype.cellCentered(2)};

            const int num_local_components = mfab.local_size();

            // Create mesh

            const std::string meshname =
                make_meshname(gi, leveldata.level.patch().index(),
                              leveldata.level.index(), tl);
            if (io_verbose)
              CCTK_VINFO("Defining mesh %s...", meshname.c_str());
            assert(!write_iter.meshes.contains(meshname));
            openPMD::Mesh mesh = write_iter.meshes[meshname];

            mesh.setGeometry(openPMD::Mesh::Geometry::cartesian);
            mesh.setAxisLabels(
                reversed(std::vector<std::string>{"x", "y", "z"}));
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
            for (int vi = 0; vi < numvars; ++vi) {
              const std::string componentname = make_componentname(gi, vi);
              record_components.push_back(mesh[componentname]);
              auto &record_component = record_components.back();
              record_component.setPosition(
                  to_vector<double>(reversed(position)));
            }
            assert(int(record_components.size()) == numvars);

            // Write data

            if (io_verbose)
              CCTK_VINFO("Writing %d variables with %d components...", numvars,
                         num_local_components);

            for (int vi = 0; vi < numvars; ++vi)
              record_components.at(vi).resetDataset(dataset);

            // Loop over components (AMReX boxes)
            for (int local_component = 0;
                 local_component < num_local_components; ++local_component) {
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
              // Restrict this component to the thickness-1 slab, skipping
              // components that miss the plane. Membership is cell-canonical so
              // cell and vertex variables pick the same components for the
              // shared vertex-framed dataset.
              if (slice) {
                const amrex::Box cvalidbox = amrex::enclosedCells(validbox);
                const Arith::vect<int, 3> cval_lo{cvalidbox.smallEnd(0),
                                                  cvalidbox.smallEnd(1),
                                                  cvalidbox.smallEnd(2)};
                const Arith::vect<int, 3> cval_hi{cvalidbox.bigEnd(0) + 1,
                                                  cvalidbox.bigEnd(1) + 1,
                                                  cvalidbox.bigEnd(2) + 1};
                const auto r = slice->restrict_component(
                    box.lo, box.hi, cval_lo, cval_hi, slice_x0, slice_dx,
                    is_cell_centred);
                if (!r)
                  continue; // this component does not intersect the plane
                box.lo = r->first;
                box.hi = r->second;
              }

              Arith::vect<int, 3> start_vec = box.lo - idomain.lo;
              if (slice) {
                const int n = slice->normal_dir;
                // The slab must fall inside this component's exterior extent.
                assert(box.lo[n] >= extbox.lo[n] && box.hi[n] <= extbox.hi[n]);
                // idomain is vertex-framed, so box.lo - idomain.lo is wrong for
                // the thickness-1 slab; force the normal offset to 0.
                start_vec[n] = 0;
              }
              const openPMD::Offset start = to_vector(reversed(start_vec));
              const openPMD::Extent count = to_vector(reversed(box.shape()));
              const int np = box.size();
              assert(int(count.at(0) * count.at(1) * count.at(2)) == np);
              for (int d = 0; d < 3; ++d)
                assert(start.at(d) <
                       std::numeric_limits<std::remove_reference_t<decltype(
                               start.at(d))> >::max() /
                           2);
              for (int d = 0; d < 3; ++d)
                assert(start.at(d) + count.at(d) <= extent.at(d));

              const amrex::FArrayBox &fab = mfab[component];
              for (int vi = 0; vi < numvars; ++vi) {
                // A thin slab (box != extbox) takes the contiguous-copy path.
                if (box == extbox) {
                  const CCTK_REAL *const ptr = fab.dataPtr() + vi * np;
#if OPENPMDAPI_VERSION_GE(0, 15, 0)
                  record_components.at(vi).storeChunkRaw(ptr, start, count);
#else
                  record_components.at(vi).storeChunk(openPMD::shareRaw(ptr),
                                                      start, count);
#endif
                } else {
                  std::shared_ptr<CCTK_REAL> ptr(
                      new CCTK_REAL[np], std::default_delete<CCTK_REAL[]>());
                  extract_subbox(ptr.get(), fab.dataPtr() + vi * extbox.size(),
                                 extbox.lo, extbox.hi, box.lo, box.hi);
                  record_components.at(vi).storeChunk(std::move(ptr), start,
                                                      count);
                }
              } // for vi
            } // for local_component
          } // for tl

          if (write_bands && !slice) {
            const auto write_band = [&](const amrex::MultiFab *const band,
                                        const band_kind kind, const int stage) {
              if (!band || band->empty())
                return;
              assert(band->nGrowVect() == 0);
              const std::string band_tag = subcycling_band_tag(kind, stage);

              const amrex::IndexType &indextype = band->ixType();
              const Arith::vect<bool, 3> is_cell_centred{
                  indextype.cellCentered(0), indextype.cellCentered(1),
                  indextype.cellCentered(2)};

              const std::string meshname =
                  make_meshname(gi, leveldata.level.patch().index(),
                                leveldata.level.index(), tl_t{0}, band_tag);
              if (io_verbose)
                CCTK_VINFO("Defining band mesh %s...", meshname.c_str());
              assert(!write_iter.meshes.contains(meshname));
              openPMD::Mesh mesh = write_iter.meshes[meshname];

              mesh.setGeometry(openPMD::Mesh::Geometry::cartesian);
              mesh.setAxisLabels(
                  reversed(std::vector<std::string>{"x", "y", "z"}));
              mesh.setGridSpacing(to_vector<CCTK_REAL>(reversed(
                  fmap([](auto x, auto y) { return x / CCTK_REAL(y); },
                       rdomain.hi - rdomain.lo, idomain.shape() - 1))));
              mesh.setGridGlobalOffset(to_vector<double>(reversed(rdomain.lo)));
              mesh.setGridUnitSI(Unit::length);
              mesh.setTimeOffset(CCTK_REAL(0));

              const Arith::vect<double, 3> position =
                  fmap([](auto c) { return 0.5 * c; }, is_cell_centred);

              std::vector<openPMD::MeshRecordComponent> record_components;
              record_components.reserve(numvars);
              for (int vi = 0; vi < numvars; ++vi) {
                const std::string componentname = make_componentname(gi, vi);
                record_components.push_back(mesh[componentname]);
                auto &record_component = record_components.back();
                record_component.setPosition(
                    to_vector<double>(reversed(position)));
              }
              assert(int(record_components.size()) == numvars);
              for (int vi = 0; vi < numvars; ++vi)
                record_components.at(vi).resetDataset(dataset);

              const int num_local_components = band->local_size();
              for (int local_component = 0;
                   local_component < num_local_components; ++local_component) {
                const int component = band->IndexArray().at(local_component);

                const amrex::Box &fabbox =
                    band->fabbox(component); // zero ghost
                const box_t<int, 3> box{
                    .lo = {fabbox.smallEnd(0), fabbox.smallEnd(1),
                           fabbox.smallEnd(2)},
                    .hi = {fabbox.bigEnd(0) + 1, fabbox.bigEnd(1) + 1,
                           fabbox.bigEnd(2) + 1}};

                const openPMD::Offset start =
                    to_vector(reversed(box.lo - idomain.lo));
                const openPMD::Extent count = to_vector(reversed(box.shape()));
                const int np = box.size();
                assert(int(count.at(0) * count.at(1) * count.at(2)) == np);
                for (int d = 0; d < 3; ++d)
                  assert(start.at(d) + count.at(d) <= extent.at(d));

                const amrex::FArrayBox &fab = (*band)[component];
                for (int vi = 0; vi < numvars; ++vi) {
                  const CCTK_REAL *const ptr = fab.dataPtr() + vi * np;
#if OPENPMDAPI_VERSION_GE(0, 15, 0)
                  record_components.at(vi).storeChunkRaw(ptr, start, count);
#else
                  record_components.at(vi).storeChunk(openPMD::shareRaw(ptr),
                                                      start, count);
#endif
                } // for vi
              } // for local_component
            };

            for (int s = 0; s < max_num_rk_stages; ++s)
              write_band(
                  rk_source_band(leveldata.level, gi, band_kind::ks_source, s),
                  band_kind::ks_source, s);
            write_band(
                rk_source_band(leveldata.level, gi, band_kind::old_source),
                band_kind::old_source, -1);
            for (int f = 0; f < 2 * dim; ++f)
              write_band(rk_source_band(leveldata.level, gi,
                                        band_kind::flux_register, f),
                         band_kind::flux_register, f);
          } // if write_bands
        }
      } // for gi

    } // for leveldata
  } // for patchdata

  // Next write grid scalars and grid arrays.
  // Slices emit only GF data; a non-GF group in out_openpmd_2d_vars produces
  // nothing here.

  if (!slice) {
    const int numgroups = CCTK_NumGroups();
    for (int gi0 = 0; gi0 < numgroups; ++gi0) {
      const gi_t gi{gi0};
      if (output_group.at(gi.index())) {

        // Check group properties

        cGroup cgroup;
        const int ierr = CCTK_GroupData(gi.index(), &cgroup);
        assert(!ierr);
        if (cgroup.grouptype == CCTK_GF)
          continue;
        assert(cgroup.disttype == CCTK_DISTRIB_CONSTANT);
        assert(cgroup.dim >= 0);
        assert(cgroup.dim <= 3);

        if (io_verbose)
          CCTK_VINFO("Writing group %d %s...", gi.index(),
                     CCTK_FullGroupName(gi.index()));

        const auto &groupdata = ghext->record(gi).array();
        if (!groupdata.record.has_storage())
          continue;
        const int numvars = groupdata.record.numvars;
        const tl_t tl{0};

        // Determine grid structure

        using ivect = Arith::vect<int, dim>;

        const box_t<int, 3> idomain{.lo = ivect{0, 0, 0},
                                    .hi = ivect(groupdata.gsh)};

        // Create dataset

        const openPMD::Datatype datatype = [](int varType) {
          switch (varType) {
          case CCTK_VARIABLE_REAL:
            return openPMD::determineDatatype<CCTK_REAL>();
          case CCTK_VARIABLE_INT:
            return openPMD::determineDatatype<CCTK_INT>();
          case CCTK_VARIABLE_COMPLEX:
            return openPMD::determineDatatype<CCTK_COMPLEX>();
          default:
            assert(0 && "Unexpected varType");
          }
        }(cgroup.vartype);
        const openPMD::Extent extent = to_vector(reversed(idomain.shape()));
        const openPMD::Dataset dataset(datatype, extent);

        // Create mesh

        const std::string meshname = make_meshname(gi, -1, -1);
        openPMD::Mesh mesh = write_iter.meshes[meshname];

        mesh.setTimeOffset(CCTK_REAL(0)); // TODO: check interface.ccl

        // Define tensor components

        // TODO: Set component names according to the tensor type
        std::vector<openPMD::MeshRecordComponent> record_components;
        record_components.reserve(numvars);
        for (int vi = 0; vi < numvars; ++vi) {
          const std::string componentname = make_componentname(gi, vi);
          record_components.push_back(mesh[componentname]);
        }
        assert(int(record_components.size()) == numvars);

        // Write data

        if (io_verbose)
          CCTK_VINFO("Writing %d variables...", numvars);

        for (int vi = 0; vi < numvars; ++vi)
          record_components.at(vi).resetDataset(dataset);

        // exterior (with ghosts)
        for (int d = 0; d < dim; ++d)
          assert(groupdata.lsh[d] == groupdata.gsh[d]);
        assert(all(Arith::vect<int, dim>(groupdata.lsh) ==
                   Arith::vect<int, dim>(groupdata.gsh)));
        const box_t<int, 3> extbox{.lo = ivect{0, 0, 0},
                                   .hi = ivect(groupdata.lsh)};
        // interior (without ghosts)
        const box_t<int, 3> intbox{.lo = ivect(groupdata.nghostzones),
                                   .hi = ivect(groupdata.lsh) -
                                         ivect(groupdata.nghostzones)};
        // It seems that openPMD assumes that chunks do not have ghost zones
        assert(!output_ghosts);
        const box_t<int, 3> &box = output_ghosts ? extbox : intbox;

        const openPMD::Offset start = to_vector(reversed(box.lo - idomain.lo));
        const openPMD::Extent count = to_vector(reversed(box.shape()));
        const int np = box.size();
        assert(int(count.at(0) * count.at(1) * count.at(2)) == np);
        for (int d = 0; d < 3; ++d)
          assert(start.at(d) <
                 std::numeric_limits<
                     std::remove_reference_t<decltype(start.at(d))> >::max() /
                     2);
        for (int d = 0; d < 3; ++d)
          assert(start.at(d) + count.at(d) <= extent.at(d));

        const Arith::vect<int, 3> cactus_shape = extbox.shape();
        constexpr int cactus_di = 1;
        const int cactus_dj = cactus_di * cactus_shape[0];
        const int cactus_dk = cactus_dj * cactus_shape[1];
        const int cactus_np = cactus_dk * cactus_shape[2];
        assert(cactus_di > 0);
        assert(cactus_dj > 0);
        assert(cactus_dk > 0);
        assert(cactus_np > 0);
        assert(int(groupdata.timelevel(tl).data.size()) == numvars * cactus_np);
        if (myproc == ioproc) {
          for (int vi = 0; vi < numvars; ++vi) {
            const void *const var_ptr =
                groupdata.timelevel(tl).data.data_at(vi * cactus_np);
            if (output_ghosts || intbox == extbox) {
#if !OPENPMDAPI_VERSION_GE(0, 15, 0)
#define storeChunkRaw(ptr, start, count)                                       \
  storeChunk(openPMD::shareRaw(ptr), start, count)
#endif
              switch (cgroup.vartype) {
              case CCTK_VARIABLE_REAL:
                record_components.at(vi).storeChunkRaw(
                    static_cast<CCTK_REAL const *>(var_ptr), start, count);
                break;
              case CCTK_VARIABLE_INT:
                record_components.at(vi).storeChunkRaw(
                    static_cast<CCTK_INT const *>(var_ptr), start, count);
                break;
              case CCTK_VARIABLE_COMPLEX:
                record_components.at(vi).storeChunkRaw(
                    static_cast<CCTK_COMPLEX const *>(var_ptr), start, count);
                break;
              default:
                assert(0 && "Unexpected variable type");
              }
            } else {
              auto cactus_ptr = &groupdata.timelevel(tl).data;
              const Arith::vect<int, 3> contig_shape = box.shape();
              constexpr int contig_di = 1;
              const int contig_dj = contig_di * contig_shape[0];
              const int contig_dk = contig_dj * contig_shape[1];
              const int contig_np = contig_dk * contig_shape[2];
              assert(contig_np == np);
              auto contig_ptr =
                  new GHExt::GlobalData::AnyTypeVector(cgroup.vartype, np);
              for (int k = 0; k < contig_shape[2]; ++k)
                for (int j = 0; j < contig_shape[1]; ++j)
                  for (int i = 0; i < contig_shape[0]; ++i)
                    // TODO: copy whole contiguous strip at once
                    memcpy(contig_ptr->data_at(contig_di * i + contig_dj * j +
                                               contig_dk * k),
                           cactus_ptr->data_at(cactus_di * i + cactus_dj * j +
                                               cactus_dk * k + vi * cactus_np),
                           CCTK_VarTypeSize(cgroup.vartype));
              switch (cgroup.vartype) {
              case CCTK_VARIABLE_REAL:
                record_components.at(vi).storeChunk(
                    std::shared_ptr<CCTK_REAL>(
                        static_cast<CCTK_REAL *>(contig_ptr->data_at(0)),
                        [=](CCTK_REAL *) { delete contig_ptr; }),
                    start, count);
                break;
              case CCTK_VARIABLE_INT:
                record_components.at(vi).storeChunk(
                    std::shared_ptr<CCTK_INT>(
                        static_cast<CCTK_INT *>(contig_ptr->data_at(0)),
                        [=](CCTK_INT *const) { delete contig_ptr; }),
                    start, count);
                break;
              case CCTK_VARIABLE_COMPLEX:
                record_components.at(vi).storeChunk(
                    std::shared_ptr<CCTK_COMPLEX>(
                        static_cast<CCTK_COMPLEX *>(contig_ptr->data_at(0)),
                        [=](CCTK_COMPLEX *const) { delete contig_ptr; }),
                    start, count);
                break;
              default:
                assert(0 && "Unexpected variable type");
              }
            }
          } // for vi
        }
      }
    }
  }

  if (slice && CCTK_MyProc(nullptr) == 0) {
    output_file_description_t ofd;
    ofd.filename = *filename;
    ofd.description = "2D CarpetX openPMD slice output";
    ofd.writer_thorn = CCTK_THORNSTRING;
    for (int gi = 0; gi < CCTK_NumGroups(); ++gi) {
      if (!output_group.at(gi) || CCTK_GroupTypeI(gi) != CCTK_GF)
        continue;
      const GHExt::GroupRecord &record = ghext->record(gi_t{gi});
      for (const vi_t vi : record.variables())
        ofd.variables.push_back(CCTK_FullVarName(record.var(vi).index()));
    }
    ofd.iterations = {cctk_iteration};
    for (const int d : slice->inplane_dirs())
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
void ShutdownOpenPMD() {}
} // namespace CarpetX

#endif // #ifdef HAVE_CAPABILITY_OPENPMD
