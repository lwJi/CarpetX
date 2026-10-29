#include "io/io_adios2.hxx"

#include "hierarchy/driver.hxx"
#include "io/io.hxx"
#include "io/selection.hxx"
#include "util/refuse.hxx"
#include "util/var_type.hxx"
#include <carpetx.hxx>

#include <div.hxx>

#include <cctk.h>
#include <cctk_Arguments.h>
#include <cctk_Parameters.h>

#include <source_location>

#ifdef HAVE_CAPABILITY_ADIOS2

#if !defined ADIOS2_USE_MPI || !ADIOS2_USE_MPI
#error                                                                         \
    "CarpetX requires an MPI-enabled ADIOS2 library. Please compile ADIOS2 with MPI support and enable its use by defining ADIOS2_USE_MPI."
#endif
#include <adios2.h>

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
#include <cstring>
#include <ios>
#include <iostream>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace CarpetX {

struct carpetx_adios2_t {
  carpetx_adios2_t() = default;

  carpetx_adios2_t(const carpetx_adios2_t &) = delete;
  carpetx_adios2_t(carpetx_adios2_t &&) = default;
  carpetx_adios2_t &operator=(const carpetx_adios2_t &) = delete;
  carpetx_adios2_t &operator=(carpetx_adios2_t &&) = default;

  static std::optional<carpetx_adios2_t> self;

  ////////////////////////////////////////////////////////////////////////////////

  static constexpr bool io_verbose = true;

  static constexpr bool combine_components = true;
  static constexpr bool combine_via_span = true;

  ////////////////////////////////////////////////////////////////////////////////

  template <typename T, std::size_t D> struct box_t {
    std::array<T, D> lo, hi;
    constexpr std::array<T, D> shape() const {
      std::array<T, D> sh;
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
  };

  template <typename T, typename I, std::size_t D> struct level_layout_t {
    box_t<T, D> rdomain;
    std::array<bool, D> is_cell_centred;
    box_t<I, D> idomain;
    constexpr std::array<T, D> rcoord(const std::array<I, D> &icoord) const {
      std::array<T, D> r;
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
    std::array<std::vector<I>, 2> offsets_sizes() const {
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

  template <typename T, typename I, std::size_t D> struct patch_layout_t {
    box_t<T, D> rdomain;
    std::vector<level_layout_t<T, I, D> > levels;
  };

  template <typename T, typename I, std::size_t D> struct grid_structure_t {
    std::vector<patch_layout_t<T, I, D> > patches;
  };

  ////////////////////////////////////////////////////////////////////////////////

  static std::string make_varname(const GHExt::GroupRecord &group,
                                  const vi_t vi, const level_t level,
                                  const int component = -1) {
    std::ostringstream buf;
    buf << CCTK_FullVarName(group.var(vi).index());
    buf << ".m" << std::setw(4) << std::setfill('0') << level.patch().index();
    buf << ".rl" << std::setw(2) << std::setfill('0') << level.index();
    if (component >= 0)
      buf << ".c" << std::setw(8) << std::setfill('0') << component;
    return buf.str();
  }

  ////////////////////////////////////////////////////////////////////////////////

  adios2::ADIOS adios;
  adios2::IO io;
  adios2::Engine engine;

  void OutputADIOS2(const cGH *const cctkGH, const group_selection_t &selection,
                    const std::string &output_dir,
                    const std::string &output_file);
  ~carpetx_adios2_t();
};

////////////////////////////////////////////////////////////////////////////////

void OutputADIOS2(const cGH *const cctkGH) {
  const operation_scope_t op("CarpetX::OutputADIOS2");
  DECLARE_CCTK_PARAMETERS;

  const group_selection_t selection = group_selection_t::select_groups(
      "CarpetX::out_adios2_vars", out_adios2_vars);
  if (selection.empty())
    return;
  selection.log("ADIOS2", group_kind::grid_function);

  for (const GHExt::GroupRecord &group :
       selection.records(group_kind::grid_function))
    if (group.var_type != var_type::real)
      refuse(std::source_location::current(),
             "ADIOS2 writes only real grid functions; group %s is %s",
             group.groupname.c_str(),
             CCTK_VarTypeName(cctk_code(group.var_type)));

  if (!carpetx_adios2_t::self)
    carpetx_adios2_t::self = std::make_optional<carpetx_adios2_t>();
  carpetx_adios2_t::self->OutputADIOS2(cctkGH, selection, out_dir,
                                       get_simulation_name());
}

void ShutdownADIOS2() { carpetx_adios2_t::self.reset(); }

////////////////////////////////////////////////////////////////////////////////

std::optional<carpetx_adios2_t> carpetx_adios2_t::self;

void carpetx_adios2_t::OutputADIOS2(const cGH *const cctkGH,
                                    const group_selection_t &selection,
                                    const std::string &output_dir,
                                    const std::string &output_file) {
  DECLARE_CCTK_ARGUMENTS;
  DECLARE_CCTK_PARAMETERS;

  // Set up timers
  static Timer timer("OutputADIOS2");
  Interval interval(timer);

  if (io_verbose)
    CCTK_VINFO("OutputADIOS2...");

  try {

    const int nprocs = CCTK_nProcs(cctkGH);

    const GHExt &ext = driver();
    if (!io) {

      if (io_verbose)
        CCTK_VINFO("  Creating ADIOS object...");
      // "Fortran" enables column-major mode
      adios = adios2::ADIOS("", MPI_COMM_WORLD, "Fortran");

      if (io_verbose)
        CCTK_VINFO("  Creating IO object...");
      io = adios.DeclareIO("IO");
      const int num_aggregators = [&]() {
        if (CCTK_EQUALS(out_mode, "proc"))
          return nprocs;
        if (CCTK_EQUALS(out_mode, "np"))
          return div_ceil(nprocs, out_proc_every);
        if (CCTK_EQUALS(out_mode, "onefile"))
          return 1;
        refuse(std::source_location::current(),
               "IO::out_mode is \"%s\", which this writer does not know",
               out_mode);
      }();
      assert(num_aggregators > 0);
      io.SetParameters({
          {"NumAggregators", std::to_string(num_aggregators)},
          // {"Threads", std::to_string(omp_get_max_threads())},
      });

      if (io_verbose)
        CCTK_VINFO("  Creating engine...");
      // Create output engine
      std::ostringstream buf;
      const int mode = 0755;
      static std::once_flag create_directory;
      call_once(create_directory, [&]() {
        const int ierr = CCTK_CreateDirectory(mode, output_dir.c_str());
        assert(ierr >= 0);
      });
      buf << output_dir << "/" << output_file << ".bp5";
      const std::string filename = buf.str();
      engine = io.Open(filename, adios2::Mode::Write);

#ifdef ADIOS2_HAVE_BLOSC2
      const adios2::Operator compressor =
          adios.DefineOperator("Blosc2Compressor", adios2::ops::LosslessBlosc);
      // Use a high compression rate and a byteshuffle filter
      const adios2::Params compressor_options{
          {adios2::ops::blosc::key::clevel,
           adios2::ops::blosc::value::clevel_9},
          {adios2::ops::blosc::key::doshuffle,
           adios2::ops::blosc::value::doshuffle_shuffle},
      };
#endif

      if (io_verbose)
        CCTK_VINFO("  Defining variables...");

      for (const auto &patchdata : ext.patches()) {

        for (const auto &leveldata : patchdata.levels()) {

          for (const auto &groupdata : leveldata.groups()) {
            if (selection.contains(groupdata.record.groupindex)) {

              const tl_t tl{0};
              const amrex::MultiFab &mfab = *groupdata.timelevel(tl).mfab;
              const int num_local_components = mfab.local_size();

              for (const vi_t vi : groupdata.record.variables()) {

                if (!combine_components) {

                  for (int local_component = 0;
                       local_component < num_local_components;
                       ++local_component) {
                    const int component = mfab.IndexArray().at(local_component);
                    const std::string varname = make_varname(
                        groupdata.record, vi, leveldata.level, component);
                    if (io_verbose)
                      CCTK_VINFO("      Defining variable %s...",
                                 varname.c_str());
                    adios2::Variable<CCTK_REAL> var =
                        io.DefineVariable<CCTK_REAL>(varname, {}, {},
                                                     {1, 1, 1});
#ifdef ADIOS2_HAVE_BLOSC2
                    var.AddOperation(compressor, compressor_options);
#endif
                  } // for local_component

                } else { // if combine_components

                  const std::string varname =
                      make_varname(groupdata.record, vi, leveldata.level);
                  if (io_verbose)
                    CCTK_VINFO("      Defining variable %s...",
                               varname.c_str());
                  io.DefineVariable<CCTK_REAL>(varname, {}, {}, {1});

                } // if combine_components

              } // for vi
            }
          }

        } // for leveldata
      } // for patchdata

    } // if !adios2_state

    if (io_verbose)
      CCTK_VINFO("  Beginning step...");
    engine.BeginStep();

    std::vector<patch_layout_t<CCTK_REAL, int, 3> > patches(ext.num_patches());
    for (const auto &patchdata : ext.patches()) {

      std::vector<level_layout_t<CCTK_REAL, int, 3> > levels(
          patchdata.num_levels());
      for (const auto &leveldata : patchdata.levels()) {
        if (io_verbose)
          CCTK_VINFO("  Writing patch %d level %d...", patchdata.patch.index(),
                     leveldata.level.index());

        for (const auto &groupdata : leveldata.groups()) {
          if (selection.contains(groupdata.record.groupindex)) {
            if (io_verbose)
              CCTK_VINFO("    Writing group %s...",
                         groupdata.record.groupname.c_str());

            const tl_t tl{0};
            const amrex::MultiFab &mfab = *groupdata.timelevel(tl).mfab;
            const amrex::IndexType &indextype = mfab.ixType();

            const amrex::Geometry &geom =
                patchdata.amrcore->Geom(leveldata.level.index());
            const amrex::Real *const xlo = geom.ProbLo();
            const amrex::Real *const xhi = geom.ProbHi();
            const box_t<CCTK_REAL, 3> rdomain{.lo = {xlo[0], xlo[1], xlo[2]},
                                              .hi = {xhi[0], xhi[1], xhi[2]}};
            const amrex::Box &dom = geom.Domain();
            const box_t<int, 3> idomain{
                .lo = {dom.smallEnd(0), dom.smallEnd(1), dom.smallEnd(2)},
                .hi = {dom.bigEnd(0) + 1, dom.bigEnd(1) + 1,
                       dom.bigEnd(2) + 1}};
            const std::array<bool, 3> is_cell_centred{
                indextype.cellCentered(0), indextype.cellCentered(1),
                indextype.cellCentered(2)};

            const int num_local_components = mfab.local_size();
            std::vector<box_t<int, 3> > grids(num_local_components);
            for (int local_component = 0;
                 local_component < num_local_components; ++local_component) {
              const int component = mfab.IndexArray().at(local_component);
              const amrex::Box &fabbox = mfab.fabbox(component); // exterior
              grids.at(local_component) = box_t<int, 3>{
                  .lo = {fabbox.smallEnd(0), fabbox.smallEnd(1),
                         fabbox.smallEnd(2)},
                  .hi = {fabbox.bigEnd(0) + 1, fabbox.bigEnd(1) + 1,
                         fabbox.bigEnd(2) + 1}};
            } // for local_component
            levels.at(leveldata.level.index()) =
                level_layout_t<CCTK_REAL, int, 3>{.rdomain = rdomain,
                                                  .is_cell_centred =
                                                      is_cell_centred,
                                                  .idomain = idomain,
                                                  .grids = std::move(grids)};
            const level_layout_t<CCTK_REAL, int, 3> &level =
                levels.at(leveldata.level.index());
            const auto offsets_sizes = level.offsets_sizes();
            const std::vector<int> &offsets = offsets_sizes[0];
            const std::vector<int> &sizes = offsets_sizes[1];

            for (const vi_t vi : groupdata.record.variables()) {
              if (!combine_components) {

                for (int local_component = 0;
                     local_component < num_local_components;
                     ++local_component) {
                  const int component = mfab.IndexArray().at(local_component);

                  adios2::Dims lsh(3);
                  for (int d = 0; d < 3; ++d)
                    lsh.at(d) = level.grids.at(local_component).shape()[d];
                  const int np = level.grids.at(local_component).size();
                  const amrex::FArrayBox &fab = mfab[component];

                  const std::string varname = make_varname(
                      groupdata.record, vi, leveldata.level, component);
                  if (io_verbose)
                    CCTK_VINFO("      Writing variable %s...", varname.c_str());
                  adios2::Variable<CCTK_REAL> var =
                      io.InquireVariable<CCTK_REAL>(varname);
                  assert(var);
                  var.SetSelection({{}, lsh});
                  const CCTK_REAL *const ptr = fab.dataPtr() + vi.index() * np;
                  assert(ptr);
                  engine.Put(var, ptr);
                } // for local_component

              } else { // if combine_components

                const std::string varname =
                    make_varname(groupdata.record, vi, leveldata.level);
                if (io_verbose)
                  CCTK_VINFO("      Writing variable %s...", varname.c_str());

                const size_t total_np = offsets.back();
                adios2::Variable<CCTK_REAL> var =
                    io.InquireVariable<CCTK_REAL>(varname);
                var.SetSelection({{}, {total_np}});

                if (!combine_via_span) {

                  std::vector<CCTK_REAL> alldata(offsets.back());

                  for (int local_component = 0;
                       local_component < num_local_components;
                       ++local_component) {
                    const int component = mfab.IndexArray().at(local_component);

                    const size_t np = sizes.at(local_component);
                    const amrex::FArrayBox &fab = mfab[component];
                    const CCTK_REAL *const ptr =
                        fab.dataPtr() + vi.index() * np;
                    assert(ptr);

                    std::memcpy(alldata.data() + offsets.at(local_component),
                                ptr, sizeof(CCTK_REAL) * np);

                  } // for local_component

                  engine.Put(var, alldata.data());

                } else { // if combine_via_span

                  const adios2::Variable<CCTK_REAL>::Span span =
                      engine.Put(var);

                  for (int local_component = 0;
                       local_component < num_local_components;
                       ++local_component) {
                    const int component = mfab.IndexArray().at(local_component);
                    const size_t np = sizes.at(local_component);
                    const amrex::FArrayBox &fab = mfab[component];
                    const CCTK_REAL *const ptr =
                        fab.dataPtr() + vi.index() * np;
                    assert(ptr);

                    std::memcpy(span.data() + offsets.at(local_component), ptr,
                                sizeof(CCTK_REAL) * np);
                  } // for local_component

                } // if combine_via_span

              } // if combine_components

            } // for vi
          }
        }

      } // for leveldata
      assert(!levels.empty());
      const auto rdomain = levels.front().rdomain;
      patches.at(patchdata.patch.index()) = patch_layout_t<CCTK_REAL, int, 3>{
          .rdomain = rdomain, .levels = std::move(levels)};
    } // for patchdata
    assert(!patches.empty());
    const grid_structure_t<CCTK_REAL, int, 3> grid_structure{
        .patches = std::move(patches)};

    if (io_verbose)
      CCTK_VINFO("  Performing puts...");
    engine.PerformPuts();

    if (io_verbose)
      CCTK_VINFO("  Ending step...");
    engine.EndStep();

  } catch (std::invalid_argument &e) {
    std::cerr << "Invalid argument exception: " << e.what() << "\n";
    CCTK_Abort(nullptr, 1);
  } catch (std::ios_base::failure &e) {
    std::cerr << "IO System base failure exception: " << e.what() << "\n";
    CCTK_Abort(nullptr, 1);
  } catch (std::exception &e) {
    std::cerr << "Exception: " << e.what() << "\n";
    CCTK_Abort(nullptr, 1);
  }

  if (io_verbose)
    CCTK_VINFO("OutputADIOS2 done.");

  if (io_verbose)
    timer.print();
}

carpetx_adios2_t::~carpetx_adios2_t() {
  if (engine) {
    if (io_verbose)
      CCTK_VINFO("ADIOS2: Closing engine...");
    engine.Close();
  }
}

} // namespace CarpetX

#else

namespace CarpetX {
void OutputADIOS2(const cGH *) {
  const operation_scope_t op("CarpetX::OutputADIOS2");
  DECLARE_CCTK_PARAMETERS;
  if (group_selection_t::select_groups("CarpetX::out_adios2_vars",
                                       out_adios2_vars)
          .empty())
    return;
  refuse(std::source_location::current(),
         "CarpetX::out_adios2_vars is set, but this build has no ADIOS2");
}

void ShutdownADIOS2() {}
} // namespace CarpetX

#endif // #ifdef HAVE_CAPABILITY_ADIOS2
