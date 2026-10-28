#include "io/checkpoint.hxx"
#include "io/io_openpmd_names.hxx"

#include <CactusBase/IOUtil/src/ioutil_CheckpointRecovery.h>
#include <cctk.h>
#include <util_Network.h>

#ifdef HAVE_CAPABILITY_openPMD_api

#include <openPMD/openPMD.hpp>

#include <AMReX_Box.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_Geometry.H>
#include <AMReX_IndexType.H>
#include <AMReX_MultiFab.H>

#include <mpi.h>

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iomanip>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace CarpetX {

namespace {

constexpr bool io_verbose = true;

static_assert(iterationEncoding == openPMD::IterationEncoding::fileBased);

template <class... Fs> struct overloaded : Fs... {
  using Fs::operator()...;
};
template <class... Fs> overloaded(Fs...) -> overloaded<Fs...>;

using index3 = std::array<int, 3>;

index3 difference(const index3 &x, const index3 &y) {
  return {x[0] - y[0], x[1] - y[1], x[2] - y[2]};
}

struct ibox_t {
  index3 lo, hi;
  index3 shape() const { return difference(hi, lo); }
  int size() const {
    const index3 sh = shape();
    return sh[0] * sh[1] * sh[2];
  }
  friend bool operator==(const ibox_t &, const ibox_t &) = default;
};

ibox_t make_ibox(const amrex::Box &box) {
  return {{box.smallEnd(0), box.smallEnd(1), box.smallEnd(2)},
          {box.bigEnd(0) + 1, box.bigEnd(1) + 1, box.bigEnd(2) + 1}};
}

template <typename T, typename V> std::vector<T> reversed(const V &v) {
  assert(std::size(v) == 3);
  return {T(v[2]), T(v[1]), T(v[0])};
}

struct level_layout_t {
  std::array<CCTK_REAL, 3> xlo, xhi;
  ibox_t idomain;
};

level_layout_t level_layout(const amrex::Geometry &geom) {
  const amrex::Box &domain = geom.Domain();
  level_layout_t layout;
  for (int d = 0; d < 3; ++d) {
    layout.xlo[d] = geom.ProbLo(d);
    layout.xhi[d] = geom.ProbHi(d);
    layout.idomain.lo[d] = domain.smallEnd(d);
    layout.idomain.hi[d] = domain.bigEnd(d) + 1 + 1;
  }
  return layout;
}

level_layout_t layout_of(const checkpoint_record_t &record,
                         const level_t level) {
  return level_layout(
      record.levels.at(level.patch().index()).at(level.index()).geom);
}

std::string series_name(const std::string &dir, const std::string &file,
                        const openPMD::Format format) {
  std::ostringstream buf;
  buf << dir << "/" << file << ".it%08T" << openPMD::suffix(format);
  return buf.str();
}

std::string mesh_name(const buffer_id_t &id) {
  if (const std::string *const tag = std::get_if<std::string>(&id.which))
    return make_meshname(id.group, id.level, tl_t{0}, *tag);
  return make_meshname(id.group, id.level, std::get<tl_t>(id.which));
}

std::string patch_suffix(const int patch) {
  std::ostringstream buf;
  buf << "_patch" << std::setw(2) << std::setfill('0') << patch;
  return buf.str();
}

std::string level_suffix(const std::string &patch_suffix, const int level) {
  std::ostringstream buf;
  buf << patch_suffix << "_lev" << std::setw(2) << std::setfill('0') << level;
  return buf.str();
}

void assert_inside(const openPMD::Offset &start, const openPMD::Extent &count,
                   const openPMD::Extent &extent) {
  for (int d = 0; d < 3; ++d)
    assert(start.at(d) + count.at(d) <= extent.at(d));
}

template <typename T>
void store_raw(openPMD::MeshRecordComponent &component, const T *const ptr,
               const openPMD::Offset &start, const openPMD::Extent &count) {
#if OPENPMDAPI_VERSION_GE(0, 15, 0)
  component.storeChunkRaw(ptr, start, count);
#else
  component.storeChunk(openPMD::shareRaw(ptr), start, count);
#endif
}

template <typename T>
void load_raw(openPMD::MeshRecordComponent &component, T *const ptr,
              const openPMD::Offset &start, const openPMD::Extent &count) {
#if OPENPMDAPI_VERSION_GE(0, 15, 0)
  component.loadChunkRaw(ptr, start, count);
#else
  component.loadChunk(openPMD::shareRaw(ptr), start, count);
#endif
}

openPMD::Datatype array_datatype(const int vartype) {
  switch (vartype) {
  case CCTK_VARIABLE_REAL:
    return openPMD::determineDatatype<CCTK_REAL>();
  case CCTK_VARIABLE_INT:
    return openPMD::determineDatatype<CCTK_INT>();
  case CCTK_VARIABLE_COMPLEX:
    return openPMD::determineDatatype<CCTK_COMPLEX>();
  }
  CCTK_VERROR("Unexpected variable type %d", vartype);
}

void store_array(openPMD::MeshRecordComponent &component, const int vartype,
                 const void *const ptr, const openPMD::Offset &start,
                 const openPMD::Extent &count) {
  switch (vartype) {
  case CCTK_VARIABLE_REAL:
    store_raw(component, static_cast<const CCTK_REAL *>(ptr), start, count);
    return;
  case CCTK_VARIABLE_INT:
    store_raw(component, static_cast<const CCTK_INT *>(ptr), start, count);
    return;
  case CCTK_VARIABLE_COMPLEX:
    store_raw(component, static_cast<const CCTK_COMPLEX *>(ptr), start, count);
    return;
  }
  CCTK_VERROR("Unexpected variable type %d", vartype);
}

void load_array(openPMD::MeshRecordComponent &component, const int vartype,
                void *const ptr, const openPMD::Offset &start,
                const openPMD::Extent &count) {
  switch (vartype) {
  case CCTK_VARIABLE_REAL:
    load_raw(component, static_cast<CCTK_REAL *>(ptr), start, count);
    return;
  case CCTK_VARIABLE_INT:
    load_raw(component, static_cast<CCTK_INT *>(ptr), start, count);
    return;
  case CCTK_VARIABLE_COMPLEX:
    load_raw(component, static_cast<CCTK_COMPLEX *>(ptr), start, count);
    return;
  }
  CCTK_VERROR("Unexpected variable type %d", vartype);
}

void extract_interior(CCTK_REAL *const dst, const CCTK_REAL *const src,
                      const ibox_t &extbox, const ibox_t &intbox) {
  const index3 ext_shape = extbox.shape();
  const index3 int_shape = intbox.shape();
  const index3 offset = difference(intbox.lo, extbox.lo);
  const int sdj = ext_shape[0];
  const int sdk = sdj * ext_shape[1];
  const CCTK_REAL *const src0 =
      src + offset[0] + sdj * offset[1] + sdk * offset[2];
  const int ddj = int_shape[0];
  const int ddk = ddj * int_shape[1];
  for (int k = 0; k < int_shape[2]; ++k)
    for (int j = 0; j < int_shape[1]; ++j)
      for (int i = 0; i < int_shape[0]; ++i)
        dst[i + ddj * j + ddk * k] = src0[i + sdj * j + sdk * k];
}

std::size_t num_points(const std::vector<int> &shape) {
  std::size_t np = 1;
  for (const int n : shape)
    np *= n;
  return np;
}

std::vector<openPMD::MeshRecordComponent>
read_components(const openPMD::Mesh &mesh, const std::string &meshname,
                const buffer_id_t &id) {
  std::vector<openPMD::MeshRecordComponent> components;
  components.reserve(id.vars.size());
  for (const std::string &var : id.vars) {
    const std::string name = make_componentname(var);
    if (!mesh.count(name))
      CCTK_VERROR("The openPMD mesh \"%s\" has no component \"%s\"",
                  meshname.c_str(), name.c_str());
    components.push_back(mesh.at(name));
  }
  return components;
}

void write_grid(openPMD::Iteration &write_iter,
                const checkpoint_record_t &record, const buffer_id_t &id,
                const grid_buffer_t &grid) {
  const level_layout_t layout = layout_of(record, *id.level);
  const index3 shape = layout.idomain.shape();

  const std::string meshname = mesh_name(id);
  if (io_verbose)
    CCTK_VINFO("Defining mesh %s...", meshname.c_str());
  assert(!write_iter.meshes.contains(meshname));
  openPMD::Mesh mesh = write_iter.meshes[meshname];

  mesh.setGeometry(openPMD::Mesh::Geometry::cartesian);
  mesh.setAxisLabels(std::vector<std::string>{"z", "y", "x"});
  std::vector<CCTK_REAL> spacing(3);
  for (int d = 0; d < 3; ++d)
    spacing.at(2 - d) =
        (layout.xhi[d] - layout.xlo[d]) / CCTK_REAL(shape[d] - 1);
  mesh.setGridSpacing(spacing);
  mesh.setGridGlobalOffset(reversed<double>(layout.xlo));
  mesh.setGridUnitSI(Unit::length);
  mesh.setTimeOffset(CCTK_REAL(0));

  const amrex::MultiFab &mfab = *grid.mfab;
  const amrex::IndexType &indextype = mfab.ixType();
  const std::vector<double> position{0.5 * indextype.cellCentered(2),
                                     0.5 * indextype.cellCentered(1),
                                     0.5 * indextype.cellCentered(0)};

  std::vector<openPMD::MeshRecordComponent> components;
  components.reserve(id.vars.size());
  for (const std::string &var : id.vars) {
    components.push_back(mesh[make_componentname(var)]);
    components.back().setPosition(position);
  }

  const openPMD::Dataset dataset(openPMD::determineDatatype<CCTK_REAL>(),
                                 reversed<std::uint64_t>(shape));
  for (openPMD::MeshRecordComponent &component : components)
    component.resetDataset(dataset);

  for (int local = 0; local < mfab.local_size(); ++local) {
    const int c = mfab.IndexArray().at(local);
    const ibox_t extbox = make_ibox(mfab.fabbox(c));
    const ibox_t intbox = make_ibox(mfab.box(c));
    const openPMD::Offset start =
        reversed<std::uint64_t>(difference(intbox.lo, layout.idomain.lo));
    const openPMD::Extent count = reversed<std::uint64_t>(intbox.shape());
    assert_inside(start, count, dataset.extent);
    const int np = intbox.size();
    const amrex::FArrayBox &fab = mfab[c];
    for (std::size_t vi = 0; vi < components.size(); ++vi) {
      if (intbox == extbox) {
        store_raw(components.at(vi), fab.dataPtr() + vi * np, start, count);
      } else {
        std::shared_ptr<CCTK_REAL> ptr(new CCTK_REAL[np],
                                       std::default_delete<CCTK_REAL[]>());
        extract_interior(ptr.get(), fab.dataPtr() + vi * extbox.size(), extbox,
                         intbox);
        components.at(vi).storeChunk(std::move(ptr), start, count);
      }
    }
  }
}

void write_array(openPMD::Iteration &write_iter, const buffer_id_t &id,
                 const array_buffer_t &array, const bool is_ioproc) {
  const std::string meshname = mesh_name(id);
  if (io_verbose)
    CCTK_VINFO("Defining mesh %s...", meshname.c_str());
  openPMD::Mesh mesh = write_iter.meshes[meshname];
  mesh.setTimeOffset(CCTK_REAL(0));

  std::vector<openPMD::MeshRecordComponent> components;
  components.reserve(id.vars.size());
  for (const std::string &var : id.vars)
    components.push_back(mesh[make_componentname(var)]);

  const openPMD::Extent extent = reversed<std::uint64_t>(array.shape);
  const openPMD::Dataset dataset(array_datatype(array.vartype), extent);
  for (openPMD::MeshRecordComponent &component : components)
    component.resetDataset(dataset);

  if (!is_ioproc)
    return;
  const openPMD::Offset start{0, 0, 0};
  const std::size_t varsize =
      num_points(array.shape) * CCTK_VarTypeSize(array.vartype);
  for (std::size_t vi = 0; vi < components.size(); ++vi)
    store_array(components.at(vi), array.vartype,
                static_cast<const char *>(array.data) + vi * varsize, start,
                extent);
}

void read_grid(openPMD::Iteration &read_iter, const checkpoint_record_t &record,
               const buffer_id_t &id, const grid_buffer_t &grid,
               std::vector<std::function<void()> > &tasks) {
  const level_layout_t layout = layout_of(record, *id.level);

  const std::string meshname = mesh_name(id);
  if (io_verbose)
    CCTK_VINFO("Reading mesh %s...", meshname.c_str());
  const openPMD::Mesh &mesh = read_iter.meshes.at(meshname);
  std::vector<openPMD::MeshRecordComponent> components =
      read_components(mesh, meshname, id);

  amrex::MultiFab &mfab = *grid.mfab;
  for (int local = 0; local < mfab.local_size(); ++local) {
    const int c = mfab.IndexArray().at(local);
    const ibox_t extbox = make_ibox(mfab.fabbox(c));
    const ibox_t intbox = make_ibox(mfab.box(c));
    const openPMD::Offset start =
        reversed<std::uint64_t>(difference(intbox.lo, layout.idomain.lo));
    const openPMD::Extent count = reversed<std::uint64_t>(intbox.shape());
    const int np = intbox.size();
    amrex::FArrayBox &fab = mfab[c];
    for (std::size_t vi = 0; vi < components.size(); ++vi) {
      assert_inside(start, count, components.at(vi).getExtent());
      if (intbox == extbox) {
        load_raw(components.at(vi), fab.dataPtr() + vi * np, start, count);
        continue;
      }
      CCTK_REAL *const var_ptr = fab.dataPtr() + vi * extbox.size();
      const index3 ext_shape = extbox.shape();
      const index3 offset = difference(intbox.lo, extbox.lo);
      const int adj = ext_shape[0];
      const int adk = adj * ext_shape[1];
      CCTK_REAL *const amrex_ptr =
          var_ptr + offset[0] + adj * offset[1] + adk * offset[2];
      const index3 int_shape = intbox.shape();
      const int cdj = int_shape[0];
      const int cdk = cdj * int_shape[1];
      CCTK_REAL *const contig_ptr = var_ptr + extbox.size() - np;
      load_raw(components.at(vi), contig_ptr, start, count);
      tasks.emplace_back([=]() {
        for (int k = 0; k < int_shape[2]; ++k)
          for (int j = 0; j < int_shape[1]; ++j)
            for (int i = 0; i < int_shape[0]; ++i)
              amrex_ptr[i + adj * j + adk * k] =
                  contig_ptr[i + cdj * j + cdk * k];
      });
    }
  }
}

void read_array(openPMD::Iteration &read_iter, const buffer_id_t &id,
                const array_buffer_t &array) {
  const std::string meshname = mesh_name(id);
  if (io_verbose)
    CCTK_VINFO("Reading mesh %s...", meshname.c_str());
  const openPMD::Mesh &mesh = read_iter.meshes.at(meshname);
  std::vector<openPMD::MeshRecordComponent> components =
      read_components(mesh, meshname, id);

  const openPMD::Offset start{0, 0, 0};
  const openPMD::Extent count = reversed<std::uint64_t>(array.shape);
  const std::size_t varsize =
      num_points(array.shape) * CCTK_VarTypeSize(array.vartype);
  for (std::size_t vi = 0; vi < components.size(); ++vi) {
    assert_inside(start, count, components.at(vi).getExtent());
    load_array(components.at(vi), array.vartype,
               static_cast<char *>(array.data) + vi * varsize, start, count);
  }
}

struct openpmd_codec_t final : checkpoint_codec_t {
  codec_limits_t limits() const override {
    return {.carries_arrays = true,
            .max_timelevels = std::numeric_limits<int>::max(),
            .rechops_on_recovery = true};
  }
  std::string key(const buffer_id_t &id) const override {
    return mesh_name(id);
  }

  int read_parameters(const std::string &dir, const std::string &file) override;
  void encode(const checkpoint_record_t &record, const std::string &dir,
              const std::string &file) override;
  checkpoint_record_t decode_structure(const std::string &dir,
                                       const std::string &file,
                                       int iteration) override;
  void decode_data(const checkpoint_record_t &record, const std::string &dir,
                   const std::string &file) override;
};

int openpmd_codec_t::read_parameters(const std::string &dir,
                                     const std::string &file) {
  assert(!dir.empty());
  assert(!file.empty());

  static Timer timer("openpmd_codec_t::read_parameters");
  Interval interval(timer);

  const std::string filename = series_name(dir, file, get_format());
  std::optional<openPMD::Series> series;
  try {
    series.emplace(filename, openPMD::Access::READ_ONLY, MPI_COMM_WORLD,
                   options);
  } catch (const openPMD::no_such_file_error &) {
    if (io_verbose) {
      CCTK_VINFO("Not recovering parameters:");
      CCTK_VINFO("  Could not find an openPMD checkpoint file \"%s\"",
                 filename.c_str());
    }
    return -1;
  }

  int iteration = -1;
  for (auto iter = series->iterations.begin(); iter != series->iterations.end();
       ++iter)
    iteration = iter->first;
  if (iteration < 0) {
    if (io_verbose) {
      CCTK_VINFO("Not recovering parameters:");
      CCTK_VINFO("  Could not find an openPMD checkpoint file \"%s\"",
                 filename.c_str());
    }
    return -1;
  }

  CCTK_VINFO("Recovering parameters from checkpoint file \"%s\" iteration %d",
             filename.c_str(), iteration);

  openPMD::Iteration read_iter = series->iterations[iteration];
  if (!read_iter.containsAttribute("AllParameters"))
    CCTK_VERROR("The openPMD checkpoint \"%s\" iteration %d holds no "
                "parameters",
                filename.c_str(), iteration);
  const openPMD::Attribute parameters_attr =
      read_iter.getAttribute("AllParameters");
  assert(parameters_attr.dtype == openPMD::Datatype::STRING);
  const std::string parameters = parameters_attr.get<std::string>();
  IOUtil_SetAllParameters(parameters.data());

  return iteration;
}

void openpmd_codec_t::encode(const checkpoint_record_t &record,
                             const std::string &dir, const std::string &file) {
  static Timer timer("openpmd_codec_t::encode");
  Interval interval(timer);

  const openPMD::Format format = get_format();
  const int ierr = CCTK_CreateDirectory(0755, dir.c_str());
  assert(ierr >= 0);

  const std::string filename = series_name(dir, file, format);
  if (io_verbose)
    CCTK_VINFO("Writing openPMD checkpoint \"%s\" iteration %d...",
               filename.c_str(), record.iteration);
  openPMD::Series series(filename, openPMD::Access::CREATE, MPI_COMM_WORLD,
                         options);
  series.setIterationEncoding(iterationEncoding);

  {
    char author[1000] = {0};
    const char *const user = std::getenv("USER");
    if (user)
      std::snprintf(author, sizeof author, "%s", user);
    MPI_Bcast(author, sizeof author, MPI_CHAR, 0, MPI_COMM_WORLD);
    if (author[0])
      series.setAuthor(author);
  }
  {
    char hostname[1000];
    Util_GetHostName(hostname, sizeof hostname);
    MPI_Bcast(hostname, sizeof hostname, MPI_CHAR, 0, MPI_COMM_WORLD);
    series.setMachine(hostname);
  }

  {
    openPMD::WriteIterations write_iters = series.writeIterations();
    openPMD::Iteration write_iter = write_iters[record.iteration];
    write_iter.setTime(record.time);
    write_iter.setDt(record.delta_time);
    write_iter.setTimeUnitSI(Unit::time);

    write_iter.setAttribute("AllParameters", record.parameters);

    const int ndims = Loop::dim;
    write_iter.setAttribute<std::int64_t>("numDims", ndims);
    const int npatches = record.levels.size();
    write_iter.setAttribute<std::int64_t>("numPatches", npatches);
    std::vector<std::string> patch_suffixes(npatches);
    for (int patch = 0; patch < npatches; ++patch)
      patch_suffixes.at(patch) = patch_suffix(patch);
    write_iter.setAttribute("patchSuffixes", patch_suffixes);
    for (int patch = 0; patch < npatches; ++patch) {
      const std::vector<level_record_t> &levels = record.levels.at(patch);
      const int nlevels = levels.size();
      write_iter.setAttribute<std::int64_t>(
          "numLevels" + patch_suffixes.at(patch), nlevels);
      std::vector<std::string> level_suffixes(nlevels);
      for (int level = 0; level < nlevels; ++level)
        level_suffixes.at(level) =
            level_suffix(patch_suffixes.at(patch), level);
      write_iter.setAttribute("levelSuffixes" + patch_suffixes.at(patch),
                              level_suffixes);
      for (int level = 0; level < nlevels; ++level) {
        const level_record_t &levelrecord = levels.at(level);
        const int nchunks = levelrecord.boxes.size();
        std::vector<std::int64_t> chunk_infos(2 * ndims * nchunks);
        for (int chunk = 0; chunk < nchunks; ++chunk) {
          const amrex::Box &box = levelrecord.boxes.at(chunk);
          for (int d = 0; d < ndims; ++d) {
            chunk_infos.at(2 * ndims * chunk + 0 * ndims + ndims - 1 - d) =
                box.smallEnd()[d];
            chunk_infos.at(2 * ndims * chunk + 1 * ndims + ndims - 1 - d) =
                box.bigEnd()[d] + 1;
          }
        }
        write_iter.setAttribute("chunkInfo" + level_suffixes.at(level),
                                chunk_infos);

        if (levelrecord.iteration) {
          write_iter.setAttribute(
              "iteration_num" + level_suffixes.at(level),
              static_cast<std::int64_t>(levelrecord.iteration->num));
          write_iter.setAttribute(
              "iteration_den" + level_suffixes.at(level),
              static_cast<std::int64_t>(levelrecord.iteration->den));
        }
        if (levelrecord.last_step_iteration)
          write_iter.setAttribute(
              "last_step_iteration" + level_suffixes.at(level),
              static_cast<std::int64_t>(*levelrecord.last_step_iteration));
      }
    }

    const bool is_ioproc = CCTK_MyProc(nullptr) == 0;
    for (const buffer_t &buffer : record.buffers)
      std::visit(overloaded{[&](const grid_buffer_t &grid) {
                              write_grid(write_iter, record, buffer.id, grid);
                            },
                            [&](const array_buffer_t &array) {
                              write_array(write_iter, buffer.id, array,
                                          is_ioproc);
                            }},
                 buffer.data);

    if (io_verbose)
      CCTK_VINFO("Performing all writes...");
    series.flush();
    write_iter.close();
  }

  if (CCTK_MyProc(nullptr) == 0) {
    std::ofstream visit(dir + "/" + file + ".openpmd.visit", std::ios::app);
    assert(visit.good());
    visit << file << ".it" << std::setw(8) << std::setfill('0')
          << record.iteration << openPMD::suffix(format) << "\n";
  }
}

checkpoint_record_t openpmd_codec_t::decode_structure(const std::string &dir,
                                                      const std::string &file,
                                                      const int iteration) {
  assert(!dir.empty());
  assert(!file.empty());
  assert(iteration >= 0);

  static Timer timer("openpmd_codec_t::decode_structure");
  Interval interval(timer);

  const std::string filename = series_name(dir, file, get_format());
  if (io_verbose)
    CCTK_VINFO("Reading openPMD checkpoint structure \"%s\" iteration %d...",
               filename.c_str(), iteration);
  openPMD::Series series(filename, openPMD::Access::READ_ONLY, MPI_COMM_WORLD,
                         options);
  if (!series.iterations.count(iteration))
    CCTK_VERROR("The openPMD checkpoint \"%s\" holds no iteration %d",
                filename.c_str(), iteration);
  openPMD::Iteration read_iter = series.iterations[iteration];
  read_iter.open();

  checkpoint_record_t record;
  record.iteration = iteration;
  record.time = read_iter.time<double>();
  record.delta_time = read_iter.dt<double>();

  const int ndims = read_iter.getAttribute("numDims").get<std::int64_t>();
  assert(ndims == 3);

  const int npatches = read_iter.getAttribute("numPatches").get<std::int64_t>();
  assert(npatches > 0);
  std::vector<std::string> patch_suffixes;
  if (npatches == 1) {
    assert(read_iter.getAttribute("patchSuffixes").dtype ==
           openPMD::Datatype::STRING);
    patch_suffixes.push_back(
        read_iter.getAttribute("patchSuffixes").get<std::string>());
  } else {
    assert(read_iter.getAttribute("patchSuffixes").dtype ==
           openPMD::Datatype::VEC_STRING);
    patch_suffixes = read_iter.getAttribute("patchSuffixes")
                         .get<std::vector<std::string> >();
  }
  assert(int(patch_suffixes.size()) == npatches);

  record.levels.resize(npatches);
  for (int patch = 0; patch < npatches; ++patch) {
    const std::string &psuffix = patch_suffixes.at(patch);
    const int nlevels =
        read_iter.getAttribute("numLevels" + psuffix).get<std::int64_t>();
    assert(nlevels > 0);
    std::vector<std::string> level_suffixes;
    if (nlevels == 1) {
      assert(read_iter.getAttribute("levelSuffixes" + psuffix).dtype ==
             openPMD::Datatype::STRING);
      level_suffixes.push_back(
          read_iter.getAttribute("levelSuffixes" + psuffix).get<std::string>());
    } else {
      assert(read_iter.getAttribute("levelSuffixes" + psuffix).dtype ==
             openPMD::Datatype::VEC_STRING);
      level_suffixes = read_iter.getAttribute("levelSuffixes" + psuffix)
                           .get<std::vector<std::string> >();
    }
    assert(int(level_suffixes.size()) == nlevels);

    record.levels.at(patch).resize(nlevels);
    for (int level = 0; level < nlevels; ++level) {
      const std::string &lsuffix = level_suffixes.at(level);
      level_record_t &levelrecord = record.levels.at(patch).at(level);

      const std::vector<std::int64_t> chunk_infos =
          read_iter.getAttribute("chunkInfo" + lsuffix)
              .get<std::vector<std::int64_t> >();
      assert(chunk_infos.size() % (2 * ndims) == 0);
      const int nchunks = chunk_infos.size() / (2 * ndims);
      for (int chunk = 0; chunk < nchunks; ++chunk) {
        const int offset = 2 * ndims * chunk;
        amrex::IntVect small, big;
        for (int d = 0; d < ndims; ++d) {
          small[d] = chunk_infos.at(offset + 0 * ndims + ndims - 1 - d);
          big[d] = chunk_infos.at(offset + 1 * ndims + ndims - 1 - d);
        }
        levelrecord.boxes.push_back(amrex::Box(small, big - 1));
      }

      const std::string num_attr = "iteration_num" + lsuffix;
      const std::string den_attr = "iteration_den" + lsuffix;
      if (read_iter.containsAttribute(num_attr) &&
          read_iter.containsAttribute(den_attr))
        levelrecord.iteration =
            rat64(read_iter.getAttribute(num_attr).get<std::int64_t>(),
                  read_iter.getAttribute(den_attr).get<std::int64_t>());

      const std::string last_step_attr = "last_step_iteration" + lsuffix;
      if (read_iter.containsAttribute(last_step_attr))
        levelrecord.last_step_iteration =
            int(read_iter.getAttribute(last_step_attr).get<std::int64_t>());
    }
  }

  for (auto iter = read_iter.meshes.begin(); iter != read_iter.meshes.end();
       ++iter)
    record.present.insert(iter->first);

  read_iter.close();
  return record;
}

void openpmd_codec_t::decode_data(const checkpoint_record_t &record,
                                  const std::string &dir,
                                  const std::string &file) {
  static Timer timer("openpmd_codec_t::decode_data");
  Interval interval(timer);

  const std::string filename = series_name(dir, file, get_format());
  if (io_verbose)
    CCTK_VINFO("Reading openPMD checkpoint data \"%s\" iteration %d...",
               filename.c_str(), record.iteration);
  openPMD::Series series(filename, openPMD::Access::READ_ONLY, MPI_COMM_WORLD,
                         options);
  if (!series.iterations.count(record.iteration))
    CCTK_VERROR("The openPMD checkpoint \"%s\" holds no iteration %d",
                filename.c_str(), record.iteration);
  openPMD::Iteration read_iter = series.iterations[record.iteration];
  read_iter.open();

  std::vector<std::function<void()> > tasks;
  for (const buffer_t &buffer : record.buffers)
    std::visit(overloaded{[&](const grid_buffer_t &grid) {
                            read_grid(read_iter, record, buffer.id, grid,
                                      tasks);
                          },
                          [&](const array_buffer_t &array) {
                            read_array(read_iter, buffer.id, array);
                          }},
               buffer.data);

  if (io_verbose)
    CCTK_VINFO("Performing all reads...");
  series.flush();
  for (std::function<void()> &task : tasks)
    std::move(task)();
  tasks.clear();

  read_iter.close();
}

} // namespace

std::unique_ptr<checkpoint_codec_t> make_openpmd_codec() {
  return std::make_unique<openpmd_codec_t>();
}

} // namespace CarpetX

#endif
