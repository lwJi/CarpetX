#include "io/checkpoint.hxx"
#include "io/io_silo_names.hxx"

#include <CactusBase/IOUtil/src/ioutil_CheckpointRecovery.h>
#include <cctk.h>
#include <cctk_Parameters.h>

#ifdef HAVE_CAPABILITY_Silo

#include "util/mpi_types.hxx"

#include <silo.hxx>

#include <AMReX_Box.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_Geometry.H>
#include <AMReX_IndexType.H>
#include <AMReX_IntVect.H>
#include <AMReX_MultiFab.H>

#include <mpi.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <climits>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <optional>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace CarpetX {

namespace {

constexpr bool io_verbose = true;

constexpr int ndims = Loop::dim;
using ivect = std::array<int, ndims>;
using extent_t = std::array<std::array<CCTK_REAL, ndims>, 2>;

template <class... Fs> struct overloaded : Fs... {
  using Fs::operator()...;
};
template <class... Fs> overloaded(Fs...) -> overloaded<Fs...>;

const MPI_Comm mpi_comm = MPI_COMM_WORLD;

constexpr int state_write_tag = 22900;
constexpr int state_read_tag = 22901;
constexpr int band_write_tag = 22902;
constexpr int band_read_tag = 22903;

int ioproc_every(const int nprocs) {
  DECLARE_CCTK_PARAMETERS;
  if (CCTK_EQUALS(out_mode, "proc"))
    return 1;
  if (CCTK_EQUALS(out_mode, "np")) {
    assert(out_proc_every > 0);
    return out_proc_every;
  }
  if (CCTK_EQUALS(out_mode, "onefile"))
    return nprocs;
  CCTK_VERROR("Unexpected IO::out_mode \"%s\"", out_mode);
}

int db_centering(const amrex::IndexType &indextype) {
  constexpr std::array<int, ndims + 1> centerings{DB_NODECENT, DB_EDGECENT,
                                                  DB_FACECENT, DB_ZONECENT};
  const int centering =
      centerings.at(indextype.cellCentered(0) + indextype.cellCentered(1) +
                    indextype.cellCentered(2));
  assert(centering != DB_EDGECENT && centering != DB_FACECENT);
  return centering;
}

ivect to_ivect(const amrex::IntVect &v) { return {v[0], v[1], v[2]}; }

ivect shape(const amrex::Box &box) {
  ivect dims;
  for (int d = 0; d < ndims; ++d)
    dims[d] = box.length(d);
  return dims;
}

std::ptrdiff_t num_points(const ivect &dims) {
  std::ptrdiff_t np = 1;
  for (const int n : dims)
    np *= n;
  assert(np >= 0 && np <= INT_MAX);
  return np;
}

std::string clock_varname(const std::string &what, const level_t level) {
  return DB::legalize_name(driver_name) + "/" +
         DB::legalize_name(what + ".m" + std::to_string(level.patch().index()) +
                           ".rl" + std::to_string(level.index()));
}

std::string state_name(const buffer_id_t &id) {
  return buffer_id_t{
      .group = id.group, .vars = {}, .level = id.level, .which = tl_t{0}}
      .name();
}

const std::string *band_tag(const buffer_id_t &id) {
  return std::get_if<std::string>(&id.which);
}

std::string without_component(const std::string &name) {
  return std::regex_replace(name, std::regex("_c[0-9]{8}"), "",
                            std::regex_constants::format_first_only);
}

std::vector<std::string> output_variables(const checkpoint_record_t &record) {
  std::vector<std::string> variables;
  std::set<std::string> groups;
  for (const buffer_t &buffer : record.buffers) {
    if (!std::holds_alternative<grid_buffer_t>(buffer.data) ||
        band_tag(buffer.id) || !groups.insert(buffer.id.group).second)
      continue;
    variables.insert(variables.end(), buffer.id.vars.begin(),
                     buffer.id.vars.end());
  }
  return variables;
}

std::map<std::string, const amrex::MultiFab *>
state_buffers(const checkpoint_record_t &record) {
  std::map<std::string, const amrex::MultiFab *> states;
  for (const buffer_t &buffer : record.buffers) {
    const grid_buffer_t *const grid = std::get_if<grid_buffer_t>(&buffer.data);
    if (!grid || band_tag(buffer.id))
      continue;
    assert(std::get<tl_t>(buffer.id.which) == tl_t{0});
    states.emplace(buffer.id.name(), grid->mfab);
  }
  return states;
}

const amrex::MultiFab &
state_of(const std::map<std::string, const amrex::MultiFab *> &states,
         const buffer_id_t &id) {
  const auto state = states.find(state_name(id));
  if (state == states.end())
    CCTK_VERROR("The Silo checkpoint lists band %s without the state it "
                "belongs to",
                id.name().c_str());
  return *state->second;
}

DB::ptr<DBoptlist> make_optlist() {
  DB::ptr<DBoptlist> optlist = DB::make(DBMakeOptlist(10));
  assert(optlist);
  return optlist;
}

void add_option(DBoptlist *const optlist, const int option, void *const value) {
  const int ierr = DBAddOption(optlist, option, value);
  assert(!ierr);
}

void write_string(DBfile *const file, const char *const name,
                  const std::string &value) {
  const int dims = value.length();
  const int ierr = DBWrite(file, name, value.data(), &dims, 1, DB_CHAR);
  assert(!ierr);
}

void write_metadata_is_time_varying(DBfile *const file) {
  const int dims = 1;
  const int value = 1;
  const int ierr =
      DBWrite(file, "MetadataIsTimeVarying", &value, &dims, 1, DB_INT);
  assert(!ierr);
}

void put_quadmesh(DBfile *const file, const std::string &meshname,
                  const std::array<const void *, ndims> &coord_ptrs,
                  ivect dims_vc, const int coordtype, const bool cartesian,
                  const ivect &offsets, int cycle, double dtime) {
  const DB::ptr<DBoptlist> optlist = make_optlist();
  int coordsys = DB_CARTESIAN;
  if (cartesian)
    add_option(optlist.get(), DBOPT_COORDSYS, &coordsys);
  add_option(optlist.get(), DBOPT_CYCLE, &cycle);
  ivect lo_offset = offsets, hi_offset = offsets;
  add_option(optlist.get(), DBOPT_LO_OFFSET, lo_offset.data());
  add_option(optlist.get(), DBOPT_HI_OFFSET, hi_offset.data());
  int column_major = 0;
  add_option(optlist.get(), DBOPT_MAJORORDER, &column_major);
  add_option(optlist.get(), DBOPT_DTIME, &dtime);
  int hide_from_gui = 1;
  add_option(optlist.get(), DBOPT_HIDE_FROM_GUI, &hide_from_gui);
  const int ierr = DBPutQuadmesh(
      file, meshname.c_str(), nullptr, coord_ptrs.data(), dims_vc.data(), ndims,
      db_datatype_v<CCTK_REAL>, coordtype, optlist.get());
  assert(!ierr);
}

void put_quadvar(DBfile *const file, const std::string &varname,
                 const std::string &meshname, const CCTK_REAL *const data,
                 ivect dims, const int centering, int cycle, double dtime) {
  const DB::ptr<DBoptlist> optlist = make_optlist();
  int coordsys = DB_CARTESIAN;
  add_option(optlist.get(), DBOPT_COORDSYS, &coordsys);
  add_option(optlist.get(), DBOPT_CYCLE, &cycle);
  int column_major = 0;
  add_option(optlist.get(), DBOPT_MAJORORDER, &column_major);
  add_option(optlist.get(), DBOPT_DTIME, &dtime);
  int hide_from_gui = 1;
  add_option(optlist.get(), DBOPT_HIDE_FROM_GUI, &hide_from_gui);
  const int ierr = DBPutQuadvar1(
      file, varname.c_str(), meshname.c_str(), data, dims.data(), ndims,
      nullptr, 0, db_datatype_v<CCTK_REAL>, centering, optlist.get());
  assert(!ierr);
}

std::array<std::vector<CCTK_REAL>, ndims>
cartesian_coords(const amrex::Geometry &geom, const amrex::Box &fabbox,
                 const ivect &dims_vc) {
  std::array<std::vector<CCTK_REAL>, ndims> coords;
  for (int d = 0; d < ndims; ++d) {
    coords[d].resize(dims_vc[d]);
    for (int i = 0; i < dims_vc[d]; ++i)
      coords[d][i] =
          geom.ProbLo(d) + (fabbox.smallEnd(d) + i) * geom.CellSize(d);
  }
  return coords;
}

struct writer_t {
  const checkpoint_record_t &record;
  int myproc, every;
  int metafile_ioproc;
  DB::ptr<DBfile> dbfile;
  std::map<std::string, const amrex::MultiFab *> states;
  std::map<std::pair<int, int>, std::set<ivect> > meshes;
  std::vector<std::vector<std::vector<std::array<CCTK_REAL, ndims> > > >
      coordinate_minima, coordinate_maxima;

  bool writes_file() const { return bool(dbfile); }

  const CCTK_REAL *gather(const amrex::MultiFab &mfab, const int component,
                          const std::ptrdiff_t count, const int tag,
                          std::vector<CCTK_REAL> &buffer) const {
    const int proc = mfab.DistributionMap()[component];
    const int ioproc = proc / every * every;
    assert(count <= INT_MAX);
    if (proc == myproc && ioproc == myproc)
      return mfab[component].dataPtr();
    if (proc == myproc) {
      MPI_Send(mfab[component].dataPtr(), count, mpi_datatype_v<CCTK_REAL>,
               ioproc, tag, mpi_comm);
      return nullptr;
    }
    buffer.resize(count);
    MPI_Recv(buffer.data(), count, mpi_datatype_v<CCTK_REAL>, proc, tag,
             mpi_comm, MPI_STATUS_IGNORE);
    return buffer.data();
  }

  bool takes_part(const amrex::MultiFab &mfab, const int component) const {
    const int proc = mfab.DistributionMap()[component];
    const int ioproc = proc / every * every;
    return proc == myproc || ioproc == myproc;
  }

  void write_state(const buffer_id_t &id, const amrex::MultiFab &mfab) {
    const level_t level = *id.level;
    const level_record_t &levelrecord =
        record.levels.at(level.patch().index()).at(level.index());
    const bool cartesian = !levelrecord.vertex_coords;
    const int numvars = id.vars.size();
    assert(mfab.nComp() == numvars);
    const amrex::IndexType &indextype = mfab.ixType();
    const ivect nghosts = to_ivect(mfab.nGrowVect());
    std::set<ivect> &level_meshes =
        meshes[{level.patch().index(), level.index()}];
    const bool have_mesh = level_meshes.count(nghosts);

    for (int component = 0; component < mfab.size(); ++component) {
      if (!takes_part(mfab, component))
        continue;
      const amrex::Box &fabbox = mfab.fabbox(component);
      const ivect dims = shape(fabbox);
      const std::ptrdiff_t zonecount = num_points(dims);
      const std::string meshname = make_meshname(nghosts, level, component);

      if (writes_file() && !have_mesh) {
        ivect dims_vc;
        for (int d = 0; d < ndims; ++d)
          dims_vc[d] = dims[d] + int(indextype.cellCentered(d));
        std::array<std::vector<CCTK_REAL>, ndims> coords;
        std::array<const void *, ndims> coord_ptrs;
        if (cartesian) {
          coords = cartesian_coords(levelrecord.geom, fabbox, dims_vc);
          for (int d = 0; d < ndims; ++d)
            coord_ptrs[d] = coords[d].data();
        } else {
          const int proc = mfab.DistributionMap()[component];
          assert(proc == myproc && proc / every * every == myproc);
          const amrex::MultiFab &coord_mfab = *levelrecord.vertex_coords;
          assert(coord_mfab.nGrowVect() == mfab.nGrowVect());
          const amrex::FArrayBox &coord_fab = coord_mfab[component];
          for (int d = 0; d < ndims; ++d) {
            coord_ptrs[d] = coord_fab.dataPtr(d);
            assert(coord_ptrs[d]);
          }
          std::array<CCTK_REAL, ndims> xmin, xmax;
          for (int d = 0; d < ndims; ++d) {
#ifdef AMREX_USE_GPU
            constexpr auto run_on = amrex::RunOn::Device;
#else
            constexpr auto run_on = amrex::RunOn::Host;
#endif
            const auto xminmax = coord_fab.minmax<run_on>(d);
            xmin[d] = xminmax.first;
            xmax[d] = xminmax.second;
          }
          coordinate_minima.at(level.patch().index())
              .at(level.index())
              .at(component) = xmin;
          coordinate_maxima.at(level.patch().index())
              .at(level.index())
              .at(component) = xmax;
        }
        put_quadmesh(dbfile.get(), meshname, coord_ptrs, dims_vc,
                     cartesian ? DB_COLLINEAR : DB_NONCOLLINEAR, cartesian,
                     nghosts, record.iteration, record.time);
        level_meshes.insert(nghosts);
      }

      std::vector<CCTK_REAL> buffer;
      const CCTK_REAL *const data =
          gather(mfab, component, numvars * zonecount, state_write_tag, buffer);

      if (writes_file()) {
        const int centering = db_centering(indextype);
        for (int vi = 0; vi < numvars; ++vi)
          put_quadvar(dbfile.get(),
                      make_varname(id.vars.at(vi), level, component), meshname,
                      data + vi * zonecount, dims, centering, record.iteration,
                      record.time);
      }
    }
  }

  void write_band(const buffer_id_t &id, const amrex::MultiFab &band) {
    const level_t level = *id.level;
    const std::string &tag = *band_tag(id);
    const level_record_t &levelrecord =
        record.levels.at(level.patch().index()).at(level.index());
    const int numvars = id.vars.size();
    assert(band.nComp() == numvars);
    assert(band.nGrowVect() == 0);
    const amrex::IndexType &indextype = state_of(states, id).ixType();
    const int centering = db_centering(indextype);
    const ivect band_nghosts{0, 0, 0};

    for (int component = 0; component < band.size(); ++component) {
      if (!takes_part(band, component))
        continue;
      const amrex::Box &fabbox = band.fabbox(component);
      const ivect dims = shape(fabbox);
      const std::ptrdiff_t zonecount = num_points(dims);
      const std::string meshname =
          make_meshname(band_nghosts, level, component, tag);

      if (writes_file()) {
        ivect dims_vc;
        for (int d = 0; d < ndims; ++d)
          dims_vc[d] = dims[d] + int(indextype.cellCentered(d));
        const std::array<std::vector<CCTK_REAL>, ndims> coords =
            cartesian_coords(levelrecord.geom, fabbox, dims_vc);
        std::array<const void *, ndims> coord_ptrs;
        for (int d = 0; d < ndims; ++d)
          coord_ptrs[d] = coords[d].data();
        put_quadmesh(dbfile.get(), meshname, coord_ptrs, dims_vc, DB_COLLINEAR,
                     true, band_nghosts, record.iteration, record.time);
      }

      std::vector<CCTK_REAL> buffer;
      const CCTK_REAL *const data =
          gather(band, component, numvars * zonecount, band_write_tag, buffer);

      if (writes_file())
        for (int vi = 0; vi < numvars; ++vi)
          put_quadvar(dbfile.get(),
                      make_varname(id.vars.at(vi), level, component, tag),
                      meshname, data + vi * zonecount, dims, centering,
                      record.iteration, record.time);
    }
  }

  void reduce_coordinate_extents() {
    for (std::size_t patch = 0; patch < coordinate_minima.size(); ++patch) {
      for (std::size_t level = 0; level < coordinate_minima.at(patch).size();
           ++level) {
        auto &minima = coordinate_minima.at(patch).at(level);
        auto &maxima = coordinate_maxima.at(patch).at(level);
        if (minima.empty())
          continue;
        const bool isroot = myproc == metafile_ioproc;
        MPI_Reduce(isroot ? MPI_IN_PLACE : minima.data(),
                   isroot ? minima.data() : nullptr, ndims * minima.size(),
                   mpi_datatype_v<CCTK_REAL>, MPI_MIN, metafile_ioproc,
                   mpi_comm);
        MPI_Reduce(isroot ? MPI_IN_PLACE : maxima.data(),
                   isroot ? maxima.data() : nullptr, ndims * maxima.size(),
                   mpi_datatype_v<CCTK_REAL>, MPI_MAX, metafile_ioproc,
                   mpi_comm);
      }
    }
  }
};

struct group_layout_t {
  std::string group;
  std::vector<std::string> vars;
  std::vector<std::vector<const amrex::MultiFab *> > mfabs;
};

std::vector<group_layout_t> group_layouts(const checkpoint_record_t &record) {
  std::vector<group_layout_t> layouts;
  std::map<std::string, std::size_t> index;
  for (const buffer_t &buffer : record.buffers) {
    const grid_buffer_t *const grid = std::get_if<grid_buffer_t>(&buffer.data);
    if (!grid || band_tag(buffer.id))
      continue;
    const auto [where, inserted] =
        index.emplace(buffer.id.group, layouts.size());
    if (inserted) {
      group_layout_t layout{.group = buffer.id.group, .vars = buffer.id.vars};
      layout.mfabs.resize(record.levels.size());
      for (std::size_t patch = 0; patch < record.levels.size(); ++patch)
        layout.mfabs.at(patch).resize(record.levels.at(patch).size(), nullptr);
      layouts.push_back(std::move(layout));
    }
    const level_t level = *buffer.id.level;
    layouts.at(where->second)
        .mfabs.at(level.patch().index())
        .at(level.index()) = grid->mfab;
  }
  for (const group_layout_t &layout : layouts)
    for (const auto &patch_mfabs : layout.mfabs)
      for (const amrex::MultiFab *const mfab : patch_mfabs)
        assert(mfab);
  return layouts;
}

void put_groupelmap(DBfile *const metafile, const std::string &name,
                    const int num_segments,
                    const std::vector<std::vector<int> > &segment_data) {
  std::vector<int> segment_types(segment_data.size(), DB_BLOCKCENT);
  std::vector<int> segment_lengths;
  std::vector<const int *> segment_data_ptrs;
  segment_lengths.reserve(segment_data.size());
  segment_data_ptrs.reserve(segment_data.size());
  for (const auto &data : segment_data) {
    segment_lengths.push_back(data.size());
    segment_data_ptrs.push_back(data.data());
  }
  const int ierr =
      DBPutGroupelmap(metafile, name.c_str(), num_segments,
                      segment_types.data(), segment_lengths.data(), nullptr,
                      segment_data_ptrs.data(), nullptr, 0, nullptr);
  assert(!ierr);
}

struct metafile_writer_t {
  const checkpoint_record_t &record;
  const std::string &file;
  const int every;
  DBfile *const metafile;
  const std::vector<std::vector<std::vector<std::array<CCTK_REAL, ndims> > > >
      &coordinate_minima, &coordinate_maxima;

  int num_patches() const { return record.levels.size(); }
  int num_levels(const int patch) const {
    return record.levels.at(patch).size();
  }
  int max_num_levels() const {
    int nlevels = 0;
    for (int patch = 0; patch < num_patches(); ++patch)
      nlevels = std::max(nlevels, num_levels(patch));
    return nlevels;
  }
  int num_components(const int patch, const int level) const {
    return record.levels.at(patch).at(level).boxes.size();
  }
  bool is_cartesian(const int patch) const {
    return !record.levels.at(patch).at(0).vertex_coords;
  }

  std::string block_path(const amrex::MultiFab &mfab,
                         const int component) const {
    const int proc = mfab.DistributionMap()[component];
    return make_subdirname(file, record.iteration) + "/" +
           make_filename(file, record.iteration, proc / every);
  }

  void write_mrgtree(const std::string &multimeshname,
                     const group_layout_t &layout) const {
    const int nlevels = max_num_levels();
    const int npatches = num_patches();
    std::vector<int> comp0_level(nlevels);
    std::vector<int> ncomps_level(nlevels);
    std::vector<std::vector<int> > comp0_level_patch(
        nlevels, std::vector<int>(npatches, 0));
    std::vector<std::vector<int> > ncomps_level_patch(
        nlevels, std::vector<int>(npatches, 0));
    int ncomps_total = 0;
    for (int level = 0; level < nlevels; ++level) {
      comp0_level.at(level) = ncomps_total;
      for (int patch = 0; patch < npatches; ++patch) {
        if (level < num_levels(patch)) {
          comp0_level_patch.at(level).at(patch) = ncomps_total;
          ncomps_total += num_components(patch, level);
          ncomps_level_patch.at(level).at(patch) =
              ncomps_total - comp0_level_patch.at(level).at(patch);
        }
      }
      ncomps_level.at(level) = ncomps_total - comp0_level.at(level);
    }

    const std::string levelmaps_name = multimeshname + "_wmrgtree_lvlMaps";
    {
      std::vector<std::vector<int> > segment_data;
      segment_data.reserve(nlevels);
      for (int level = 0; level < nlevels; ++level) {
        std::vector<int> data;
        data.reserve(ncomps_level.at(level));
        for (int c = 0; c < ncomps_level.at(level); ++c)
          data.push_back(comp0_level.at(level) + c);
        segment_data.push_back(std::move(data));
      }
      put_groupelmap(metafile, levelmaps_name, nlevels, segment_data);
    }

    const std::string childmaps_name = multimeshname + "_wmrgtree_chldMaps";
    std::vector<int> num_children;
    {
      std::vector<std::vector<int> > segment_data;
      segment_data.reserve(ncomps_total);
      for (int patch = 0; patch < npatches; ++patch) {
        for (int level = 0; level < num_levels(patch); ++level) {
          if (level + 1 < num_levels(patch)) {
            const amrex::MultiFab &mfab = *layout.mfabs.at(patch).at(level);
            const amrex::MultiFab &fine_mfab =
                *layout.mfabs.at(patch).at(level + 1);
            const int ncomps = ncomps_level_patch.at(level).at(patch);
            const int fine_comp0 = comp0_level_patch.at(level + 1).at(patch);
            const amrex::BoxArray &fine_boxarray = fine_mfab.boxarray;
            for (int component = 0; component < ncomps; ++component) {
              amrex::Box refined_box(mfab.box(component));
              refined_box.refine(2);
              std::vector<int> children;
              for (const auto &ib : fine_boxarray.intersections(refined_box))
                children.push_back(fine_comp0 + ib.first);
              segment_data.push_back(std::move(children));
            }
          } else {
            for (int component = 0; component < ncomps_level.at(level);
                 ++component)
              segment_data.emplace_back();
          }
        }
      }
      for (const auto &data : segment_data)
        num_children.push_back(data.size());
      put_groupelmap(metafile, childmaps_name, ncomps_total, segment_data);
    }

    {
      const int max_mgrtree_children = 2;
      const DB::ptr<DBmrgtree> mrgtree = DB::make(
          DBMakeMrgtree(DB_MULTIMESH, 0, max_mgrtree_children, nullptr));
      assert(mrgtree);

      const int max_amr_decomp_children = 2;
      int ierr =
          DBAddRegion(mrgtree.get(), "amr_decomp", 0, max_amr_decomp_children,
                      nullptr, 0, nullptr, nullptr, nullptr, nullptr);
      assert(!ierr);
      ierr = DBSetCwr(mrgtree.get(), "amr_decomp");
      assert(ierr >= 0);

      {
        ierr = DBAddRegion(mrgtree.get(), "levels", 0, nlevels, nullptr, 0,
                           nullptr, nullptr, nullptr, nullptr);
        assert(!ierr);
        ierr = DBSetCwr(mrgtree.get(), "levels");
        assert(ierr >= 0);

        const char *const region_name = "@level%d@n";
        std::vector<int> segment_ids;
        std::vector<int> segment_types(nlevels, DB_BLOCKCENT);
        for (int level = 0; level < nlevels; ++level)
          segment_ids.push_back(level);
        ierr = DBAddRegionArray(mrgtree.get(), nlevels, &region_name, 0,
                                levelmaps_name.c_str(), 1, segment_ids.data(),
                                ncomps_level.data(), segment_types.data(),
                                nullptr);
        assert(!ierr);
        ierr = DBSetCwr(mrgtree.get(), "..");
        assert(ierr >= 0);
      }

      {
        ierr = DBAddRegion(mrgtree.get(), "patches", 0, ncomps_total, nullptr,
                           0, nullptr, nullptr, nullptr, nullptr);
        assert(ierr >= 0);
        ierr = DBSetCwr(mrgtree.get(), "patches");
        assert(ierr >= 0);

        const char *const region_name = "@patch%d@n";
        std::vector<int> segment_ids;
        std::vector<int> segment_types(ncomps_total, DB_BLOCKCENT);
        for (int c = 0; c < ncomps_total; ++c)
          segment_ids.push_back(c);
        ierr = DBAddRegionArray(mrgtree.get(), ncomps_total, &region_name, 0,
                                childmaps_name.c_str(), 1, segment_ids.data(),
                                num_children.data(), segment_types.data(),
                                nullptr);
        ierr = DBSetCwr(mrgtree.get(), "..");
        assert(ierr >= 0);
      }

      {
        const std::vector<std::string> mrgv_onames{
            multimeshname + "_wmrgtree_lvlRatios",
            multimeshname + "_wmrgtree_ijkExts",
            multimeshname + "_wmrgtree_xyzExts", "rank"};
        std::vector<const char *> mrgv_oname_ptrs;
        for (const std::string &name : mrgv_onames)
          mrgv_oname_ptrs.push_back(name.c_str());
        mrgv_oname_ptrs.push_back(nullptr);

        const DB::ptr<DBoptlist> optlist = make_optlist();
        add_option(optlist.get(), DBOPT_MRGV_ONAMES, mrgv_oname_ptrs.data());
        ierr = DBPutMrgtree(metafile, "mrgTree", "amr_mesh", mrgtree.get(),
                            optlist.get());
        assert(!ierr);
      }
    }

    {
      const std::string levelrationame = multimeshname + "_wmrgtree_lvlRatios";
      const std::array<const char *, ndims> compnames{"iRatio", "jRatio",
                                                      "kRatio"};
      const char *const region_name = "@level%d@n";
      const std::vector<int> ratios(nlevels, 2);
      std::array<const void *, ndims> data_ptrs;
      for (int d = 0; d < ndims; ++d)
        data_ptrs[d] = ratios.data();
      const int ierr = DBPutMrgvar(
          metafile, levelrationame.c_str(), "mrgTree", ndims, compnames.data(),
          nlevels, &region_name, DB_INT, data_ptrs.data(), nullptr);
      assert(!ierr);
    }
  }

  void write_extents(const std::string &multimeshname,
                     const std::vector<std::array<ivect, 2> > &iextents,
                     const std::vector<extent_t> &extents) const {
    const int ncomps = iextents.size();
    const std::string iextentsname = multimeshname + "_wmrgtree_ijkExts";
    const std::string extentsname = multimeshname + "_wmrgtree_xyzExts";
    const std::array<const char *, 2 * ndims> icompnames{
        "iMin", "iMax", "jMin", "jMax", "kMin", "kMax"};
    const std::array<const char *, 2 * ndims> compnames{"xMin", "xMax", "yMin",
                                                        "yMax", "zMin", "zMax"};
    const char *const region_name = "@patch%d@n";

    std::array<std::array<std::vector<int>, 2>, ndims> idata;
    std::array<std::array<std::vector<CCTK_REAL>, 2>, ndims> data;
    for (int c = 0; c < ncomps; ++c)
      for (int d = 0; d < ndims; ++d)
        for (int f = 0; f < 2; ++f) {
          idata[d][f].push_back(iextents[c][f][d]);
          data[d][f].push_back(extents[c][f][d]);
        }
    std::array<std::array<const void *, 2>, ndims> idata_ptrs;
    std::array<std::array<const void *, 2>, ndims> data_ptrs;
    for (int d = 0; d < ndims; ++d)
      for (int f = 0; f < 2; ++f) {
        idata_ptrs[d][f] = idata[d][f].data();
        data_ptrs[d][f] = data[d][f].data();
      }

    int ierr = DBPutMrgvar(metafile, iextentsname.c_str(), "mrgTree", 2 * ndims,
                           icompnames.data(), ncomps, &region_name, DB_INT,
                           idata_ptrs.data(), nullptr);
    assert(!ierr);
    ierr = DBPutMrgvar(metafile, extentsname.c_str(), "mrgTree", 2 * ndims,
                       compnames.data(), ncomps, &region_name,
                       db_datatype_v<CCTK_REAL>, data_ptrs.data(), nullptr);
    assert(!ierr);

    const std::vector<int> ranks(ncomps, ndims);
    const void *const rank_ptr = ranks.data();
    ierr = DBPutMrgvar(metafile, "rank", "mrgTree", 1, nullptr, ncomps,
                       &region_name, DB_INT, &rank_ptr, nullptr);
    assert(!ierr);
  }

  void write_multimesh(const group_layout_t &layout) const {
    const amrex::MultiFab &mfab0 = *layout.mfabs.at(0).at(0);
    const amrex::IndexType &indextype = mfab0.ixType();
    const ivect nghosts = to_ivect(mfab0.nGrowVect());
    const std::string multimeshname = make_meshname(nghosts);

    write_mrgtree(multimeshname, layout);

    std::vector<std::array<ivect, 2> > iextents;
    std::vector<extent_t> extents;
    std::vector<std::string> meshnames;
    std::vector<int> zonecounts;
    for (int patch = 0; patch < num_patches(); ++patch) {
      for (int lev = 0; lev < num_levels(patch); ++lev) {
        const level_t level{patch_t{patch}, lev};
        const level_record_t &levelrecord = record.levels.at(patch).at(lev);
        const amrex::MultiFab &mfab = *layout.mfabs.at(patch).at(lev);
        for (int c = 0; c < mfab.size(); ++c) {
          const amrex::Box &fabbox = mfab.fabbox(c);
          std::array<ivect, 2> iextent;
          extent_t extent;
          for (int d = 0; d < ndims; ++d) {
            iextent[0][d] = fabbox.smallEnd(d);
            iextent[1][d] = fabbox.bigEnd(d);
            if (is_cartesian(patch)) {
              const amrex::Geometry &geom = levelrecord.geom;
              extent[0][d] =
                  geom.ProbLo(d) + fabbox.smallEnd(d) * geom.CellSize(d);
              extent[1][d] =
                  geom.ProbLo(d) + fabbox.bigEnd(d) * geom.CellSize(d);
            } else {
              extent[0][d] = coordinate_minima.at(patch).at(lev).at(c)[d];
              extent[1][d] = coordinate_maxima.at(patch).at(lev).at(c)[d];
            }
          }
          iextents.push_back(iextent);
          extents.push_back(extent);

          meshnames.push_back(block_path(mfab, c) + ":" +
                              make_meshname(nghosts, level, c));

          int zonecount = 1;
          for (int d = 0; d < ndims; ++d)
            zonecount *= fabbox.length(d) + int(indextype.cellCentered(d));
          zonecounts.push_back(zonecount);
        }
      }
    }

    write_extents(multimeshname, iextents, extents);

    std::vector<const char *> meshname_ptrs;
    for (const std::string &meshname : meshnames)
      meshname_ptrs.push_back(meshname.c_str());

    std::vector<std::array<std::array<double, ndims>, 2> > dextents;
    for (const extent_t &extent : extents) {
      std::array<std::array<double, ndims>, 2> dextent;
      for (int f = 0; f < 2; ++f)
        for (int d = 0; d < ndims; ++d)
          dextent[f][d] = extent[f][d];
      dextents.push_back(dextent);
    }

    const DB::ptr<DBoptlist> optlist = make_optlist();
    int cycle = record.iteration;
    add_option(optlist.get(), DBOPT_CYCLE, &cycle);
    double dtime = record.time;
    add_option(optlist.get(), DBOPT_DTIME, &dtime);
    int quadmesh = DB_QUADMESH;
    add_option(optlist.get(), DBOPT_MB_BLOCK_TYPE, &quadmesh);
    int extents_size = 2 * ndims;
    add_option(optlist.get(), DBOPT_EXTENTS_SIZE, &extents_size);
    add_option(optlist.get(), DBOPT_EXTENTS, dextents.data());
    add_option(optlist.get(), DBOPT_ZONECOUNTS, zonecounts.data());
    std::string mrgtreename = "mrgtree";
    add_option(optlist.get(), DBOPT_MRGTREE_NAME, mrgtreename.data());

    const int ierr =
        DBPutMultimesh(metafile, multimeshname.c_str(), meshname_ptrs.size(),
                       meshname_ptrs.data(), nullptr, optlist.get());
    assert(!ierr);
  }

  void write_multivars(const group_layout_t &layout) const {
    const amrex::MultiFab &mfab0 = *layout.mfabs.at(0).at(0);
    std::string multimeshname = make_meshname(to_ivect(mfab0.nGrowVect()));

    const DB::ptr<DBoptlist> optlist = make_optlist();
    int cycle = record.iteration;
    add_option(optlist.get(), DBOPT_CYCLE, &cycle);
    double dtime = record.time;
    add_option(optlist.get(), DBOPT_DTIME, &dtime);
    add_option(optlist.get(), DBOPT_MMESH_NAME, multimeshname.data());
    int vartype_scalar = DB_VARTYPE_SCALAR;
    add_option(optlist.get(), DBOPT_TENSOR_RANK, &vartype_scalar);
    int quadvar = DB_QUADVAR;
    add_option(optlist.get(), DBOPT_MB_BLOCK_TYPE, &quadvar);

    for (const std::string &var : layout.vars) {
      std::vector<std::string> varnames;
      for (int patch = 0; patch < num_patches(); ++patch)
        for (int lev = 0; lev < num_levels(patch); ++lev) {
          const level_t level{patch_t{patch}, lev};
          const amrex::MultiFab &mfab = *layout.mfabs.at(patch).at(lev);
          for (int c = 0; c < mfab.size(); ++c)
            varnames.push_back(block_path(mfab, c) + ":" +
                               make_varname(var, level, c));
        }
      std::vector<const char *> varname_ptrs;
      for (const std::string &varname : varnames)
        varname_ptrs.push_back(varname.c_str());
      const int ierr = DBPutMultivar(metafile, make_varname(var).c_str(),
                                     varname_ptrs.size(), varname_ptrs.data(),
                                     nullptr, optlist.get());
      assert(!ierr);
    }
  }

  void write_driver_state() const {
    const std::string dirname = DB::legalize_name(driver_name);
    int ierr = DBMkDir(metafile, dirname.c_str());
    assert(!ierr);

    {
      const int npatches = num_patches();
      std::vector<int> nlevels(npatches);
      for (int patch = 0; patch < npatches; ++patch)
        nlevels.at(patch) = num_levels(patch);
      const std::string varname = dirname + "/" + DB::legalize_name("nlevels");
      ierr = DBWrite(metafile, varname.c_str(), nlevels.data(), &npatches, 1,
                     DB_INT);
      assert(!ierr);
    }

    for (int patch = 0; patch < num_patches(); ++patch)
      for (int lev = 0; lev < num_levels(patch); ++lev) {
        const level_t level{patch_t{patch}, lev};
        const std::vector<amrex::Box> &levelboxes =
            record.levels.at(patch).at(lev).boxes;
        const int ncomponents = levelboxes.size();
        std::vector<int> boxes(2 * ndims * ncomponents);
        for (int component = 0; component < ncomponents; ++component) {
          const amrex::Box &box = levelboxes.at(component);
          for (int d = 0; d < ndims; ++d) {
            boxes[d + 2 * ndims * component] = box.smallEnd(d);
            boxes[d + ndims + 2 * ndims * component] = box.bigEnd(d);
          }
        }
        const int dims[2] = {ncomponents, 2 * ndims};
        const std::string varname =
            dirname + "/" + make_fabarraybasename(level);
        ierr =
            DBWrite(metafile, varname.c_str(), boxes.data(), dims, 2, DB_INT);
        assert(!ierr);
      }

    const auto write_clock = [&](const std::string &what, const level_t level,
                                 long long value) {
      const int dims = 1;
      const int ierr = DBWrite(metafile, clock_varname(what, level).c_str(),
                               &value, &dims, 1, DB_LONG_LONG);
      assert(!ierr);
    };
    for (int patch = 0; patch < num_patches(); ++patch)
      for (int lev = 0; lev < num_levels(patch); ++lev) {
        const level_t level{patch_t{patch}, lev};
        const level_record_t &levelrecord = record.levels.at(patch).at(lev);
        if (levelrecord.iteration) {
          write_clock("iteration_num", level, levelrecord.iteration->num);
          write_clock("iteration_den", level, levelrecord.iteration->den);
        }
        if (levelrecord.last_step_iteration)
          write_clock("last_step_iteration", level,
                      *levelrecord.last_step_iteration);
      }
  }

  void write(const std::string &dir) const {
    write_string(metafile, "AllParameters", record.parameters);
    write_metadata_is_time_varying(metafile);

    const std::vector<group_layout_t> layouts = group_layouts(record);
    std::set<ivect> multimeshes;
    for (const group_layout_t &layout : layouts) {
      const ivect nghosts = to_ivect(layout.mfabs.at(0).at(0)->nGrowVect());
      if (multimeshes.insert(nghosts).second)
        write_multimesh(layout);
      write_multivars(layout);
    }

    write_driver_state();

    std::ofstream visit(dir + "/" + file + ".silo.visit", std::ios::app);
    assert(visit.good());
    visit << make_filename(file, record.iteration) << "\n";
  }
};

std::set<std::string> union_over_ranks(const std::set<std::string> &local) {
  const int myproc = CCTK_MyProc(nullptr);
  const int nprocs = CCTK_nProcs(nullptr);
  const int root = 0;

  std::string joined;
  for (const std::string &name : local)
    joined += name + "\n";
  const int length = joined.size();
  std::vector<int> lengths(nprocs);
  MPI_Gather(&length, 1, MPI_INT, lengths.data(), 1, MPI_INT, root, mpi_comm);
  std::vector<int> offsets(nprocs, 0);
  std::string gathered;
  if (myproc == root) {
    for (int proc = 1; proc < nprocs; ++proc)
      offsets.at(proc) = offsets.at(proc - 1) + lengths.at(proc - 1);
    gathered.resize(offsets.back() + lengths.back());
  }
  MPI_Gatherv(joined.data(), length, MPI_CHAR, gathered.data(), lengths.data(),
              offsets.data(), MPI_CHAR, root, mpi_comm);

  std::set<std::string> names;
  std::string unified;
  if (myproc == root) {
    std::istringstream buf(gathered);
    for (std::string name; std::getline(buf, name);)
      names.insert(name);
    for (const std::string &name : names)
      unified += name + "\n";
  }
  int unified_length = unified.size();
  MPI_Bcast(&unified_length, 1, MPI_INT, root, mpi_comm);
  unified.resize(unified_length);
  MPI_Bcast(unified.data(), unified_length, MPI_CHAR, root, mpi_comm);
  if (myproc != root) {
    std::istringstream buf(unified);
    for (std::string name; std::getline(buf, name);)
      names.insert(name);
  }
  return names;
}

std::set<std::string> local_inventory(DBfile *const dbfile) {
  std::set<std::string> names;
  const DBtoc *const toc = DBGetToc(dbfile);
  assert(toc);
  for (int i = 0; i < toc->nqvar; ++i)
    names.insert(without_component(toc->qvar_names[i]));
  return names;
}

std::string block_file_path(const std::string &dir, const std::string &file,
                            const int iteration, const int ioserver) {
  return dir + "/" + make_subdirname(file, iteration) + "/" +
         make_filename(file, iteration, ioserver);
}

DB::ptr<DBfile> open_block_file(const std::string &filename) {
  if (!std::filesystem::exists(filename))
    CCTK_VERROR("The Silo checkpoint file \"%s\" does not exist. A Silo "
                "checkpoint recovers only with the number of processes and the "
                "IO::out_mode it was written with.",
                filename.c_str());
  DB::ptr<DBfile> dbfile = DB::make(DBOpen(filename.c_str(), DB_HDF5, DB_READ));
  if (!dbfile)
    CCTK_VERROR("Could not open the Silo checkpoint file \"%s\"",
                filename.c_str());
  return dbfile;
}

struct reader_t {
  const std::string &filename;
  int myproc, every;
  DB::ptr<DBfile> dbfile;
  std::map<std::string, const amrex::MultiFab *> states;

  void read(const buffer_id_t &id, amrex::MultiFab &mfab) const {
    const level_t level = *id.level;
    const std::string *const tag = band_tag(id);
    const int numvars = id.vars.size();
    assert(mfab.nComp() == numvars);
    if (tag)
      assert(mfab.nGrowVect() == 0);
    const amrex::IndexType &indextype =
        tag ? state_of(states, id).ixType() : mfab.ixType();
    const int centering = db_centering(indextype);
    const int mpi_tag = tag ? band_read_tag : state_read_tag;
    const amrex::DistributionMapping &dm = mfab.DistributionMap();

    for (int component = 0; component < dm.size(); ++component) {
      const int proc = dm[component];
      const int ioproc = proc / every * every;
      const bool recv_this_fab = proc == myproc;
      const bool read_this_fab = ioproc == myproc;
      if (!(recv_this_fab || read_this_fab))
        continue;

      const ivect dims = shape(mfab.fabbox(component));
      const std::ptrdiff_t zonecount = num_points(dims);
      assert(numvars * zonecount <= INT_MAX);

      std::vector<CCTK_REAL> buffer;
      MPI_Request mpi_req;
      CCTK_REAL *data = nullptr;
      if (recv_this_fab && read_this_fab) {
        data = mfab[component].dataPtr();
      } else if (recv_this_fab) {
        MPI_Irecv(mfab[component].dataPtr(), numvars * zonecount,
                  mpi_datatype_v<CCTK_REAL>, ioproc, mpi_tag, mpi_comm,
                  &mpi_req);
      } else {
        buffer.resize(numvars * zonecount);
        data = buffer.data();
      }

      if (read_this_fab) {
        for (int vi = 0; vi < numvars; ++vi) {
          const std::string varname = make_varname(
              id.vars.at(vi), level, component, tag ? *tag : std::string());
          if (io_verbose)
            CCTK_VINFO("Reading variable %s", varname.c_str());
          const DB::ptr<DBquadvar> quadvar =
              DB::make(DBGetQuadvar(dbfile.get(), varname.c_str()));
          if (!quadvar)
            CCTK_VERROR("The Silo checkpoint file \"%s\" lacks the variable "
                        "\"%s\" of %s. A Silo checkpoint recovers only with "
                        "the number of processes and the IO::out_mode it was "
                        "written with.",
                        filename.c_str(), varname.c_str(), id.name().c_str());
          assert(quadvar->ndims == ndims);
          for (int d = 0; d < ndims; ++d)
            assert(quadvar->dims[d] == dims[d]);
          assert(quadvar->datatype == db_datatype_v<CCTK_REAL>);
          assert(quadvar->centering == centering);
          assert(quadvar->nvals == 1);
          assert(quadvar->major_order == 0);
          std::memcpy(data + vi * zonecount, quadvar->vals[0],
                      zonecount * sizeof(CCTK_REAL));
        }
      }

      if (recv_this_fab && !read_this_fab)
        MPI_Wait(&mpi_req, MPI_STATUS_IGNORE);
      else if (!recv_this_fab)
        MPI_Send(buffer.data(), numvars * zonecount, mpi_datatype_v<CCTK_REAL>,
                 proc, mpi_tag, mpi_comm);
    }
  }
};

struct silo_codec_t final : checkpoint_codec_t {
  codec_limits_t limits() const override {
    return {.carries_arrays = false,
            .max_timelevels = 1,
            .rechops_on_recovery = false};
  }
  std::string key(const buffer_id_t &id) const override {
    const std::string *const tag = band_tag(id);
    return without_component(make_varname(id.vars.at(0), id.level,
                                          id.level ? 0 : -1,
                                          tag ? *tag : std::string()));
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

int silo_codec_t::read_parameters(const std::string &dir,
                                  const std::string &file) {
  assert(!dir.empty());
  assert(!file.empty());

  static Timer timer("silo_codec_t::read_parameters");
  Interval interval(timer);

  const int myproc = CCTK_MyProc(nullptr);
  const int metafile_ioproc = 0;
  const bool read_metafile = myproc == metafile_ioproc;

  DBShowErrors(DB_ALL_AND_DRVR, nullptr);
  DBSetAllowEmptyObjects(1);

  int iteration = -1;
  if (read_metafile) {
    try {
      for (const auto &direntry : std::filesystem::directory_iterator(dir))
        iteration = std::max(
            iteration, match_filename(direntry.path().filename().string()));
    } catch (const std::filesystem::filesystem_error &) {
    }
  }
  MPI_Bcast(&iteration, 1, MPI_INT, metafile_ioproc, mpi_comm);

  if (iteration < 0) {
    if (io_verbose) {
      CCTK_VINFO("Not recovering parameters:");
      CCTK_VINFO("  Could not find a Silo checkpoint file \"%s/%s.*.silo\"",
                 dir.c_str(), file.c_str());
    }
    return -1;
  }

  std::string parameters;
  if (read_metafile) {
    CCTK_VINFO("Recovering parameters from checkpoint file \"%s\" iteration %d",
               make_filename(dir + "/" + file, iteration).c_str(), iteration);
    const std::string metafilename = dir + "/" + make_filename(file, iteration);
    const DB::ptr<DBfile> metafile =
        DB::make(DBOpen(metafilename.c_str(), DB_HDF5, DB_READ));
    if (!metafile)
      CCTK_VERROR("Could not open the Silo checkpoint file \"%s\"",
                  metafilename.c_str());

    const int type = DBGetVarType(metafile.get(), "AllParameters");
    assert(type == DB_CHAR);
    int length;
    const int varndims =
        DBGetVarDims(metafile.get(), "AllParameters", 1, &length);
    assert(varndims == 1);
    char *const data =
        static_cast<char *>(DBGetVar(metafile.get(), "AllParameters"));
    assert(data);
    parameters = std::string(data, length);
    std::free(data);
  }

  int length = parameters.size();
  MPI_Bcast(&length, 1, MPI_INT, metafile_ioproc, mpi_comm);
  parameters.resize(length);
  MPI_Bcast(parameters.data(), length, MPI_CHAR, metafile_ioproc, mpi_comm);

  IOUtil_SetAllParameters(parameters.data());

  return iteration;
}

void silo_codec_t::encode(const checkpoint_record_t &record,
                          const std::string &dir, const std::string &file) {
  DECLARE_CCTK_PARAMETERS;

  static Timer timer("silo_codec_t::encode");
  Interval interval(timer);

  if (record.buffers.empty())
    return;

  const int myproc = CCTK_MyProc(nullptr);
  const int nprocs = CCTK_nProcs(nullptr);
  const int every = ioproc_every(nprocs);
  const bool write_file = myproc % every == 0;
  const int metafile_ioproc = nprocs == 1 || every == 1 ? 0 : 1;

  if (io_verbose)
    CCTK_VINFO("Writing Silo checkpoint \"%s/%s\" iteration %d...", dir.c_str(),
               make_filename(file, record.iteration).c_str(), record.iteration);

  DBShowErrors(DB_ALL_AND_DRVR, nullptr);
  DBSetCompression(out_silo_compression_options);
  DBSetEnableChecksums(1);

  const std::string pathname =
      dir + "/" + make_subdirname(file, record.iteration);
  int ierr = CCTK_CreateDirectory(0755, dir.c_str());
  assert(ierr >= 0);
  ierr = CCTK_CreateDirectory(0755, pathname.c_str());
  assert(ierr >= 0);

  writer_t writer{.record = record,
                  .myproc = myproc,
                  .every = every,
                  .metafile_ioproc = metafile_ioproc,
                  .dbfile = {},
                  .states = state_buffers(record),
                  .meshes = {},
                  .coordinate_minima = {},
                  .coordinate_maxima = {}};
  writer.coordinate_minima.resize(record.levels.size());
  writer.coordinate_maxima.resize(record.levels.size());
  for (std::size_t patch = 0; patch < record.levels.size(); ++patch) {
    const std::vector<level_record_t> &levels = record.levels.at(patch);
    if (levels.at(0).vertex_coords == nullptr)
      continue;
    writer.coordinate_minima.at(patch).resize(levels.size());
    writer.coordinate_maxima.at(patch).resize(levels.size());
    for (std::size_t level = 0; level < levels.size(); ++level) {
      writer.coordinate_minima.at(patch).at(level).resize(
          levels.at(level).boxes.size());
      writer.coordinate_maxima.at(patch).at(level).resize(
          levels.at(level).boxes.size());
    }
  }

  if (write_file) {
    const std::string filename =
        pathname + "/" + make_filename(file, record.iteration, myproc / every);
    writer.dbfile = DB::make(DBCreate(filename.c_str(), DB_CLOBBER, DB_LOCAL,
                                      file.c_str(), DB_HDF5));
    assert(writer.dbfile);
    write_string(writer.dbfile.get(), "AllParameters", record.parameters);
    write_metadata_is_time_varying(writer.dbfile.get());
  }

  for (const buffer_t &buffer : record.buffers)
    std::visit(overloaded{[&](const grid_buffer_t &grid) {
                            if (band_tag(buffer.id))
                              writer.write_band(buffer.id, *grid.mfab);
                            else
                              writer.write_state(buffer.id, *grid.mfab);
                          },
                          [&](const array_buffer_t &) {
                            CCTK_VERROR("A Silo checkpoint cannot carry the "
                                        "grid array or scalar %s",
                                        buffer.id.group.c_str());
                          }},
               buffer.data);
  writer.dbfile.reset();

  writer.reduce_coordinate_extents();

  if (myproc == metafile_ioproc) {
    const std::string metafilename =
        dir + "/" + make_filename(file, record.iteration);
    const DB::ptr<DBfile> metafile = DB::make(DBCreate(
        metafilename.c_str(), DB_CLOBBER, DB_LOCAL, file.c_str(), DB_HDF5));
    assert(metafile);
    const metafile_writer_t metafile_writer{
        .record = record,
        .file = file,
        .every = every,
        .metafile = metafile.get(),
        .coordinate_minima = writer.coordinate_minima,
        .coordinate_maxima = writer.coordinate_maxima};
    metafile_writer.write(dir);
  }

  if (record.purpose == record_purpose::output)
    OutputMeta_RegisterOutputFile(silo_output_description(
        dir + "/" + make_filename(file, record.iteration),
        "3D CarpetX HDF5 Silo output", output_variables(record),
        record.iteration, {0, 1, 2}));
}

checkpoint_record_t silo_codec_t::decode_structure(const std::string &dir,
                                                   const std::string &file,
                                                   const int iteration) {
  assert(!dir.empty());
  assert(!file.empty());
  assert(iteration >= 0);

  static Timer timer("silo_codec_t::decode_structure");
  Interval interval(timer);

  const int myproc = CCTK_MyProc(nullptr);
  const int nprocs = CCTK_nProcs(nullptr);
  const int metafile_ioproc = 0;
  const bool read_metafile = myproc == metafile_ioproc;

  DBShowErrors(DB_ALL_AND_DRVR, nullptr);

  DB::ptr<DBfile> metafile;
  if (read_metafile) {
    const std::string metafilename = dir + "/" + make_filename(file, iteration);
    metafile = DB::make(DBOpen(metafilename.c_str(), DB_HDF5, DB_READ));
    if (!metafile)
      CCTK_VERROR("Could not open the Silo checkpoint file \"%s\"",
                  metafilename.c_str());
  }

  checkpoint_record_t record;
  record.delta_time = 0;

  if (read_metafile) {
    assert(DBGetVarType(metafile.get(), "cycle") == DB_INT);
    int length;
    const int varndims = DBGetVarDims(metafile.get(), "cycle", 1, &length);
    assert(varndims == 1 && length == 1);
    const int ierr = DBReadVar(metafile.get(), "cycle", &record.iteration);
    assert(!ierr);
  }
  MPI_Bcast(&record.iteration, 1, MPI_INT, metafile_ioproc, mpi_comm);

  double dtime;
  if (read_metafile) {
    assert(DBGetVarType(metafile.get(), "dtime") == DB_DOUBLE);
    int length;
    const int varndims = DBGetVarDims(metafile.get(), "dtime", 1, &length);
    assert(varndims == 1 && length == 1);
    const int ierr = DBReadVar(metafile.get(), "dtime", &dtime);
    assert(!ierr);
  }
  MPI_Bcast(&dtime, 1, MPI_DOUBLE, metafile_ioproc, mpi_comm);
  record.time = dtime;

  const std::string dirname = DB::legalize_name(driver_name);

  int npatches;
  std::vector<int> nlevels;
  if (read_metafile) {
    const std::string varname = dirname + "/" + DB::legalize_name("nlevels");
    assert(DBGetVarType(metafile.get(), varname.c_str()) == DB_INT);
    npatches = DBGetVarLength(metafile.get(), varname.c_str());
    assert(npatches >= 0);
    nlevels.resize(npatches);
    const int ierr = DBReadVar(metafile.get(), varname.c_str(), nlevels.data());
    assert(!ierr);
  }
  MPI_Bcast(&npatches, 1, MPI_INT, metafile_ioproc, mpi_comm);
  nlevels.resize(npatches);
  MPI_Bcast(nlevels.data(), npatches, MPI_INT, metafile_ioproc, mpi_comm);
  if (io_verbose)
    CCTK_VINFO("Found %d patches", npatches);

  record.levels.resize(npatches);
  for (int patch = 0; patch < npatches; ++patch) {
    if (io_verbose)
      CCTK_VINFO("  Found %d levels on patch %d", nlevels.at(patch), patch);
    record.levels.at(patch).resize(nlevels.at(patch));
    for (int lev = 0; lev < nlevels.at(patch); ++lev) {
      const level_t level{patch_t{patch}, lev};
      level_record_t &levelrecord = record.levels.at(patch).at(lev);

      const std::string varname = dirname + "/" + make_fabarraybasename(level);
      int nfabs;
      if (read_metafile) {
        assert(DBGetVarType(metafile.get(), varname.c_str()) == DB_INT);
        int vardims[2];
        const int varndims =
            DBGetVarDims(metafile.get(), varname.c_str(), 2, vardims);
        assert(varndims == 2);
        assert(vardims[1] == 2 * ndims);
        nfabs = vardims[0];
        assert(nfabs >= 0);
      }
      MPI_Bcast(&nfabs, 1, MPI_INT, metafile_ioproc, mpi_comm);

      std::vector<int> data(2 * ndims * nfabs);
      if (read_metafile) {
        const int ierr =
            DBReadVar(metafile.get(), varname.c_str(), data.data());
        assert(!ierr);
      }
      MPI_Bcast(data.data(), data.size(), MPI_INT, metafile_ioproc, mpi_comm);
      for (int component = 0; component < nfabs; ++component) {
        const amrex::IntVect small(&data.at(2 * ndims * component));
        const amrex::IntVect big(&data.at(ndims + 2 * ndims * component));
        levelrecord.boxes.push_back(amrex::Box(small, big));
      }

      const auto read_clock =
          [&](const std::string &what) -> std::optional<long long> {
        const std::string clockname = clock_varname(what, level);
        long long value = 0;
        bool exists = false;
        if (read_metafile &&
            DBInqVarExists(metafile.get(), clockname.c_str())) {
          const int ierr = DBReadVar(metafile.get(), clockname.c_str(), &value);
          assert(!ierr);
          exists = true;
        }
        MPI_Bcast(&exists, 1, MPI_C_BOOL, metafile_ioproc, mpi_comm);
        if (!exists)
          return std::nullopt;
        MPI_Bcast(&value, 1, MPI_LONG_LONG, metafile_ioproc, mpi_comm);
        return value;
      };
      const std::optional<long long> num = read_clock("iteration_num");
      const std::optional<long long> den = read_clock("iteration_den");
      if (num && den)
        levelrecord.iteration = rat64(*num, *den);
      if (const std::optional<long long> last_step =
              read_clock("last_step_iteration"))
        levelrecord.last_step_iteration = int(*last_step);
    }
  }
  metafile.reset();

  const int every = ioproc_every(nprocs);
  std::set<std::string> local;
  if (myproc % every == 0) {
    const DB::ptr<DBfile> dbfile =
        open_block_file(block_file_path(dir, file, iteration, myproc / every));
    local = local_inventory(dbfile.get());
  }
  record.present = union_over_ranks(local);

  return record;
}

void silo_codec_t::decode_data(const checkpoint_record_t &record,
                               const std::string &dir,
                               const std::string &file) {
  static Timer timer("silo_codec_t::decode_data");
  Interval interval(timer);

  const int myproc = CCTK_MyProc(nullptr);
  const int nprocs = CCTK_nProcs(nullptr);
  const int every = ioproc_every(nprocs);

  DBShowErrors(DB_ALL_AND_DRVR, nullptr);

  const std::string filename =
      block_file_path(dir, file, record.iteration, myproc / every);
  reader_t reader{.filename = filename,
                  .myproc = myproc,
                  .every = every,
                  .dbfile = {},
                  .states = state_buffers(record)};
  if (myproc % every == 0)
    reader.dbfile = open_block_file(filename);

  for (const buffer_t &buffer : record.buffers)
    std::visit(overloaded{[&](const grid_buffer_t &grid) {
                            reader.read(buffer.id, *grid.mfab);
                          },
                          [&](const array_buffer_t &) {
                            CCTK_VERROR("A Silo checkpoint cannot carry the "
                                        "grid array or scalar %s",
                                        buffer.id.group.c_str());
                          }},
               buffer.data);
}

} // namespace

std::unique_ptr<checkpoint_codec_t> make_silo_codec() {
  return std::make_unique<silo_codec_t>();
}

} // namespace CarpetX

#endif
