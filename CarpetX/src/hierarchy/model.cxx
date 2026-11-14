#include "config/boundaries.hxx"
#include "config/groups.hxx"
#include "config/stages.hxx"
#include "hierarchy/binding.hxx"
#include "hierarchy/model.hxx"
#include "hierarchy/valid.hxx"

#include "hierarchy/loops.hxx"
#include <carpetx.hxx>

#include <cctk.h>
#include <cctk_Arguments.h>
#include <cctk_Parameters.h>
#include <util_Table.h>

#include <AMReX.H>
#include <AMReX_FillPatchUtil.H>
#include <AMReX_iMultiFab.H>

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <optional>
#include <ostream>
#include <source_location>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace CarpetX {

namespace {
GHExt *ghext = nullptr;

GHExt &existing_hierarchy() {
  if (!ghext)
    CCTK_ERROR("The CarpetX grid hierarchy does not exist: the public API is "
               "available between the driver's SetupGH and its Shutdown");
  return *ghext;
}

cGH *copy_cctkGH(const cGH *restrict const sourceGH) {
  cGH *restrict const cctkGH = new cGH;

  *cctkGH = *sourceGH;

  const auto copy_array = [](const auto *restrict const srcptr, const int sz) {
    using T = std::decay_t<decltype(*srcptr)>;
    T *restrict const ptr = new T[sz];
    std::copy(srcptr, srcptr + sz, ptr);
    return ptr;
  };
  cctkGH->cctk_gsh = copy_array(sourceGH->cctk_gsh, dim);
  cctkGH->cctk_lsh = copy_array(sourceGH->cctk_lsh, dim);
  cctkGH->cctk_lbnd = copy_array(sourceGH->cctk_lbnd, dim);
  cctkGH->cctk_ubnd = copy_array(sourceGH->cctk_ubnd, dim);
  cctkGH->cctk_tile_min = copy_array(sourceGH->cctk_tile_min, dim);
  cctkGH->cctk_tile_max = copy_array(sourceGH->cctk_tile_max, dim);
  cctkGH->cctk_ash = copy_array(sourceGH->cctk_ash, dim);
  cctkGH->cctk_to = copy_array(sourceGH->cctk_to, dim);
  cctkGH->cctk_from = copy_array(sourceGH->cctk_from, dim);
  cctkGH->cctk_delta_space = copy_array(sourceGH->cctk_delta_space, dim);
  cctkGH->cctk_origin_space = copy_array(sourceGH->cctk_origin_space, dim);
  cctkGH->cctk_bbox = copy_array(sourceGH->cctk_bbox, 2 * dim);
  cctkGH->cctk_levfac = copy_array(sourceGH->cctk_levfac, dim);
  cctkGH->cctk_levoff = copy_array(sourceGH->cctk_levoff, dim);
  cctkGH->cctk_levoffdenom = copy_array(sourceGH->cctk_levoffdenom, dim);
  cctkGH->cctk_nghostzones = copy_array(sourceGH->cctk_nghostzones, dim);

  const int numvars = CCTK_NumVars();
  cctkGH->data = new void **[numvars];
  for (const var_t var : index_range<var_t>(var_t{0}, var_t{numvars}))
    cctkGH->data[var.index()] = copy_array(
        sourceGH->data[var.index()], CCTK_DeclaredTimeLevelsVI(var.index()));

  return cctkGH;
}

void delete_cctkGH(cGH *cctkGH) {
  delete[] cctkGH->cctk_gsh;
  delete[] cctkGH->cctk_lsh;
  delete[] cctkGH->cctk_lbnd;
  delete[] cctkGH->cctk_ubnd;
  delete[] cctkGH->cctk_tile_min;
  delete[] cctkGH->cctk_tile_max;
  delete[] cctkGH->cctk_ash;
  delete[] cctkGH->cctk_to;
  delete[] cctkGH->cctk_from;
  delete[] cctkGH->cctk_delta_space;
  delete[] cctkGH->cctk_origin_space;
  delete[] cctkGH->cctk_bbox;
  delete[] cctkGH->cctk_levfac;
  delete[] cctkGH->cctk_levoff;
  delete[] cctkGH->cctk_levoffdenom;
  delete[] cctkGH->cctk_nghostzones;
  const int numvars = CCTK_NumVars();
  for (const var_t var : index_range<var_t>(var_t{0}, var_t{numvars}))
    delete[] cctkGH->data[var.index()];
  delete[] cctkGH->data;
#ifdef CCTK_DEBUG
  memset(cctkGH, 0, sizeof *cctkGH);
#endif
  delete cctkGH;
}
} // namespace

GHExt &GHExt::mutable_instance() { return existing_hierarchy(); }

const GHExt &driver() { return existing_hierarchy(); }

int current_iteration() {
  const cGH *const cctkGH = driver().get_global_cctkGH();
  return cctkGH ? cctkGH->cctk_iteration : 0;
}

void GHExt::open_door(GHExt &hierarchy) {
  if (ghext)
    CCTK_ERROR("The CarpetX grid hierarchy already exists: SetupGH creates it "
               "once per run");
  ghext = &hierarchy;
}

void GHExt::close_door(const GHExt &hierarchy) {
  assert(ghext == &hierarchy);
  ghext = nullptr;
}

void assert_hierarchy_destroyed() { assert(!ghext); }

GHExt::GHExt(const config_t config)
    : use_subcycling(config.use_subcycling), do_reflux(config.do_reflux),
      pending_timelevels_(std::in_place, CCTK_NumGroups(), 0),
      make_core_(config.make_core), make_buffers_(config.make_buffers) {
  assert(make_core_);
  assert(make_buffers_);
}

void GHExt::mark_initialized() {
  assert(!initialized_);
  initialized_ = true;
}

GHExt::PatchData &GHExt::add_patch(level_writer_t &levels,
                                   const patch_t patch) {
  assert(patch.index() == num_patches());
  return patchdata.emplace_back(levels, patch, make_core_);
}

////////////////////////////////////////////////////////////////////////////////

GHExt::cctkGHptr &GHExt::cctkGHptr::operator=(cctkGHptr &&ptr) {
  if (cctkGH)
    delete_cctkGH(cctkGH);
  cctkGH = ptr.cctkGH;
  ptr.cctkGH = nullptr;
  return *this;
}

GHExt::cctkGHptr GHExt::cctkGHptr::copy_of(const cGH *const source) {
  cctkGHptr owned;
  owned.cctkGH = copy_cctkGH(source);
  return owned;
}

GHExt::cctkGHptr::~cctkGHptr() {
  if (cctkGH)
    delete_cctkGH(cctkGH);
  cctkGH = nullptr;
}

GHExt::PatchData::PatchData(level_writer_t &levels, const patch_t patch,
                            const amrcore_factory make_core)
    : patch(patch) {
  DECLARE_CCTK_PARAMETERS;

  // Domain
  amrex::RealBox domain({xmin, ymin, zmin}, {xmax, ymax, zmax});

  // Number of coarse grid cells
  amrex::Vector<int> ncells{ncells_x, ncells_y, ncells_z};

  is_cartesian = true;

  if (CCTK_IsFunctionAliased("MultiPatch_GetPatchSpecification2")) {
    CCTK_INT is_cartesian1;
    CCTK_INT ncells1[dim];
    CCTK_REAL xmin1[dim], xmax1[dim];
    const int ierr = MultiPatch_GetPatchSpecification2(
        patch.index(), &is_cartesian1, dim, ncells1, xmin1, xmax1);
    assert(!ierr);
    is_cartesian = is_cartesian1;
    for (int d = 0; d < dim; ++d)
      ncells[d] = ncells1[d];
    domain = amrex::RealBox(xmin1, xmax1);
  }

  symmetries = get_symmetries(patch);

  {
    const std::array faces{"lower", "upper"};
    const std::array dirs{"x", "y", "z"};
    const auto boundaries = get_default_boundaries();
    std::ostringstream buf;
    buf << "\nSymmetries:";
    for (int f = 0; f < 2; ++f)
      for (int d = 0; d < dim; ++d)
        buf << "\n  " << faces[f] << " " << dirs[d] << ": " << symmetries[f][d];
    buf << "\nDefault boundaries:";
    for (int f = 0; f < 2; ++f)
      for (int d = 0; d < dim; ++d)
        buf << "\n  " << faces[f] << " " << dirs[d] << ": " << boundaries[f][d];
    CCTK_INFO(buf.str().c_str());
  }
  const amrex::Array<int, dim> is_periodic{
      symmetries[0][0] == symmetry_t::periodic,
      symmetries[0][1] == symmetry_t::periodic,
      symmetries[0][2] == symmetry_t::periodic};

  amrcore =
      make_core(levels, patch, domain, max_num_levels - 1, ncells, is_periodic);

  if (verbose) {
#pragma omp critical
    {
      const int maxnumlevels = amrcore->maxLevel() + 1;
      for (const level_t level :
           level_range(level_t{patch, 0}, level_t{patch, maxnumlevels})) {
        CCTK_VINFO("amrex::Geometry level %d:", level.index());
        std::cout << amrcore->Geom(level.index()) << "\n";
      }
    }
  }
}

GHExt::PatchData::LevelData &
GHExt::PatchData::add_level(const level_t level, const amrex::BoxArray &ba,
                            const amrex::DistributionMapping &dm,
                            const std::function<std::string()> &why) {
  assert(level.patch() == patch && level.index() == num_levels());
  assert(int(level_modified.size()) == num_levels());
  ++GHExt::mutable_instance().epoch_;
  level_modified.push_back(true);
  return leveldata.emplace_back(level, ba, dm, why);
}

GHExt::PatchData::LevelData GHExt::PatchData::replace_level(const level_t level,
                                                            LevelData &&fresh) {
  assert(fresh.level == level);
  ++GHExt::mutable_instance().epoch_;
  using std::swap;
  swap(this->level(level), fresh);
  level_modified.at(level.index()) = true;
  return std::move(fresh);
}

void GHExt::PatchData::erase_levels_from(const level_t level) {
  assert(level.patch() == patch && level.index() >= 0 &&
         level.index() <= num_levels());
  ++GHExt::mutable_instance().epoch_;
  leveldata.erase(leveldata.begin() + level.index(), leveldata.end());
  level_modified.resize(level.index());
}

void GHExt::PatchData::reset_level_modified() {
  assert(num_levels() == amrcore->finestLevel() + 1);
  level_modified.assign(num_levels(), false);
}

std::optional<window_t::bounds_t> GHExt::PatchData::modified_levels() const {
  std::optional<window_t::bounds_t> bounds;
  for (int lev = 0; lev < int(level_modified.size()); ++lev) {
    if (!level_modified.at(lev))
      continue;
    if (!bounds)
      bounds = window_t::bounds_t{lev, lev + 1};
    else
      bounds->max = lev + 1;
  }
  return bounds;
}

GHExt::PatchData::LevelData::LevelData(const level_t level,
                                       const amrex::BoxArray &ba,
                                       const amrex::DistributionMapping &dm,
                                       const std::function<std::string()> &why)
    : level(level),
      buffers(GHExt::mutable_instance().make_buffers_(level, ba, dm)) {
  assert(this->buffers);
  DECLARE_CCTK_PARAMETERS;

  GHExt &ext = GHExt::mutable_instance();
  if (level.index() == 0) {
    iteration = 0;
    delta_iteration = 1;
    last_step_iteration = 0;
    last_finalized_iteration = 0;
  } else {
    auto &coarseleveldata = ext.level(level.parent());
    iteration = coarseleveldata.iteration;
    delta_iteration = coarseleveldata.delta_iteration /
                      (ext.use_subcycling ? time_refinement : 1);
    // A level that a regrid creates or remakes shares its parent's clock, and
    // so the parent's last step and last end-of-step window
    last_step_iteration = coarseleveldata.last_step_iteration;
    last_finalized_iteration = coarseleveldata.last_finalized_iteration;
  }

  const amrex::IntVect nghostzones = {
      ghost_size >= 0 ? ghost_size : ghost_size_x,
      ghost_size >= 0 ? ghost_size : ghost_size_y,
      ghost_size >= 0 ? ghost_size : ghost_size_z};
  fab = std::make_unique<amrex::FabArrayBase>(ba, dm, 1, nghostzones);
  assert(ba.ixType() == amrex::IndexType(amrex::IndexType::CELL,
                                         amrex::IndexType::CELL,
                                         amrex::IndexType::CELL));

  const int numgroups = CCTK_NumGroups();
  assert(groupdata.empty());
  groupdata.reserve(numgroups);
  for (const gi_t gi : index_range<gi_t>(gi_t{0}, gi_t{numgroups})) {
    const GHExt::GroupRecord &group = ext.record(gi);
    if (group.kind != group_kind::grid_function)
      groupdata.emplace_back(nullptr);
    else
      groupdata.push_back(
          std::make_unique<GHExt::PatchData::LevelData::GroupData>(
              group, level, ba, dm, why));
  }

  const amrex::Geometry &geom =
      ext.patch(level.patch()).amrcore->Geom(level.index());
  for (const std::unique_ptr<GroupData> &group : groupdata) {
    if (!group)
      continue;
    const GroupRecord::grid_function_t &grid_function = group->grid_function;
    const ghost_mask_key_t key{grid_function.indextype,
                               grid_function.nghostzones};
    if (ghost_masks.count(key))
      continue;
    const amrex::IndexType ixtype(
        grid_function.indextype[0] ? amrex::IndexType::CELL
                                   : amrex::IndexType::NODE,
        grid_function.indextype[1] ? amrex::IndexType::CELL
                                   : amrex::IndexType::NODE,
        grid_function.indextype[2] ? amrex::IndexType::CELL
                                   : amrex::IndexType::NODE);
    amrex::iMultiFab mask(convert(ba, ixtype), dm, 1,
                          amrex::IntVect(grid_function.nghostzones));
    mask.BuildMask(geom.Domain(), geom.periodicity(), ghost_mask_same_level,
                   ghost_mask_coarse_fine, ghost_mask_physical_boundary,
                   ghost_mask_interior);
    ghost_masks.emplace(key, std::move(mask));
  }

  if (int(ext.level_cctkGHs.size()) <= level.index()) {
    cGH *const cctkGH = ext.get_global_cctkGH();
    enter_level_mode(cctkGH, level);
    assert(int(ext.level_cctkGHs.size()) == level.index());
    ext.level_cctkGHs.push_back(GHExt::cctkGHptr::copy_of(cctkGH));
    leave_level_mode(cctkGH, level);
  }

  {
    LevelData &leveldata = *this;
    cGH *const cctkGH = ext.get_level_cctkGH(level);
    enter_patch_mode(cctkGH, leveldata);
    patch_cctkGH = GHExt::cctkGHptr::copy_of(cctkGH);
    const auto mfitinfo = amrex::MFItInfo().EnableTiling();
    for (amrex::MFIter mfi(*leveldata.fab, mfitinfo); mfi.isValid(); ++mfi) {
      const MFPointer mfp(mfi);
      enter_local_mode(cctkGH, leveldata, mfp);
      cctkGH_binding::bind_local(cctkGH, leveldata, mfp);
      assert(int(local_cctkGHs.size()) == tile_of(mfi).index());
      assert(mfp.index() == tile_of(mfi).component());
      local_cctkGHs.push_back(GHExt::cctkGHptr::copy_of(cctkGH));
      cctkGH_binding::unbind_local(cctkGH, leveldata, mfp);
      leave_local_mode(cctkGH, leveldata, mfp);
    }
    leave_patch_mode(cctkGH, leveldata);
  }
}

GHExt::PatchData::LevelData::GroupData::GroupData(
    const GroupRecord &record, const level_t level, const amrex::BoxArray &ba,
    const amrex::DistributionMapping &dm,
    const std::function<std::string()> &why)
    : record(record), grid_function(record.grid_function()), level(level) {
  const int numvars = record.numvars;

  assert(record.kind == group_kind::grid_function);
  assert(record.var_type == var_type::real);

  const GHExt &ext = driver();
  const auto &geom = ext.patch(level.patch()).amrcore->Geom(level.index());
  for (int d = 0; d < dim; ++d)
    if (geom.isPeriodic(d))
      assert(geom.Domain().length(d) >= grid_function.nghostzones[d]);

  const amrex::BoxArray &gba = convert(
      ba,
      amrex::IndexType(grid_function.indextype[0] ? amrex::IndexType::CELL
                                                  : amrex::IndexType::NODE,
                       grid_function.indextype[1] ? amrex::IndexType::CELL
                                                  : amrex::IndexType::NODE,
                       grid_function.indextype[2] ? amrex::IndexType::CELL
                                                  : amrex::IndexType::NODE));
  tls.reserve(record.num_timelevels());
  for ([[maybe_unused]] const tl_t tl : record.timelevels())
    tls.emplace_back(
        std::make_unique<amrex::MultiFab>(
            gba, dm, numvars, amrex::IntVect(grid_function.nghostzones)),
        numvars, why);
}

////////////////////////////////////////////////////////////////////////////////

bool GHExt::PatchData::LevelData::GroupData::
    all_faces_have_symmetries_or_boundaries() const {
  const auto &symmetries = driver().patch(level.patch()).symmetries;
  bool res = true;
  for (int f = 0; f < 2; ++f)
    for (int d = 0; d < dim; ++d)
      res &= symmetries.at(f).at(d) != symmetry_t::none ||
             (symmetries.at(f).at(d) == symmetry_t::none &&
              grid_function.boundaries.at(f).at(d) != boundary_t::none);
  return res;
}

void GHExt::PatchData::LevelData::GroupData::rotate_timelevels() {
  assert(num_timelevels() > 1);
  assert(open_write_count() == 0);
  std::rotate(tls.begin(), tls.end() - 1, tls.end());
}

////////////////////////////////////////////////////////////////////////////////

GHExt::GroupRecord::GroupRecord(const gi_t gi, const int num_timelevels)
    : groupindex(gi), num_timelevels_(num_timelevels) {
  cGroup cgroup;
  const int ierr = CCTK_GroupData(gi.index(), &cgroup);
  assert(!ierr);
  assert(num_timelevels >= 0 and num_timelevels <= cgroup.numtimelevels);

  groupname = CCTK_FullGroupName(gi.index());
  firstvarindex = CCTK_FirstVarIndexI(gi.index());
  numvars = cgroup.numvars;
  do_checkpoint = get_group_checkpoint_flag(gi);
  do_evolve = get_group_evolve_flag(gi);
  const std::optional<bool> restrict_tag = get_group_restrict_tag(gi);
  do_restrict = restrict_tag.value_or(do_checkpoint);
  restrict_on_sync = restrict_tag.value_or(true);
  switch (cgroup.grouptype) {
  case CCTK_GF:
    kind = group_kind::grid_function;
    break;
  case CCTK_ARRAY:
    kind = group_kind::array;
    break;
  case CCTK_SCALAR:
    kind = group_kind::scalar;
    break;
  default:
    CCTK_VERROR("Internal error: unexpected group type %d for group '%s'",
                (int)cgroup.grouptype, groupname.c_str());
  }
  const std::optional<CarpetX::var_type> type =
      var_type_from_cctk(cgroup.vartype);
  if (!type)
    refuse(std::source_location::current(),
           "group %s has variable type %s, which CarpetX does not store",
           groupname.c_str(), CCTK_VarTypeName(cgroup.vartype));
  var_type = *type;
  if (kind == group_kind::grid_function) {
    assert(cgroup.disttype == CCTK_DISTRIB_DEFAULT);
    assert(cgroup.dim == dim);
    grid_function_ = declared_grid_function(gi, numvars);
  } else {
    assert(cgroup.disttype == CCTK_DISTRIB_CONSTANT);
    assert(cgroup.dim >= 0);
    assert(cgroup.dim <= dim);
    array_ = std::make_unique<GlobalData::ArrayGroupData>(*this);
  }
}

GHExt::GlobalData::ArrayGroupData::ArrayGroupData(const GroupRecord &record)
    : record(record) {
  assert(record.kind != group_kind::grid_function);
  dimension = CCTK_GroupDimI(record.groupindex.index());

  CCTK_INT const *const *const sz = CCTK_GroupSizesI(record.groupindex.index());
  array_size = 1;
  for (int d = 0; d < dimension; ++d)
    array_size = array_size * *sz[d];

  for (int d = 0; d < dimension; ++d) {
    lsh[d] = *sz[d];
    ash[d] = *sz[d];
    gsh[d] = *sz[d];
    nghostzones[d] = 0;
    lbnd[d] = 0;
    ubnd[d] = *sz[d] - 1;
    bbox[2 * d] = bbox[2 * d + 1] = 1;
  }
  for (int d = dimension; d < dim; ++d) {
    lsh[d] = 1;
    ash[d] = 1;
    gsh[d] = 1;
    nghostzones[d] = 0;
    lbnd[d] = 0;
    ubnd[d] = 0;
    bbox[2 * d] = bbox[2 * d + 1] = 1;
  }

  tls.reserve(record.num_timelevels());
  for ([[maybe_unused]] const tl_t tl : record.timelevels())
    tls.emplace_back(
        AnyTypeVector(record.var_type, record.numvars * array_size),
        record.numvars, []() { return "freeze_storage"; });
}

void GHExt::GlobalData::ArrayGroupData::rotate_timelevels() {
  assert(record.num_timelevels() > 1);
  std::rotate(tls.begin(), tls.end() - 1, tls.end());
}

int GHExt::active_timelevels(const gi_t gi) const {
  if (pending_timelevels_)
    return pending_timelevels_->at(gi.index());
  return record(gi).num_timelevels();
}

void GHExt::record_storage(const gi_t gi, const int timelevels) {
  assert(timelevels >= 0 &&
         timelevels <= CCTK_DeclaredTimeLevelsGI(gi.index()));
  if (pending_timelevels_) {
    pending_timelevels_->at(gi.index()) = timelevels;
    return;
  }
  const int active = record(gi).num_timelevels();
  if (timelevels != active)
    CCTK_VWARN(CCTK_WARN_ALERT,
               "Attempting to %s storage for group '%s' from %d to %d "
               "timelevels after the STORAGE registry has been frozen; "
               "ignoring",
               timelevels > active ? "increase" : "decrease",
               CCTK_FullGroupName(gi.index()), active, timelevels);
}

void GHExt::freeze_storage() {
  DECLARE_CCTK_PARAMETERS;

  if (verbose)
#pragma omp critical
    CCTK_VINFO("freeze_storage");

  assert(pending_timelevels_);
  assert(group_records.empty());
  const int numgroups = CCTK_NumGroups();
  assert(int(pending_timelevels_->size()) == numgroups);
  group_records.reserve(numgroups);
  for (const gi_t gi : index_range<gi_t>(gi_t{0}, gi_t{numgroups}))
    group_records.emplace_back(gi, pending_timelevels_->at(gi.index()));
  pending_timelevels_.reset();

  for (const GroupRecord &group : group_records) {
    if (group.kind != group_kind::grid_function)
      continue;
    const std::optional<std::array<gi_t, dim>> &fluxes =
        group.grid_function().fluxes;
    if (!fluxes)
      continue;
    for (int d = 0; d < dim; ++d) {
      assert((*fluxes)[d] != group.groupindex);
      std::array<int, dim> flux_indextype{1, 1, 1};
      flux_indextype[d] = 0;
      assert(record((*fluxes)[d]).grid_function().indextype == flux_indextype);
      assert(record((*fluxes)[d]).numvars >= group.numvars);
    }
  }

  for (const GroupRecord &group : records()) {
    if (group.kind == group_kind::grid_function)
      continue;
    for (const tl_t tl : group.timelevels())
      invalidate_array({.gi = group.groupindex,
                        .tl = tl,
                        .where = make_valid_int(),
                        .event = "freeze_storage"});
  }
}

void GHExt::record_integrator(integrator_t integrator,
                              const std::source_location site) {
  if (!integrator_open_)
    refuse(site, "the driver froze its integrator record after the WRAGH "
                 "traversal; schedule the registration AT WRAGH");
  if (integrator_)
    refuse(site,
           "a time integrator is already registered (%d RK stages, %d "
           "integrated groups); an integrator registers exactly once, AT "
           "WRAGH",
           integrator_->stages().size(), int(integrator_->rhs.size()));
  integrator_ = std::move(integrator);
}

void GHExt::freeze_integrator(const std::source_location site) {
  if (!integrator_open_)
    refuse(site, "the integrator record is already frozen: the driver freezes "
                 "it once, after its WRAGH traversal");
  integrator_open_ = false;
}

const std::optional<GHExt::integrator_t> &
GHExt::integrator(const std::source_location site) const {
  if (integrator_open_)
    refuse(site, "the integrator record is read before the driver froze it; "
                 "a reader runs after the WRAGH traversal");
  return integrator_;
}

const GHExt::GroupRecord &
GHExt::record_of(const var_t var, const std::source_location site) const {
  const gi_t gi{CCTK_GroupIndexFromVarI(var.index())};
  if (gi.index() < 0)
    refuse(site, "variable index %d does not exist: there are %d variables",
           var.index(), CCTK_NumVars());
  return record(gi, site);
}

} // namespace CarpetX
