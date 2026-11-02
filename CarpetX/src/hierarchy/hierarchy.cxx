#include "hierarchy/driver.hxx"
#include "hierarchy/group_tags.hxx"
#include "hierarchy/operations.hxx"

#include "hierarchy/loops.hxx"
#include <carpetx.hxx>

#include <cctk.h>
#include <cctk_Arguments.h>
#include <cctk_Parameters.h>
#include <util_Table.h>

#include <AMReX.H>
#include <AMReX_BCRec.H>
#include <AMReX_FillPatchUtil.H>

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <optional>
#include <ostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace CarpetX {

namespace {
std::unique_ptr<GHExt> ghext;

GHExt &existing_hierarchy() {
  if (!ghext)
    CCTK_ERROR("The CarpetX grid hierarchy does not exist: the public API is "
               "available between the driver's SetupGH and its Shutdown");
  return *ghext;
}
} // namespace

GHExt &GHExt::mutable_instance() { return existing_hierarchy(); }

const GHExt &driver() { return existing_hierarchy(); }

GHExt &create_hierarchy(const GHExt::config_t config) {
  if (ghext)
    CCTK_ERROR("The CarpetX grid hierarchy already exists: SetupGH creates it "
               "once per run");
  ghext = std::make_unique<GHExt>(config);
  return *ghext;
}

void destroy_hierarchy() {
  if (!ghext)
    CCTK_ERROR("The CarpetX grid hierarchy does not exist: Shutdown destroys "
               "the hierarchy that SetupGH created");
  ghext = nullptr;
}

void assert_hierarchy_destroyed() { assert(!ghext); }

GHExt::GHExt(const config_t config)
    : use_subcycling(config.use_subcycling), do_reflux(config.do_reflux),
      pending_timelevels_(std::in_place, CCTK_NumGroups(), 0) {}

void GHExt::mark_initialized() {
  assert(!initialized_);
  initialized_ = true;
}

GHExt::PatchData &GHExt::add_patch(const patch_t patch) {
  assert(patch.index() == num_patches());
  return patchdata.emplace_back(patch);
}

std::atomic<CCTK_INT> carpetx_epoch{0};

////////////////////////////////////////////////////////////////////////////////

GHExt::cctkGHptr &GHExt::cctkGHptr::operator=(cctkGHptr &&ptr) {
  if (cctkGH)
    delete_cctkGH(cctkGH);
  cctkGH = ptr.cctkGH;
  ptr.cctkGH = nullptr;
  return *this;
}

GHExt::cctkGHptr &GHExt::cctkGHptr::operator=(cGH *&&cctkGH) {
  if (this->cctkGH)
    delete_cctkGH(this->cctkGH);
  this->cctkGH = cctkGH;
  return *this;
}

GHExt::cctkGHptr::~cctkGHptr() {
  if (cctkGH)
    delete_cctkGH(cctkGH);
  cctkGH = nullptr;
}

GHExt::PatchData::PatchData(const patch_t patch) : patch(patch) {
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

  const int coord = -1; // undefined?

  // Refinement ratios
  const amrex::Vector<amrex::IntVect> reffacts{}; // empty

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

  amrcore = std::make_unique<CactusAmrCore>(
      patch, domain, max_num_levels - 1, ncells, coord, reffacts, is_periodic);

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
                            const std::function<std::string()> &why,
                            level_buffers_ptr buffers) {
  assert(level.patch() == patch && level.index() == num_levels());
  return leveldata.emplace_back(level, ba, dm, why, std::move(buffers));
}

GHExt::PatchData::LevelData GHExt::PatchData::replace_level(const level_t level,
                                                            LevelData &&fresh) {
  assert(fresh.level == level);
  using std::swap;
  swap(this->level(level), fresh);
  return std::move(fresh);
}

void GHExt::PatchData::erase_levels_from(const level_t level) {
  assert(level.patch() == patch && level.index() >= 0 &&
         level.index() <= num_levels());
  leveldata.erase(leveldata.begin() + level.index(), leveldata.end());
}

GHExt::PatchData::LevelData::LevelData(const level_t level,
                                       const amrex::BoxArray &ba,
                                       const amrex::DistributionMapping &dm,
                                       const std::function<std::string()> &why,
                                       level_buffers_ptr buffers)
    : level(level), buffers(std::move(buffers)) {
  assert(this->buffers);
  DECLARE_CCTK_PARAMETERS;

  GHExt &ext = GHExt::mutable_instance();
  const int timereffact = ext.use_subcycling ? 2 : 1;
  if (level.index() == 0) {
    // We are creating the coarsest level
    is_subcycling_level = false; // unused
    iteration = 0;
    delta_iteration = 1;
    last_step_iteration = 0;
    last_finalized_iteration = 0;
  } else {
    auto &coarseleveldata = ext.level(level.parent());
    is_subcycling_level = ext.use_subcycling;
    iteration = coarseleveldata.iteration;
    delta_iteration = coarseleveldata.delta_iteration / timereffact;
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

  for (const gi_t gi : index_range<gi_t>(gi_t{0}, gi_t{numgroups})) {
    const GHExt::GroupRecord &group = ext.record(gi);
    if (group.kind != group_kind::grid_function)
      continue;
    const std::optional<std::array<gi_t, dim> > fluxes = get_group_fluxes(gi);
    if (fluxes) {
      for (int d = 0; d < dim; ++d) {
        assert((*fluxes)[d] != gi);
        std::array<int, dim> flux_indextype{1, 1, 1};
        flux_indextype[d] = 0;
        assert(get_group_indextype((*fluxes)[d]) == flux_indextype);
        assert(ext.record((*fluxes)[d]).numvars >= group.numvars);
      }
    }
  }

  if (int(ext.level_cctkGHs.size()) <= level.index()) {
    cGH *const cctkGH = ext.get_global_cctkGH();
    enter_level_mode(cctkGH, level);
    assert(int(ext.level_cctkGHs.size()) == level.index());
    ext.level_cctkGHs.emplace_back(copy_cctkGH(cctkGH));
    leave_level_mode(cctkGH, level);
  }

  {
    LevelData &leveldata = *this;
    cGH *const cctkGH = ext.get_level_cctkGH(level);
    enter_patch_mode(cctkGH, leveldata);
    patch_cctkGH = GHExt::cctkGHptr(copy_cctkGH(cctkGH));
    int component = 0;
    const auto mfitinfo = amrex::MFItInfo().EnableTiling();
    for (amrex::MFIter mfi(*leveldata.fab, mfitinfo); mfi.isValid();
         ++mfi, ++component) {
      const MFPointer mfp(mfi);
      enter_local_mode(cctkGH, leveldata, mfp);
      assert(int(local_cctkGHs.size()) == component);
      local_cctkGHs.emplace_back(copy_cctkGH(cctkGH));
      leave_local_mode(cctkGH, leveldata, mfp);
    }
    leave_patch_mode(cctkGH, leveldata);
  }
}

GHExt::PatchData::LevelData::GroupData::GroupData(
    const GroupRecord &record, const level_t level, const amrex::BoxArray &ba,
    const amrex::DistributionMapping &dm,
    const std::function<std::string()> &why)
    : record(record), level(level) {
  const int numvars = record.numvars;
  const gi_t groupindex = record.groupindex;

  assert(record.kind == group_kind::grid_function);
  assert(record.var_type == var_type::real);

  indextype = get_group_indextype(groupindex);
  nghostzones = get_group_nghostzones(groupindex);

  interpolator = get_interpolator(*this); // reads groupindex, indextype

  const GHExt &ext = driver();
  const auto &geom = ext.patch(level.patch()).amrcore->Geom(level.index());
  for (int d = 0; d < dim; ++d)
    if (geom.isPeriodic(d))
      assert(geom.Domain().length(d) >= nghostzones[d]);

  boundaries = get_group_boundaries(groupindex);
  parities = get_group_parities(groupindex);
  if (parities.empty()) {
    std::array<int, dim> parity;
    for (int d = 0; d < dim; ++d)
      parity[d] = +1;
    parities.resize(numvars, parity);
  }
  assert(int(parities.size()) == numvars);
  for (const vi_t vi : record.variables())
    for (int d = 0; d < dim; ++d)
      assert(abs(parities.at(vi.index())[d]) == 1);

  dirichlet_values = get_group_dirichlet_values(groupindex);
  if (dirichlet_values.empty())
    dirichlet_values.resize(numvars, 0);
  robin_values = get_group_robin_values(groupindex);
  if (robin_values.empty())
    robin_values.resize(numvars, 0);

  amrex::BCRec bcrec;
  for (int f = 0; f < 2; ++f) {
    for (int d = 0; d < dim; ++d) {
      const auto bc =
          ext.patch(level.patch()).symmetries[f][d] == symmetry_t::periodic
              ? amrex::BCType::int_dir
              : amrex::BCType::ext_dir;
      if (f == 0)
        bcrec.setLo(d, bc);
      else
        bcrec.setHi(d, bc);
    }
  }
  bcrecs.resize(numvars, bcrec);

  const amrex::BoxArray &gba = convert(
      ba, amrex::IndexType(
              indextype[0] ? amrex::IndexType::CELL : amrex::IndexType::NODE,
              indextype[1] ? amrex::IndexType::CELL : amrex::IndexType::NODE,
              indextype[2] ? amrex::IndexType::CELL : amrex::IndexType::NODE));
  tls.reserve(record.num_timelevels());
  for ([[maybe_unused]] const tl_t tl : record.timelevels())
    tls.emplace_back(std::make_unique<amrex::MultiFab>(
                         gba, dm, numvars, amrex::IntVect(nghostzones)),
                     numvars, why);

  fluxes = get_group_fluxes(groupindex);
  if (fluxes) {
    assert((indextype == std::array<int, dim>{1, 1, 1}));
    if (level.index() > 0 && group_has_flux_register(groupindex)) {
      const amrex::IntVect ratio =
          ext.patch(level.patch()).amrcore->refRatio(level.parent().index());
      freg = std::make_unique<amrex::FluxRegister>(gba, dm, ratio,
                                                   level.index(), numvars);
      freg->setVal(0);
    }
  }
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
              boundaries.at(f).at(d) != boundary_t::none);
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

const GHExt::GroupRecord &
GHExt::record_of(const var_t var, const std::source_location site) const {
  const gi_t gi{CCTK_GroupIndexFromVarI(var.index())};
  if (gi.index() < 0)
    refuse(site, "variable index %d does not exist: there are %d variables",
           var.index(), CCTK_NumVars());
  return record(gi, site);
}

////////////////////////////////////////////////////////////////////////////////

const GHExt::PatchData &hierarchy_writer::add_patch(const patch_t patch) {
  return GHExt::mutable_instance().add_patch(patch);
}

const GHExt::PatchData::LevelData &
hierarchy_writer::add_level(const level_t level, const amrex::BoxArray &ba,
                            const amrex::DistributionMapping &dm,
                            const std::function<std::string()> &why,
                            level_buffers_ptr buffers) {
  return GHExt::mutable_instance()
      .patch(level.patch())
      .add_level(level, ba, dm, why, std::move(buffers));
}

GHExt::PatchData::LevelData
hierarchy_writer::remake_level(const level_t level, const amrex::BoxArray &ba,
                               const amrex::DistributionMapping &dm,
                               const std::function<std::string()> &why,
                               level_buffers_ptr buffers) {
  GHExt::PatchData::LevelData fresh(level, ba, dm, why, std::move(buffers));
  return GHExt::mutable_instance()
      .patch(level.patch())
      .replace_level(level, std::move(fresh));
}

void hierarchy_writer::erase_levels_from(const level_t level) {
  GHExt::mutable_instance().patch(level.patch()).erase_levels_from(level);
}

void hierarchy_writer::set_max_grid_size(
    const patch_t patch, const amrex::Vector<amrex::IntVect> &max_grid_sizes) {
  GHExt::mutable_instance().patch(patch).amrcore->SetMaxGridSize(
      max_grid_sizes);
}

void hierarchy_writer::make_new_grids(const patch_t patch,
                                      const CCTK_REAL time) {
  GHExt::mutable_instance().patch(patch).amrcore->MakeNewGrids(time);
}

void hierarchy_writer::mark_initialized() {
  GHExt::mutable_instance().mark_initialized();
}

void hierarchy_writer::regrid(const level_t base_level, const CCTK_REAL time) {
  GHExt::mutable_instance()
      .patch(base_level.patch())
      .amrcore->regrid(base_level.index(), time);
}

void hierarchy_writer::set_finest_level(const level_t finest_level) {
  GHExt::mutable_instance()
      .patch(finest_level.patch())
      .amrcore->SetFinestLevel(finest_level.index());
}

void hierarchy_writer::recover_level(const level_t level, amrex::BoxArray ba,
                                     const bool rechop) {
  CactusAmrCore &amrcore =
      *GHExt::mutable_instance().patch(level.patch()).amrcore;
  const int lev = level.index();
  if (lev > 0) {
    amrex::Geometry geom = amrcore.Geom(lev - 1);
    geom.refine({2, 2, 2});
    amrcore.SetGeometry(lev, geom);
  }
  if (rechop)
    ba = amrcore.RechopLevel(level, ba);
  amrcore.SetBoxArray(lev, ba);
  const amrex::DistributionMapping dm(ba);
  amrcore.SetDistributionMap(lev, dm);
  amrcore.make_level(level, ba, dm, "Recovering");
}

void hierarchy_writer::record_storage(const gi_t gi, const int timelevels) {
  GHExt::mutable_instance().record_storage(gi, timelevels);
}

void hierarchy_writer::freeze_storage() {
  GHExt::mutable_instance().freeze_storage();
}

void hierarchy_writer::set_global_cctkGH(const cGH *const cctkGH) {
  GHExt &ext = GHExt::mutable_instance();
  assert(!ext.global_cctkGH);
  ext.global_cctkGH = GHExt::cctkGHptr(copy_cctkGH(cctkGH));
}

void *hierarchy_writer::array_data(const gi_t gi, const tl_t tl) {
  return GHExt::mutable_instance()
      .record(gi)
      .array()
      .timelevel(tl)
      .data.data_at(0);
}

std::size_t hierarchy_writer::acquire_scratch(const group_t group,
                                              const std::source_location site) {
  auto &groupdata = GHExt::mutable_instance().group(group, site);
  auto &pool = groupdata.scratch_pool;
  std::size_t slot = 0;
  while (slot < pool.size() && pool.at(slot).in_use)
    ++slot;
  if (slot == pool.size()) {
    const amrex::MultiFab &mfab0 = *groupdata.timelevel(tl_t{0}).mfab;
    pool.push_back({std::make_unique<amrex::MultiFab>(
                        mfab0.boxArray(), mfab0.DistributionMap(),
                        mfab0.nComp(), mfab0.nGrowVect()),
                    false});
  }
  pool.at(slot).in_use = true;
  return slot;
}

void hierarchy_writer::release_scratch(const group_t group,
                                       const std::size_t slot) {
  auto &pool = GHExt::mutable_instance().group(group).scratch_pool;
  assert(slot < pool.size() && pool.at(slot).in_use);
  pool.at(slot).in_use = false;
}

void hierarchy_writer::advance_clocks(const window_t window) {
  loop_serially(window, [](const GHExt::PatchData::LevelData &leveldata) {
    auto &clock = GHExt::mutable_instance().level(leveldata.level);
    clock.iteration += clock.delta_iteration;
  });
}

void hierarchy_writer::stamp_last_step(const window_t window,
                                       const int iteration) {
  loop_serially(window, [&](const GHExt::PatchData::LevelData &leveldata) {
    GHExt::mutable_instance().level(leveldata.level).last_step_iteration =
        iteration;
  });
}

void hierarchy_writer::stamp_last_finalized(const window_t window,
                                            const int iteration) {
  loop_serially(window, [&](const GHExt::PatchData::LevelData &leveldata) {
    GHExt::mutable_instance().level(leveldata.level).last_finalized_iteration =
        iteration;
  });
}

void hierarchy_writer::restore_level_clock(
    const level_t level, const rat64 iteration,
    const std::optional<int> last_step_iteration) {
  auto &leveldata = GHExt::mutable_instance().level(level);
  leveldata.iteration = iteration;
  if (last_step_iteration)
    leveldata.last_step_iteration = *last_step_iteration;
}

void hierarchy_writer::rotate_timelevels(const window_t window, const gi_t gi) {
  const int ntls0 = driver().record(gi).num_timelevels();
  loop_serially(window, [&](const GHExt::PatchData::LevelData &leveldata) {
    auto &restrict groupdata =
        GHExt::mutable_instance().level(leveldata.level).group(gi);
    const int ntls = groupdata.num_timelevels();
    assert(ntls == ntls0);
    if (ntls > 1) {
      groupdata.rotate_timelevels();
      for (auto &restrict local_cctkGH_ptr : leveldata.local_cctkGHs) {
        cGH *const local_cctkGH = local_cctkGH_ptr.get();
        const int component = local_cctkGH->cctk_component;
        const GridPtrDesc1 grid1(leveldata, groupdata, component);
        for (const tl_t tl : groupdata.record.timelevels()) {
          const amrex::Array4<CCTK_REAL> vars =
              groupdata.timelevel(tl).mfab->array(component);
          for (const vi_t vi : groupdata.record.variables())
            local_cctkGH->data[groupdata.record.var(vi).index()][tl.index()] =
                grid1.ptr(vars, vi);
        }
      }
    }
  });
}

void hierarchy_writer::rotate_array_timelevels(const gi_t gi) {
  GHExt::mutable_instance().record(gi).array().rotate_timelevels();
}

////////////////////////////////////////////////////////////////////////////////

YAML::Emitter &operator<<(YAML::Emitter &yaml, const GHExt &ext) {
  yaml << YAML::LocalTag("ghext-2.0.0");
  yaml << YAML::BeginMap;
  yaml << YAML::Key << "globaldata" << YAML::Value;
  yaml << YAML::LocalTag("globaldata-1.0.0");
  yaml << YAML::BeginMap;
  yaml << YAML::Key << "arraygroupdata" << YAML::Value << YAML::BeginSeq;
  for (const auto &group : ext.group_records)
    if (group.kind != group_kind::grid_function)
      yaml << group.array();
  yaml << YAML::EndSeq;
  yaml << YAML::EndMap;
  yaml << YAML::Key << "patchdata" << YAML::Value << YAML::BeginSeq;
  for (const auto &patchdata : ext.patches())
    yaml << patchdata;
  yaml << YAML::EndSeq;
  yaml << YAML::EndMap;
  return yaml;
}

} // namespace CarpetX
