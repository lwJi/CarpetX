#include "hierarchy/driver.hxx"
#include "hierarchy/group_tags.hxx"

#include "comm/boundaries.hxx"
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
#include <iomanip>
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
} // namespace

GHExt &driver() {
  if (!ghext)
    CCTK_ERROR("The CarpetX grid hierarchy does not exist: the public API is "
               "available between the driver's SetupGH and its Shutdown");
  return *ghext;
}

GHExt &create_hierarchy() {
  if (ghext)
    CCTK_ERROR("The CarpetX grid hierarchy already exists: SetupGH creates it "
               "once per run");
  ghext = std::make_unique<GHExt>();
  ghext->active_timelevels.assign(CCTK_NumGroups(), 0);
  ghext->storage_frozen = false;
  return *ghext;
}

void destroy_hierarchy() {
  if (!ghext)
    CCTK_ERROR("The CarpetX grid hierarchy does not exist: Shutdown destroys "
               "the hierarchy that SetupGH created");
  ghext = nullptr;
}

void assert_hierarchy_destroyed() { assert(!ghext); }

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
      for (int level = 0; level < maxnumlevels; ++level) {
        CCTK_VINFO("amrex::Geometry level %d:", level);
        std::cout << amrcore->Geom(level) << "\n";
      }
    }
  }
}

GHExt::PatchData::LevelData &
GHExt::PatchData::add_level(const level_t level, const amrex::BoxArray &ba,
                            const amrex::DistributionMapping &dm,
                            const std::function<std::string()> &why) {
  assert(level.patch() == patch && level.index() == num_levels());
  return leveldata.emplace_back(level, ba, dm, why);
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
                                       const std::function<std::string()> &why)
    : level(level) {
  DECLARE_CCTK_PARAMETERS;

  GHExt &ext = driver();
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
  for (int gi0 = 0; gi0 < numgroups; ++gi0) {
    const GHExt::GroupRecord &group = ext.record(gi_t{gi0});
    if (group.kind != group_kind::grid_function)
      groupdata.emplace_back(nullptr);
    else
      groupdata.push_back(
          std::make_unique<GHExt::PatchData::LevelData::GroupData>(
              group, level, ba, dm, why));
  }

  for (int gi0 = 0; gi0 < numgroups; ++gi0) {
    const gi_t gi{gi0};
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
  const int gi = record.groupindex.index();
  const int numvars = record.numvars;
  const gi_t groupindex = record.groupindex;
  cGroup cgroup;
  int ierr = CCTK_GroupData(gi, &cgroup);
  assert(!ierr);

  assert(record.kind == group_kind::grid_function);
  assert(cgroup.vartype == CCTK_VARIABLE_REAL);
  assert(cgroup.disttype == CCTK_DISTRIB_DEFAULT);
  assert(cgroup.dim == dim);

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
  const int ntls = record.num_timelevels();
  tls.reserve(ntls);
  for (int tl = 0; tl < ntls; ++tl)
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
      // FluxRegister::define leaves the FabSets uninitialized. Zero them so
      // that a register that is never fed (e.g. ODESolvers::method =
      // "constant") is a no-op in Reflux and serializes as zeros at a
      // mid-cycle checkpoint.
      freg->setVal(0);
      freg_scratch = std::make_unique<amrex::FluxRegister>(
          gba, dm, ratio, level.index(), numvars);
    }
  }
}

////////////////////////////////////////////////////////////////////////////////

std::string subcycling_band_tag(const band_kind kind, const int stage) {
  std::ostringstream buf;
  switch (kind) {
  case band_kind::ks_source:
    assert(stage >= 0 && stage < max_num_rk_stages);
    buf << "kss_s" << std::setw(2) << std::setfill('0') << stage;
    break;
  case band_kind::old_source:
    buf << "olds";
    break;
  case band_kind::flux_register:
    assert(stage >= 0 && stage < 2 * dim);
    buf << "freg_" << "xyz"[stage / 2] << (stage % 2 ? "hi" : "lo");
    break;
  default:
    assert(0);
  }
  return buf.str();
}

amrex::MultiFab *rk_source_band(const level_t level, const gi_t gi,
                                const band_kind kind, const int stage) {
  assert(level.index() >= 0);
  const std::optional<level_t> child = level.child();
  if (!child)
    return nullptr;
  const auto &childleveldata = driver().level(*child);
  const auto &groupdata = childleveldata.group(gi);
  switch (kind) {
  case band_kind::ks_source:
    assert(stage >= 0 && stage < max_num_rk_stages);
    return groupdata.ks_source_band[stage].get();
  case band_kind::old_source:
    return groupdata.old_source_band.get();
  case band_kind::flux_register: {
    assert(stage >= 0 && stage < 2 * dim);
    if (!groupdata.freg)
      return nullptr;
    const int dir = stage / 2;
    const amrex::Orientation::Side side =
        stage % 2 ? amrex::Orientation::high : amrex::Orientation::low;
    return &(*groupdata.freg)[amrex::Orientation(dir, side)].multiFab();
  }
  default:
    assert(0);
  }
  return nullptr;
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

void GHExt::PatchData::LevelData::GroupData::apply_boundary_conditions(
    amrex::MultiFab &mfab) const {
  DECLARE_CCTK_PARAMETERS;

  static Timer timer("apply_boundary_conditions");
  Interval interval(timer);

  const amrex::Geometry &geom =
      driver().patch(level.patch()).amrcore->Geom(level.index());

  if (geom.isAllPeriodic())
    return;

  // Create a grown domain box containing valid + periodic cells
  const amrex::Box &domain = geom.Domain();
  amrex::Box gdomain = amrex::convert(domain, mfab.boxArray().ixType());
  for (int d = 0; d < dim; ++d)
    if (geom.isPeriodic(d))
      gdomain.grow(d, mfab.nGrow(d));

  // Do not tile because the boundary boxes are likely already small
  const auto mfitinfo = amrex::MFItInfo().DisableDeviceSync();
#pragma omp parallel
  for (amrex::MFIter mfi(mfab, mfitinfo); mfi.isValid(); ++mfi) {
    amrex::FArrayBox &dest = mfab[mfi];
    // If there are cells not in the valid + periodic grown box, then
    // we need to fill them here
    if (!gdomain.contains(dest.box()))
      BoundaryCondition(*this, dest).apply();
  }
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
  if (kind != group_kind::grid_function)
    array_ = std::make_unique<GlobalData::ArrayGroupData>(*this);
}

GHExt::GlobalData::ArrayGroupData::ArrayGroupData(const GroupRecord &record)
    : record(record) {
  const int gi = record.groupindex.index();
  cGroup cgroup;
  const int ierr = CCTK_GroupData(gi, &cgroup);
  assert(!ierr);
  assert(record.kind != group_kind::grid_function);
  assert(cgroup.disttype == CCTK_DISTRIB_CONSTANT);
  assert(cgroup.dim >= 0);
  assert(cgroup.dim <= dim);

  CCTK_INT const *const *const sz = CCTK_GroupSizesI(gi);
  array_size = 1;
  for (int d = 0; d < cgroup.dim; ++d)
    array_size = array_size * *sz[d];

  dimension = cgroup.dim;
  for (int d = 0; d < cgroup.dim; ++d) {
    lsh[d] = *sz[d];
    ash[d] = *sz[d];
    gsh[d] = *sz[d];
    nghostzones[d] = 0;
    lbnd[d] = 0;
    ubnd[d] = *sz[d] - 1;
    bbox[2 * d] = bbox[2 * d + 1] = 1;
  }
  for (int d = cgroup.dim; d < dim; ++d) {
    lsh[d] = 1;
    ash[d] = 1;
    gsh[d] = 1;
    nghostzones[d] = 0;
    lbnd[d] = 0;
    ubnd[d] = 0;
    bbox[2 * d] = bbox[2 * d + 1] = 1;
  }

  const int ntls = record.num_timelevels();
  tls.reserve(ntls);
  for (int tl = 0; tl < ntls; ++tl)
    tls.emplace_back(AnyTypeVector(cgroup.vartype, record.numvars * array_size),
                     record.numvars, []() { return "freeze_storage"; });
}

void GHExt::GlobalData::ArrayGroupData::rotate_timelevels() {
  assert(record.num_timelevels() > 1);
  std::rotate(tls.begin(), tls.end() - 1, tls.end());
}

void GHExt::freeze_storage() {
  DECLARE_CCTK_PARAMETERS;

  if (verbose)
#pragma omp critical
    CCTK_VINFO("freeze_storage");

  assert(!storage_frozen);
  assert(group_records.empty());
  const int numgroups = CCTK_NumGroups();
  assert(int(active_timelevels.size()) == numgroups);
  group_records.reserve(numgroups);
  for (int gi = 0; gi < numgroups; ++gi)
    group_records.emplace_back(gi_t{gi}, active_timelevels.at(gi));
  storage_frozen = true;

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
