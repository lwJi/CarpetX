#include "driver.hxx"

#include "boundaries.hxx"
#include "fillpatch.hxx"
#include "interp.hxx"
#include "io.hxx"
#include "logo.hxx"
#include "loop_device.hxx"
#include "prolongate_3d_rf2.hxx"
#include "schedule.hxx"
#include "subcycling.hxx"
#include <carpetx.hxx>

#include <cctk.h>
#include <cctk_Arguments.h>
#include <cctk_Parameters.h>
#include <util_Table.h>

#include <AMReX.H>
#include <AMReX_BCRec.H>
#include <AMReX_FillPatchUtil.H>
#include <AMReX_ParmParse.H>

#include <mpi.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdarg>
#include <cstdio>
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

// Global variables

int ghext_handle = -1;

amrex::AMReX *restrict pamrex = nullptr;
std::unique_ptr<GHExt> ghext;

GHExt &driver() {
  if (!ghext)
    CCTK_ERROR("The CarpetX grid hierarchy does not exist: the public API is "
               "available between the driver's SetupGH and its Shutdown");
  return *ghext;
}

namespace {
thread_local const char *operation_name = nullptr;
}

const char *current_operation() { return operation_name; }

operation_scope_t::operation_scope_t(const char *const name)
    : previous(operation_name) {
  operation_name = name;
}

operation_scope_t::~operation_scope_t() { operation_name = previous; }

void refuse(const std::source_location site, const char *const fmt, ...) {
  va_list ap, ap2;
  va_start(ap, fmt);
  va_copy(ap2, ap);
  const int length = std::vsnprintf(nullptr, 0, fmt, ap);
  va_end(ap);
  assert(length >= 0);
  std::vector<char> message(length + 1);
  std::vsnprintf(message.data(), message.size(), fmt, ap2);
  va_end(ap2);
  if (operation_name)
    CCTK_VError(site.line(), site.file_name(), CCTK_THORNSTRING, "%s: %s",
                operation_name, message.data());
  CCTK_VError(site.line(), site.file_name(), CCTK_THORNSTRING, "%s",
              message.data());
}

GHExt::PatchData &GHExt::add_patch(const int patch) {
  assert(patch == num_patches());
  return patchdata.emplace_back(patch);
}

std::atomic<CCTK_INT> carpetx_epoch{0};

// Registered functions

void *SetupGH(tFleshConfig *fc, int convLevel, cGH *cctkGH);
int InitGH(cGH *cctkGH);
int ScheduleTraverseGH(cGH *cctkGH, const char *where);

int MyProc(const cGH *cctkGH);
int nProcs(const cGH *cctkGH);
int Exit(cGH *cctkGH, int retval);
int Abort(cGH *cctkGH, int retval);
int Barrier(const cGH *cctkGHa);
const int *ArrayGroupSizeB(const cGH *cctkGH, int dir, int gi,
                           const char *groupname);
int GroupDynamicData(const cGH *cctkGH, int gi, cGroupDynamicData *data);

// Aliased functions

extern "C" void CarpetX_CallScheduleGroup(void *cctkGH, const char *groupname);
extern "C" CCTK_INT CarpetX_GetDomainSpecification(
    CCTK_INT patch, CCTK_INT size, CCTK_REAL *restrict const physical_min,
    CCTK_REAL *restrict const physical_max,
    CCTK_REAL *restrict const interior_min,
    CCTK_REAL *restrict const interior_max,
    CCTK_REAL *restrict const exterior_min,
    CCTK_REAL *restrict const exterior_max, CCTK_REAL *restrict const spacing);
extern "C" CCTK_INT CarpetX_GetBoundarySizesAndTypes(
    const void *cctkGH_, CCTK_INT patch, CCTK_INT size,
    CCTK_INT *restrict const bndsize, CCTK_INT *restrict const is_ghostbnd,
    CCTK_INT *restrict const is_symbnd, CCTK_INT *restrict const is_physbnd);

/* Return the current AMR grid epoch — a monotonically increasing counter that
 * is incremented whenever the grid hierarchy is structurally modified.
 *
 * Specifically, the epoch advances on:
 *   - Level creation   (MakeNewLevelFromScratch, MakeNewLevelFromCoarse)
 *   - Level remeshing  (RemakeLevel)
 *   - Level removal    (ClearLevel)
 *   - Checkpoint recovery (InputInitial)
 *
 * Callers that cache data derived from the grid topology (e.g. interpolation
 * setup objects) can use this value as a cheap validity stamp: if the epoch
 * returned here differs from the one stored when the cache was built, the
 * cached data must be rebuilt before use.
 *
 * The counter starts at 0 and is read atomically with relaxed ordering, which
 * is safe for a simple snapshot — no synchronisation with other memory
 * operations is implied or required.
 */
extern "C" CCTK_INT CarpetX_GetEpoch(void) {
  return carpetx_epoch.load(std::memory_order_relaxed);
}

////////////////////////////////////////////////////////////////////////////////

std::ostream &operator<<(std::ostream &os, const symmetry_t symmetry) {
  switch (symmetry) {
  case symmetry_t::none:
    return os << "none";
  case symmetry_t::interpatch:
    return os << "interpatch";
  case symmetry_t::periodic:
    return os << "periodic";
  case symmetry_t::reflection:
    return os << "reflection";
  default:
    assert(0);
  }
  return os;
}

std::ostream &operator<<(std::ostream &os, const boundary_t boundary) {
  switch (boundary) {
  case boundary_t::none:
    return os << "none";
  case boundary_t::symmetry_boundary:
    return os << "symmetry_boundary";
  case boundary_t::dirichlet:
    return os << "dirichlet";
  case boundary_t::linear_extrapolation:
    return os << "linear_extrapolation";
  case boundary_t::neumann:
    return os << "neumann";
  case boundary_t::robin:
    return os << "robin";
  default:
    assert(0);
  }
  return os;
}

std::array<std::array<symmetry_t, dim>, 2> get_symmetries(const int patch) {
  // patch < 0 return symmetries without taking interpatch boundaries into
  // account
  DECLARE_CCTK_PARAMETERS;

  std::array<std::array<bool, 3>, 2> is_interpatch{
      {{{false, false, false}}, {{false, false, false}}}};
  if (patch >= 0 &&
      CCTK_IsFunctionAliased("MultiPatch_GetBoundarySpecification2")) {
    CCTK_INT is_interpatch_boundary[2 * dim];
    const int ierr = MultiPatch_GetBoundarySpecification2(
        patch, 2 * dim, is_interpatch_boundary);
    assert(!ierr);
    for (int f = 0; f < 2; ++f)
      for (int d = 0; d < dim; ++d)
        is_interpatch[f][d] = is_interpatch_boundary[2 * d + f];
  }
  const std::array<std::array<bool, 3>, 2> is_periodic{{
      {{bool(periodic && periodic_x), bool(periodic && periodic_y),
        bool(periodic && periodic_z)}},
      {{bool(periodic && periodic_x), bool(periodic && periodic_y),
        bool(periodic && periodic_z)}},
  }};
  const std::array<std::array<bool, 3>, 2> is_reflection{{
      {{bool(reflection_x), bool(reflection_y), bool(reflection_z)}},
      {{bool(reflection_upper_x), bool(reflection_upper_y),
        bool(reflection_upper_z)}},
  }};

  for (int f = 0; f < 2; ++f)
    for (int d = 0; d < dim; ++d)
      assert(is_interpatch[f][d] + is_periodic[f][d] + is_reflection[f][d] <=
             1);

  std::array<std::array<symmetry_t, dim>, 2> symmetries;
  for (int f = 0; f < 2; ++f)
    for (int d = 0; d < dim; ++d)
      symmetries[f][d] = is_interpatch[f][d]   ? symmetry_t::interpatch
                         : is_periodic[f][d]   ? symmetry_t::periodic
                         : is_reflection[f][d] ? symmetry_t::reflection
                                               : symmetry_t::none;

  return symmetries;
}

std::array<std::array<boundary_t, dim>, 2> get_default_boundaries() {
  DECLARE_CCTK_PARAMETERS;

  const std::array<std::array<symmetry_t, dim>, 2> symmetries =
      get_symmetries(-1);
  std::array<std::array<bool, 3>, 2> is_symmetry;
  for (int f = 0; f < 2; ++f)
    for (int d = 0; d < dim; ++d)
      is_symmetry[f][d] = symmetries[f][d] != symmetry_t::none;

  const std::array<std::array<bool, 3>, 2> is_dirichlet{{
      {{
          bool(CCTK_EQUALS(boundary_x, "dirichlet")),
          bool(CCTK_EQUALS(boundary_y, "dirichlet")),
          bool(CCTK_EQUALS(boundary_z, "dirichlet")),
      }},
      {{
          bool(CCTK_EQUALS(boundary_upper_x, "dirichlet")),
          bool(CCTK_EQUALS(boundary_upper_y, "dirichlet")),
          bool(CCTK_EQUALS(boundary_upper_z, "dirichlet")),
      }},
  }};
  const std::array<std::array<bool, 3>, 2> is_linear_extrapolation{{
      {{
          bool(CCTK_EQUALS(boundary_x, "linear extrapolation")),
          bool(CCTK_EQUALS(boundary_y, "linear extrapolation")),
          bool(CCTK_EQUALS(boundary_z, "linear extrapolation")),
      }},
      {{
          bool(CCTK_EQUALS(boundary_upper_x, "linear extrapolation")),
          bool(CCTK_EQUALS(boundary_upper_y, "linear extrapolation")),
          bool(CCTK_EQUALS(boundary_upper_z, "linear extrapolation")),
      }},
  }};
  const std::array<std::array<bool, 3>, 2> is_neumann{{
      {{
          bool(CCTK_EQUALS(boundary_x, "neumann")),
          bool(CCTK_EQUALS(boundary_y, "neumann")),
          bool(CCTK_EQUALS(boundary_z, "neumann")),
      }},
      {{
          bool(CCTK_EQUALS(boundary_upper_x, "neumann")),
          bool(CCTK_EQUALS(boundary_upper_y, "neumann")),
          bool(CCTK_EQUALS(boundary_upper_z, "neumann")),
      }},
  }};
  const std::array<std::array<bool, 3>, 2> is_robin{{
      {{
          bool(CCTK_EQUALS(boundary_x, "robin")),
          bool(CCTK_EQUALS(boundary_y, "robin")),
          bool(CCTK_EQUALS(boundary_z, "robin")),
      }},
      {{
          bool(CCTK_EQUALS(boundary_upper_x, "robin")),
          bool(CCTK_EQUALS(boundary_upper_y, "robin")),
          bool(CCTK_EQUALS(boundary_upper_z, "robin")),
      }},
  }};
  for (int f = 0; f < 2; ++f)
    for (int d = 0; d < dim; ++d)
      assert(is_symmetry[f][d] + is_dirichlet[f][d] +
                 is_linear_extrapolation[f][d] + is_neumann[f][d] +
                 is_robin[f][d] <=
             1);

  std::array<std::array<boundary_t, dim>, 2> boundaries;
  for (int f = 0; f < 2; ++f)
    for (int d = 0; d < dim; ++d)
      boundaries[f][d] = is_symmetry[f][d]    ? boundary_t::symmetry_boundary
                         : is_dirichlet[f][d] ? boundary_t::dirichlet
                         : is_linear_extrapolation[f][d]
                             ? boundary_t::linear_extrapolation
                         : is_neumann[f][d] ? boundary_t::neumann
                         : is_robin[f][d]   ? boundary_t::robin
                                            : boundary_t::none;

  return boundaries;
}

std::array<std::array<boundary_t, dim>, 2> get_group_boundaries(const int gi) {
  DECLARE_CCTK_PARAMETERS;

  const std::array<std::array<symmetry_t, dim>, 2> symmetries =
      get_symmetries(-1);
  std::array<std::array<bool, 3>, 2> is_symmetry;
  for (int f = 0; f < 2; ++f)
    for (int d = 0; d < dim; ++d)
      is_symmetry[f][d] = symmetries[f][d] != symmetry_t::none;

  std::array<std::array<boundary_t, dim>, 2> boundaries =
      get_default_boundaries();

  const auto override_group_boundary = [&](const char *const var_set_string,
                                           const int dir, const int face,
                                           const boundary_t boundary) {
    // Arguments for the callback function
    struct arg_t {
      const int gi;
      const std::array<std::array<bool, dim>, 2> &is_symmetry;
      std::array<std::array<boundary_t, dim>, 2> &boundaries;
      const int dir, face;
      const boundary_t boundary;
    } arg{gi, is_symmetry, boundaries, dir, face, boundary};
    const int ierr = CCTK_TraverseString(
        var_set_string,
        [](const int vi, const char *const optstring, void *argptr) {
          auto &arg = *static_cast<arg_t *>(argptr);
          // Ignore any symmetry boundaries; we are never applying outer
          // boundary conditions there
          if (arg.is_symmetry[arg.face][arg.dir])
            return;
          // If the group index matches, override the respective boundary
          if (CCTK_GroupIndexFromVarI(vi) == arg.gi)
            arg.boundaries[arg.face][arg.dir] = arg.boundary;
        },
        &arg, CCTK_GROUP);
    assert(ierr >= 0);
  };
  override_group_boundary(dirichlet_x_vars, 0, 0, boundary_t::dirichlet);
  override_group_boundary(dirichlet_y_vars, 1, 0, boundary_t::dirichlet);
  override_group_boundary(dirichlet_z_vars, 2, 0, boundary_t::dirichlet);
  override_group_boundary(dirichlet_upper_x_vars, 0, 1, boundary_t::dirichlet);
  override_group_boundary(dirichlet_upper_y_vars, 1, 1, boundary_t::dirichlet);
  override_group_boundary(dirichlet_upper_z_vars, 2, 1, boundary_t::dirichlet);
  override_group_boundary(linear_extrapolation_x_vars, 0, 0,
                          boundary_t::linear_extrapolation);
  override_group_boundary(linear_extrapolation_y_vars, 1, 0,
                          boundary_t::linear_extrapolation);
  override_group_boundary(linear_extrapolation_z_vars, 2, 0,
                          boundary_t::linear_extrapolation);
  override_group_boundary(linear_extrapolation_upper_x_vars, 0, 1,
                          boundary_t::linear_extrapolation);
  override_group_boundary(linear_extrapolation_upper_y_vars, 1, 1,
                          boundary_t::linear_extrapolation);
  override_group_boundary(linear_extrapolation_upper_z_vars, 2, 1,
                          boundary_t::linear_extrapolation);
  override_group_boundary(neumann_x_vars, 0, 0, boundary_t::neumann);
  override_group_boundary(neumann_y_vars, 1, 0, boundary_t::neumann);
  override_group_boundary(neumann_z_vars, 2, 0, boundary_t::neumann);
  override_group_boundary(neumann_upper_x_vars, 0, 1, boundary_t::neumann);
  override_group_boundary(neumann_upper_y_vars, 1, 1, boundary_t::neumann);
  override_group_boundary(neumann_upper_z_vars, 2, 1, boundary_t::neumann);
  override_group_boundary(robin_x_vars, 0, 0, boundary_t::robin);
  override_group_boundary(robin_y_vars, 1, 0, boundary_t::robin);
  override_group_boundary(robin_z_vars, 2, 0, boundary_t::robin);
  override_group_boundary(robin_upper_x_vars, 0, 1, boundary_t::robin);
  override_group_boundary(robin_upper_y_vars, 1, 1, boundary_t::robin);
  override_group_boundary(robin_upper_z_vars, 2, 1, boundary_t::robin);

  return boundaries;
}

bool get_group_checkpoint_flag(const gi_t gi) {
  int tags = CCTK_GroupTagsTableI(gi.index());
  assert(tags >= 0);
  char buf[100];
  int iret = Util_TableGetString(tags, sizeof buf, buf, "checkpoint");
  if (iret == UTIL_ERROR_TABLE_NO_SUCH_KEY) {
    return true;
  } else if (iret >= 0) {
    std::string str(buf);
    for (auto &c : str)
      c = tolower(c);
    if (str == "yes")
      return true;
    if (str == "no")
      return false;
    assert(0);
  } else {
    assert(0);
  }
}

std::optional<bool> get_group_restrict_tag(const gi_t gi) {
  int tags = CCTK_GroupTagsTableI(gi.index());
  assert(tags >= 0);
  char buf[100];
  int iret = Util_TableGetString(tags, sizeof buf, buf, "restrict");
  if (iret == UTIL_ERROR_TABLE_NO_SUCH_KEY) {
    return std::nullopt;
  } else if (iret >= 0) {
    std::string str(buf);
    for (auto &c : str)
      c = tolower(c);
    if (str == "yes")
      return true;
    if (str == "no")
      return false;
    assert(0);
  } else {
    assert(0);
  }
}

bool get_group_evolve_flag(const gi_t gi) {
  int tags = CCTK_GroupTagsTableI(gi.index());
  assert(tags >= 0);
  char buf[100];
  int iret = Util_TableGetString(tags, sizeof buf, buf, "evolve");
  if (iret == UTIL_ERROR_TABLE_NO_SUCH_KEY) {
    // Default to checkpoint flag value when not explicitly set
    return get_group_checkpoint_flag(gi);
  } else if (iret >= 0) {
    std::string str(buf);
    for (auto &c : str)
      c = tolower(c);
    if (str == "yes")
      return true;
    if (str == "no")
      return false;
    assert(0);
  } else {
    assert(0);
  }
}

std::array<int, dim> get_group_indextype(const int gi) {
  DECLARE_CCTK_PARAMETERS;

  assert(gi >= 0);

  const int tags = CCTK_GroupTagsTableI(gi);
  assert(tags >= 0);
  std::array<CCTK_INT, dim> index;

  // The CST stage doesn't look for the `index` tag, and
  // `CCTK_ARGUMENTSX_...` would thus ignore it
  int iret = Util_TableGetIntArray(tags, dim, index.data(), "index");
  if (iret != UTIL_ERROR_TABLE_NO_SUCH_KEY)
    CCTK_VERROR(
        "The grid function group %s has a tag `index=...`. This is not "
        "supported any more; use a `CENTERING{...}` declaration instead.",
        CCTK_FullGroupName(gi));

  // Use the centering table
  const int centering = CCTK_GroupCenteringTableI(gi);
  assert(centering >= 0);
  iret = Util_TableGetIntArray(centering, dim, index.data(), "centering");
  if (iret == UTIL_ERROR_TABLE_NO_SUCH_KEY) {
    // Default: vertex-centred
    index = {0, 0, 0};
  } else if (iret >= 0) {
    assert(iret == dim);
  } else {
    assert(0);
  }

  // Convert to index type
  std::array<int, dim> indextype;
  for (int d = 0; d < dim; ++d)
    indextype[d] = index[d];

  return indextype;
}

std::optional<std::array<gi_t, dim> > get_group_fluxes(const gi_t gi) {
  assert(gi.index() >= 0);
  const int tags = CCTK_GroupTagsTableI(gi.index());
  assert(tags >= 0);
  std::vector<char> fluxes_buf(1000);
  const int iret =
      Util_TableGetString(tags, fluxes_buf.size(), fluxes_buf.data(), "fluxes");
  if (iret == UTIL_ERROR_TABLE_NO_SUCH_KEY) {
    fluxes_buf[0] = '\0'; // default: empty (no fluxes)
  } else if (iret >= 0) {
    // do nothing
  } else {
    assert(0);
  }

  const std::string str(fluxes_buf.data());
  std::vector<std::string> strs;
  std::size_t end = 0;
  while (end < str.size()) {
    const std::size_t begin = str.find_first_not_of(' ', end);
    if (begin == std::string::npos)
      break;
    end = str.find(' ', begin);
    strs.push_back(str.substr(begin, end - begin));
  }

  if (strs.empty())
    return std::nullopt;

  assert(strs.size() == dim); // Check number of fluxes
  const auto flux = [&](const int d) {
    auto str1 = strs[d];
    if (str1.find(':') == std::string::npos) {
      const char *const impl = CCTK_GroupImplementationI(gi.index());
      str1 = std::string(impl) + "::" + str1;
    }
    const gi_t gi1{CCTK_GroupIndex(str1.c_str())};
    assert(gi1.index() >= 0);
    return gi1;
  };
  static_assert(dim == 3);
  const std::array<gi_t, dim> fluxes{flux(0), flux(1), flux(2)};

  for (int d = 0; d < dim; ++d)
    for (int d1 = d + 1; d1 < dim; ++d1)
      assert(fluxes[d] != fluxes[d1]); // Check groups are all different
  return fluxes;
}

bool group_is_integrated(const gi_t gi) {
  assert(gi.index() >= 0 && gi.index() < CCTK_NumGroups());
  const std::optional<rk_config_t> &integrator = ghext->rk();
  return integrator && integrator->is_integrated(gi);
}

int num_rk_stages() {
  const std::optional<rk_config_t> &integrator = ghext->rk();
  return integrator ? integrator->nstages() : max_num_rk_stages;
}

rk_dense_output_t registered_dense_output() {
  const std::optional<rk_config_t> &integrator = ghext->rk();
  return integrator ? integrator->dense_output : rk_dense_output_t::none;
}

bool group_has_flux_register(const gi_t gi) {
  assert(gi.index() >= 0 && gi.index() < CCTK_NumGroups());
  return ghext->do_reflux && CCTK_GroupTypeI(gi.index()) == CCTK_GF &&
         group_is_integrated(gi) && get_group_fluxes(gi).has_value();
}

void warn_inert_flux_tags() {
  DECLARE_CCTK_PARAMETERS;
  const std::optional<rk_config_t> &integrator = ghext->rk();
  if (max_num_levels <= 1)
    return;
  if (CCTK_MyProc(nullptr) != 0)
    return;
  if (!ghext->do_reflux) {
    for (int gi0 = 0; gi0 < CCTK_NumGroups(); ++gi0) {
      const gi_t gi{gi0};
      if (CCTK_GroupTypeI(gi.index()) == CCTK_GF && integrator &&
          integrator->is_integrated(gi) && get_group_fluxes(gi)) {
        CCTK_INFO("Grid functions integrated by ODESolvers carry fluxes= "
                  "tags, but refluxing at coarse-fine boundaries is off; set "
                  "CarpetX::do_reflux = yes to enable it");
        break;
      }
    }
    return;
  }
  for (int gi0 = 0; gi0 < CCTK_NumGroups(); ++gi0) {
    const gi_t gi{gi0};
    if (CCTK_GroupTypeI(gi.index()) != CCTK_GF || !get_group_fluxes(gi) ||
        group_has_flux_register(gi))
      continue;
    CCTK_VWARN(CCTK_WARN_ALERT,
               "Group \"%s\" has a fluxes= tag but is not integrated by "
               "ODESolvers: no flux register is allocated for it, and it is "
               "not refluxed at coarse-fine boundaries (see "
               "CarpetX::do_reflux)",
               CCTK_FullGroupName(gi.index()));
  }
}

std::array<int, dim> get_group_nghostzones(const int gi) {
  DECLARE_CCTK_PARAMETERS;
  assert(gi >= 0);
  const int tags = CCTK_GroupTagsTableI(gi);
  assert(tags >= 0);
  std::array<CCTK_INT, dim> nghostzones;
  int iret =
      Util_TableGetIntArray(tags, dim, nghostzones.data(), "nghostzones");
  if (iret == UTIL_ERROR_TABLE_NO_SUCH_KEY) {
    // default: use driver parameter
    nghostzones = {ghost_size >= 0 ? ghost_size : ghost_size_x,
                   ghost_size >= 0 ? ghost_size : ghost_size_y,
                   ghost_size >= 0 ? ghost_size : ghost_size_z};
  } else if (iret >= 0) {
    assert(iret == dim);
  } else {
    assert(0);
  }
  return nghostzones;
}

std::vector<std::array<int, dim> > get_group_parities(const int gi) {
  DECLARE_CCTK_PARAMETERS;
  assert(gi >= 0);
  const int tags = CCTK_GroupTagsTableI(gi);
  assert(tags >= 0);
  const int nelems = Util_TableGetIntArray(tags, 0, nullptr, "parities");
  if (nelems == UTIL_ERROR_TABLE_NO_SUCH_KEY) {
    // unset (will use +1)
    return {};
  } else if (nelems >= 0) {
    // do nothing
  } else {
    assert(0);
  }
  std::vector<CCTK_INT> parities1(nelems);
  const int iret =
      Util_TableGetIntArray(tags, nelems, parities1.data(), "parities");
  assert(iret == nelems);
  assert(nelems % dim == 0);
  std::vector<std::array<int, dim> > parities(nelems / dim);
  for (std::size_t n = 0; n < parities.size(); ++n)
    for (int d = 0; d < dim; ++d)
      parities.at(n).at(d) = parities1.at(dim * n + d);
  return parities;
}

std::vector<CCTK_REAL> get_group_dirichlet_values(const int gi) {
  DECLARE_CCTK_PARAMETERS;
  assert(gi >= 0);
  const int tags = CCTK_GroupTagsTableI(gi);
  assert(tags >= 0);
  const int nelems =
      Util_TableGetRealArray(tags, 0, nullptr, "dirichlet_values");
  if (nelems == UTIL_ERROR_TABLE_NO_SUCH_KEY) {
    // unset (will use zero)
    return {};
  } else if (nelems >= 0) {
    // do nothing
  } else {
    assert(0);
  }
  std::vector<CCTK_REAL> dirichlet_values(nelems);
  const int iret = Util_TableGetRealArray(tags, nelems, dirichlet_values.data(),
                                          "dirichlet_values");
  assert(iret == nelems);
  return dirichlet_values;
}

std::vector<CCTK_REAL> get_group_robin_values(const int gi) {
  DECLARE_CCTK_PARAMETERS;
  assert(gi >= 0);
  const int tags = CCTK_GroupTagsTableI(gi);
  assert(tags >= 0);
  const int nelems = Util_TableGetRealArray(tags, 0, nullptr, "robin_values");
  if (nelems == UTIL_ERROR_TABLE_NO_SUCH_KEY) {
    // unset (will use zero)
    return {};
  } else if (nelems >= 0) {
    // do nothing
  } else {
    assert(0);
  }
  std::vector<CCTK_REAL> robin_values(nelems);
  const int iret =
      Util_TableGetRealArray(tags, nelems, robin_values.data(), "robin_values");
  assert(iret == nelems);
  return robin_values;
}

std::string get_group_prolongation_type(const int gi) {
  DECLARE_CCTK_PARAMETERS;
  assert(gi >= 0);
  const int tags = CCTK_GroupTagsTableI(gi);
  assert(tags >= 0);
  std::vector<char> group_prolongation_type_buf(1000);
  const int iret = Util_TableGetString(tags, group_prolongation_type_buf.size(),
                                       group_prolongation_type_buf.data(),
                                       "prolongation_type");
  if (iret == UTIL_ERROR_TABLE_NO_SUCH_KEY)
    return prolongation_type; // unset (use global default)
  assert(iret >= 0);
  const std::string group_prolongation_type(group_prolongation_type_buf.data());
  return group_prolongation_type;
}

int get_group_prolongation_order(const int gi) {
  DECLARE_CCTK_PARAMETERS;
  assert(gi >= 0);
  const int tags = CCTK_GroupTagsTableI(gi);
  assert(tags >= 0);
  CCTK_INT group_prolongation_order;
  int iret =
      Util_TableGetInt(tags, &group_prolongation_order, "prolongation_order");
  if (iret == UTIL_ERROR_TABLE_NO_SUCH_KEY) {
    // default: use driver parameter
    group_prolongation_order = prolongation_order;
  } else if (iret >= 0) {
    assert(iret == 1);
  } else {
    assert(0);
  }
  return group_prolongation_order;
}

amrex::Interpolater *get_interpolator(const std::string prolongation_type,
                                      const int prolongation_order,
                                      const std::array<int, dim> indextype) {
  const int indextype_scalar =
      (indextype[0] << 2) | (indextype[1] << 1) | (indextype[2] << 0);

  const std::map<int, std::array<amrex::Interpolater *, 8> > *operators =
      nullptr;

  if (prolongation_type == "interpolate")
    operators = &prolongate_poly_3d_rf2;
  else if (prolongation_type == "conservative")
    operators = &prolongate_cons_3d_rf2;
  else if (prolongation_type == "ddf")
    operators = &prolongate_ddf_3d_rf2;
  else if (prolongation_type == "eno")
    operators = &prolongate_eno_3d_rf2;
  else if (prolongation_type == "minmod")
    operators = &prolongate_minmod_3d_rf2;
  else if (prolongation_type == "hermite")
    operators = &prolongate_hermite_3d_rf2;
  else if (prolongation_type == "natural")
    operators = &prolongate_natural_3d_rf2;
  else if (prolongation_type == "poly-cons3lfb")
    operators = &prolongate_poly_cons3lfb_3d_rf2;
  else if (prolongation_type == "poly-eno3lfb")
    operators = &prolongate_poly_eno3lfb_3d_rf2;
  else
    CCTK_VERROR("Unsupported prolongation type %s", prolongation_type.c_str());

  if (operators->count(prolongation_order) == 0)
    CCTK_VERROR("Unsupported prolongation order %d for prolongation type %s",
                prolongation_order, prolongation_type.c_str());

  return operators->at(prolongation_order).at(indextype_scalar);
}

amrex::Interpolater *
get_interpolator(const GHExt::PatchData::LevelData::GroupData &groupdata) {
  DECLARE_CCTK_PARAMETERS;

  const std::string group_prolongation_type =
      get_group_prolongation_type(groupdata.record.groupindex.index());
  const int group_prolongation_order =
      get_group_prolongation_order(groupdata.record.groupindex.index());
  const std::array<int, dim> &group_indextype = groupdata.indextype;

  return get_interpolator(group_prolongation_type, group_prolongation_order,
                          group_indextype);
}

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

GHExt::PatchData::PatchData(const int patch) : patch(patch) {
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
        patch, &is_cartesian1, dim, ncells1, xmin1, xmax1);
    assert(!ierr);
    is_cartesian = is_cartesian1;
    for (int d = 0; d < dim; ++d)
      ncells[d] = ncells1[d];
    domain = amrex::RealBox(xmin1, xmax1);
  }

  const int coord = -1; // undefined?

  // Refinement ratios
  const amrex::Vector<amrex::IntVect> reffacts{}; // empty

  // Symmetries
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
GHExt::PatchData::add_level(const int level, const amrex::BoxArray &ba,
                            const amrex::DistributionMapping &dm,
                            const std::function<std::string()> &why) {
  assert(level == num_levels());
  return leveldata.emplace_back(patch, level, ba, dm, why);
}

GHExt::PatchData::LevelData GHExt::PatchData::replace_level(const int level,
                                                            LevelData &&fresh) {
  assert(fresh.patch == patch && fresh.level == level);
  using std::swap;
  swap(this->level(level), fresh);
  return std::move(fresh);
}

void GHExt::PatchData::erase_levels_from(const int level) {
  assert(level >= 0 && level <= num_levels());
  leveldata.erase(leveldata.begin() + level, leveldata.end());
}

GHExt::PatchData::LevelData::LevelData(const int patch, const int level,
                                       const amrex::BoxArray &ba,
                                       const amrex::DistributionMapping &dm,
                                       const std::function<std::string()> &why)
    : patch(patch), level(level) {
  DECLARE_CCTK_PARAMETERS;

  const int timereffact = ghext->use_subcycling ? 2 : 1;
  if (level == 0) {
    // We are creating the coarsest level
    is_subcycling_level = false; // unused
    iteration = 0;
    delta_iteration = 1;
    last_step_iteration = 0;
    last_finalized_iteration = 0;
  } else {
    auto &coarseleveldata = ghext->level(patch, level - 1);
    is_subcycling_level = ghext->use_subcycling;
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
    const GHExt::GroupRecord &group = ghext->record(gi_t{gi0});
    if (group.kind != group_kind::grid_function)
      groupdata.emplace_back(nullptr);
    else
      groupdata.push_back(
          std::make_unique<GHExt::PatchData::LevelData::GroupData>(
              group, patch, level, ba, dm, why));
  }

  for (int gi0 = 0; gi0 < numgroups; ++gi0) {
    const gi_t gi{gi0};
    const GHExt::GroupRecord &group = ghext->record(gi);
    if (group.kind != group_kind::grid_function)
      continue;
    const std::optional<std::array<gi_t, dim> > fluxes = get_group_fluxes(gi);
    if (fluxes) {
      for (int d = 0; d < dim; ++d) {
        assert((*fluxes)[d] != gi);
        std::array<int, dim> flux_indextype{1, 1, 1};
        flux_indextype[d] = 0;
        assert(get_group_indextype((*fluxes)[d].index()) == flux_indextype);
        assert(ghext->record((*fluxes)[d]).numvars >= group.numvars);
      }
    }
  }

  // Set up cctkGHs
  if (int(ghext->level_cctkGHs.size()) <= level) {
    cGH *const cctkGH = ghext->get_global_cctkGH();
    enter_level_mode(cctkGH, level);
    assert(int(ghext->level_cctkGHs.size()) == level);
    ghext->level_cctkGHs.emplace_back(copy_cctkGH(cctkGH));
    leave_level_mode(cctkGH, level);
  }

  {
    LevelData &leveldata = *this;
    cGH *const cctkGH = ghext->get_level_cctkGH(level);
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
    const GroupRecord &record, const int patch, const int level,
    const amrex::BoxArray &ba, const amrex::DistributionMapping &dm,
    const std::function<std::string()> &why)
    : record(record), patch(patch), level(level) {
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

  indextype = get_group_indextype(gi);
  nghostzones = get_group_nghostzones(gi);

  interpolator = get_interpolator(*this); // reads groupindex, indextype

  const auto &geom = ghext->patch(patch).amrcore->Geom(level);
  for (int d = 0; d < dim; ++d)
    if (geom.isPeriodic(d))
      assert(geom.Domain().length(d) >= nghostzones[d]);

  boundaries = get_group_boundaries(gi);
  parities = get_group_parities(gi);
  if (parities.empty()) {
    std::array<int, dim> parity;
    for (int d = 0; d < dim; ++d)
      parity[d] = +1;
    parities.resize(numvars, parity);
  }
  assert(int(parities.size()) == numvars);
  for (int vi = 0; vi < numvars; ++vi)
    for (int d = 0; d < dim; ++d)
      assert(abs(parities.at(vi)[d]) == 1);

  dirichlet_values = get_group_dirichlet_values(gi);
  if (dirichlet_values.empty())
    dirichlet_values.resize(numvars, 0);
  robin_values = get_group_robin_values(gi);
  if (robin_values.empty())
    robin_values.resize(numvars, 0);

  amrex::BCRec bcrec;
  for (int f = 0; f < 2; ++f) {
    for (int d = 0; d < dim; ++d) {
      const auto bc =
          ghext->patch(patch).symmetries[f][d] == symmetry_t::periodic
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
    if (level > 0 && group_has_flux_register(groupindex)) {
      const amrex::IntVect ratio =
          ghext->patch(patch).amrcore->refRatio(level - 1);
      freg =
          std::make_unique<amrex::FluxRegister>(gba, dm, ratio, level, numvars);
      // FluxRegister::define leaves the FabSets uninitialized. Zero them so
      // that a register that is never fed (e.g. ODESolvers::method =
      // "constant") is a no-op in Reflux and serializes as zeros at a
      // mid-cycle checkpoint.
      freg->setVal(0);
      // The coarse side's staging buffer (AccumulateFluxes): the register's
      // layout by construction. Left uninitialized; every use zeroes it.
      freg_scratch =
          std::make_unique<amrex::FluxRegister>(gba, dm, ratio, level, numvars);
    }
  }
}

////////////////////////////////////////////////////////////////////////////////

bool all_levels_synchronized() {
  if (!ghext->use_subcycling)
    return true;
  std::optional<rat64> ref;
  for (const auto &patchdata : ghext->patches())
    for (const auto &leveldata : patchdata.levels()) {
      if (!ref)
        ref = leveldata.iteration;
      else if (leveldata.iteration != *ref)
        return false;
    }
  return true;
}

// Some level finer than (patch, level) is behind it, i.e. the window that
// holds `level` and the finest level has not been restricted at `level`'s
// clock yet: that happens once every level from `level` down to the finest is
// time-aligned (see the evolve loop and window_is_restricted). The bands hold
// this level's last step, which the child reads until that restriction. Under
// 2:1 time refinement the child is either half a coarse step behind this
// level, and its next substep reads the bands, or aligned with it, and
// recovery reads them for the child's end-of-step ghosts (ODESolvers refills
// those by dense output from the bands until that restriction, and after it
// by spatial prolongation from the restricted parent, as the restriction did
// for every evolved group). The pair (level, level + 1) is refluxed right
// before that restriction, so its flux register is live until then; the pair
// may itself be aligned with the register complete but unapplied.
bool recovered_finer_level_is_behind(const int patch, const int level) {
  if (!ghext->use_subcycling)
    return false;
  const auto &recovered = ghext->recovered_levels;
  if (patch < 0 || patch >= int(recovered.size()))
    return false;
  const auto &levels = recovered.at(patch);
  if (level < 0 || level + 1 >= int(levels.size()))
    return false; // finest level: nothing finer
  const std::optional<rat64> &self = levels.at(level).iteration;
  if (!self)
    return false; // old checkpoint without per-level iteration: time-aligned
  for (int finer = level + 1; finer < int(levels.size()); ++finer) {
    const std::optional<rat64> &finer_iteration = levels.at(finer).iteration;
    if (finer_iteration && *finer_iteration < *self)
      return true;
  }
  return false;
}

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

amrex::MultiFab *rk_source_band(const int patch, const int level,
                                const gi_t gi, const band_kind kind,
                                const int stage) {
  const auto &patchdata = ghext->patch(patch);
  assert(level >= 0);
  if (level + 1 >= patchdata.num_levels())
    return nullptr;
  const auto &childleveldata = patchdata.level(level + 1);
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
  const auto &symmetries = ghext->patch(patch).symmetries;
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

  const amrex::Geometry &geom = ghext->patch(patch).amrcore->Geom(level);

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

// amrex::AmrCore functions

CactusAmrCore::CactusAmrCore() {}
CactusAmrCore::CactusAmrCore(const int patch, const amrex::RealBox *rb,
                             int max_level_in,
                             const amrex::Vector<int> &n_cell_in, int coord,
                             amrex::Vector<amrex::IntVect> ref_ratios,
                             const int *is_per)
    : amrex::AmrCore(rb, max_level_in, n_cell_in, coord, ref_ratios, is_per),
      patch(patch) {}

CactusAmrCore::CactusAmrCore(const int patch, const amrex::RealBox &rb,
                             int max_level_in,
                             const amrex::Vector<int> &n_cell_in, int coord,
                             amrex::Vector<amrex::IntVect> const &ref_ratios,
                             amrex::Array<int, AMREX_SPACEDIM> const &is_per)
    : amrex::AmrCore(rb, max_level_in, n_cell_in, coord, ref_ratios, is_per),
      patch(patch) {}

CactusAmrCore::~CactusAmrCore() {}

void CactusAmrCore::ErrorEst(const int level, amrex::TagBoxArray &tags,
                             const amrex::Real time, const int ngrow) {
  DECLARE_CCTK_PARAMETERS;

  // Don't regrid before Cactus is ready to
  if (!cactus_is_initialized)
    return;

  if (verbose)
#pragma omp critical
    CCTK_VINFO("ErrorEst patch %d level %d", patch, level);

  const gi_t gi{CCTK_GroupIndex("CarpetXRegrid::regrid_error")};
  assert(gi.index() >= 0);
  const int vi = 0;
  const int tl = 0;

  auto &restrict leveldata = ghext->level(patch, level);
  auto &restrict groupdata = leveldata.group(gi);
  require(active_levels_t(level, level + 1, patch, patch + 1),
          {.gi = gi,
           .tl = tl,
           .vi = vi,
           .where = make_valid_int(),
           .event = "ErrorEst"});
  std::size_t npoints_set = 0, npoints_clear = 0, npoints_total = 0;
  auto mfitinfo = amrex::MFItInfo().SetDynamic(true).EnableTiling();
#pragma omp parallel
  for (amrex::MFIter mfi(*leveldata.fab, mfitinfo); mfi.isValid(); ++mfi) {
    GridPtrDesc1 grid(leveldata, groupdata, mfi);

    const amrex::Array4<const CCTK_REAL> &err_array4 =
        groupdata.timelevel(tl).mfab->array(mfi);
    const GF3D1<const CCTK_REAL> &err_ = grid.gf3d(err_array4, vi);
    const amrex::Array4<char> &tags_array4 = tags.array(mfi);

    std::size_t npoints_set_local = 0, npoints_clear_local = 0,
                npoints_total_local = 0;
    grid.loop_idx(
        where_t::interior, groupdata.indextype, [&](const Loop::PointDesc &p) {
          const int tag = err_(p.I) >= regrid_error_threshold
                              ? amrex::TagBox::SET
                              : amrex::TagBox::CLEAR;
          npoints_set_local += tag == amrex::TagBox::SET;
          npoints_clear_local += tag == amrex::TagBox::CLEAR;
          ++npoints_total_local;
          tags_array4(grid.cactus_offset.x + p.i, grid.cactus_offset.y + p.j,
                      grid.cactus_offset.z + p.k) = tag;
        });
    // Do not set the boundary; AMReX's error grid function might have a
    // different number of ghost zones, and these ghost zones are unused anyway.

#pragma omp atomic update
    npoints_set += npoints_set_local;
#pragma omp atomic update
    npoints_clear += npoints_clear_local;
#pragma omp atomic update
    npoints_total += npoints_total_local;
  }

  if (verbose)
#pragma omp critical
    CCTK_VINFO("ErrorEst patch %d level %d done. "
               "Set/clear/total=%td/%td/%td=%.0f%%/%.0f%%/%.0f%%",
               patch, level, npoints_set, npoints_clear, npoints_total,
               100.0 * npoints_set / npoints_total,
               100.0 * npoints_clear / npoints_total, 100.0);
}

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
    for (int tl = 0; tl < group.num_timelevels(); ++tl)
      invalidate_array({.gi = group.groupindex,
                        .tl = tl,
                        .where = make_valid_int(),
                        .event = "freeze_storage"});
  }
}

void CactusAmrCore::SetupLevel(const int level, const amrex::BoxArray &ba,
                               const amrex::DistributionMapping &dm,
                               const std::function<std::string()> &why) {
  DECLARE_CCTK_PARAMETERS;

  if (verbose)
#pragma omp critical
    CCTK_VINFO("SetupLevel patch %d level %d", patch, level);

  ghext->patch(patch).add_level(level, ba, dm, why);

  if (level >= int(level_modified.size()))
    level_modified.resize(level + 1, true);
  level_modified.at(level) = true;
  carpetx_epoch.fetch_add(1, std::memory_order_relaxed); // Increment epoch

  if (verbose)
#pragma omp critical
    CCTK_VINFO("SetupLevel patch %d level %d done.", patch, level);
}

amrex::BoxArray CactusAmrCore::RechopLevel(const int level,
                                           amrex::BoxArray ba) const {
  if (level == 0)
    // Level 0 always covers the full domain: regenerate the canonical
    // cold-start decomposition (maxSize + optional ChopGrids over the whole
    // domain). Independent of the checkpoint partition, so it yields larger OR
    // smaller boxes purely as max_grid_size / NProcs dictate.
    return MakeBaseGrids();

  // Levels > 0: coalesce the checkpoint union into a (near-)minimal box set
  // first, so re-chopping can MERGE into larger boxes on scale-down — not just
  // split into smaller ones — then tile to max_grid_size and load-balance.
  amrex::BoxList bl = ba.boxList();
  while (bl.simplify()) // greedily merge abutting boxes to a fixpoint
    ;
  ba = amrex::BoxArray(std::move(bl));
  ba.maxSize(maxGridSize(level)); // tile to current max_grid_size
  if (refine_grid_layout)         // load-balance to current node count
    ChopGrids(level, ba, amrex::ParallelDescriptor::NProcs());
  return ba;
}

void CactusAmrCore::MakeNewLevelFromScratch(
    int level, amrex::Real time, const amrex::BoxArray &ba,
    const amrex::DistributionMapping &dm) {
  DECLARE_CCTK_PARAMETERS;

  if (verbose)
#pragma omp critical
    CCTK_VINFO("MakeNewLevelFromScratch patch %d level %d", patch, level);

  SetupLevel(level, ba, dm, []() { return "MakeNewLevelFromScratch"; });

  const auto &leveldata = ghext->level(patch, level);
  const active_levels_t active_levels(level, level + 1, patch, patch + 1);
  for (const auto &groupdata : leveldata.groups()) {
    for (int tl = 0; tl < groupdata.num_timelevels(); ++tl)
      invalidate(active_levels, {.gi = groupdata.record.groupindex,
                                 .tl = tl,
                                 .where = make_valid_all(),
                                 .event = "MakeNewLevelFromScratch"});
  }

  if (verbose)
#pragma omp critical
    CCTK_VINFO("MakeNewLevelFromScratch patch %d level %d done.", patch, level);
}

static int
regrid_prolongate_tls(const GHExt::PatchData::LevelData::GroupData &groupdata) {
  const int ntls = groupdata.num_timelevels();
  return groupdata.record.do_evolve       ? (ntls > 1 ? ntls - 1 : ntls)
         : groupdata.record.do_checkpoint ? ntls
                                          : 0;
}

static bool
regrid_source_valid(const GHExt::PatchData::LevelData::GroupData &groupdata,
                    const int tl) {
  for (int vi = 0; vi < groupdata.record.numvars; ++vi)
    if (!groupdata.timelevel(tl).valid(vi).valid_all())
      return false;
  return true;
}

static valid_t
regrid_fills(const GHExt::PatchData::LevelData::GroupData &groupdata) {
  return make_valid_int() | make_valid_ghosts() |
         (groupdata.all_faces_have_symmetries_or_boundaries()
              ? make_valid_outer()
              : valid_t());
}

void CactusAmrCore::MakeNewLevelFromCoarse(
    const int level, const amrex::Real time, const amrex::BoxArray &ba,
    const amrex::DistributionMapping &dm) {
  DECLARE_CCTK_PARAMETERS;

  if (verbose)
#pragma omp critical
    CCTK_VINFO("MakeNewLevelFromCoarse patch %d level %d", patch, level);

  assert(level > 0);

  SetupLevel(level, ba, dm, []() { return "MakeNewLevelFromCoarse"; });

  auto &patchdata = ghext->patch(patch);
  auto &leveldata = patchdata.level(level);
  auto &coarseleveldata = patchdata.level(level - 1);
  const active_levels_t active_levels(level, level + 1, patch, patch + 1);
  const active_levels_t active_coarse_levels(level - 1, level, patch,
                                             patch + 1);

  // only allow creation of new levels when the source and target are aligned
  // in time
  if (leveldata.iteration != coarseleveldata.iteration) {
    std::ostringstream msg;
    msg << "Coarse (rl=" << (level - 1) << ", it=" << coarseleveldata.iteration
        << ") and fine (rl=" << level << ", it=" << leveldata.iteration
        << ") grid do not align in time when regridding";
#pragma omp critical
    CCTK_VERROR(msg.str().c_str());
  }

  for (auto &groupdata : leveldata.groups()) {
    const gi_t gi = groupdata.record.groupindex;
    auto &coarsegroupdata = coarseleveldata.group(gi);
    assert(&coarsegroupdata.record == &groupdata.record);
    amrex::Interpolater *const interpolator = groupdata.interpolator;

    const int ntls = groupdata.num_timelevels();
    const int prolongate_tl = regrid_prolongate_tls(groupdata);
    const valid_t fills = regrid_fills(groupdata);

    for (int tl = 0; tl < ntls; ++tl) {
      if (tl < prolongate_tl) {
        const bool do_fill = groupdata.record.do_evolve ||
                             regrid_source_valid(coarsegroupdata, tl);
        if (do_fill) {
          require(active_coarse_levels,
                  {.gi = gi,
                   .tl = tl,
                   .where = make_valid_all(),
                   .event = "MakeNewLevelFromCoarse before prolongation"});
          const write_scope_t scope = open_write(
              active_levels,
              {.gi = gi,
               .tl = tl,
               .fills = fills,
               .keeps = {},
               .event = "MakeNewLevelFromCoarse after prolongation"});
          FillPatch_NewLevel(
              groupdata, coarsegroupdata, *groupdata.timelevel(tl).mfab,
              *coarsegroupdata.timelevel(tl).mfab,
              patchdata.amrcore->Geom(level - 1),
              patchdata.amrcore->Geom(level), interpolator, groupdata.bcrecs);
        } else {
          invalidate(active_levels,
                     {.gi = gi,
                      .tl = tl,
                      .where = make_valid_all(),
                      .event = "MakeNewLevelFromCoarse: not prolongated "
                               "because source was invalid at regrid time"});
        }
      } else {
        invalidate(active_levels,
                   {.gi = gi,
                    .tl = tl,
                    .where = make_valid_all(),
                    .event = groupdata.record.do_evolve
                                 ? "MakeNewLevelFromCoarse: oldest time level "
                                   "is not prolongated at regrid"
                                 : "MakeNewLevelFromCoarse: not prolongated "
                                   "because variable is not evolved"});
      }
    } // for tl
  }

  if (verbose)
#pragma omp critical
    CCTK_VINFO("MakeNewLevelFromCoarse patch %d level %d done.", patch, level);
}

void CactusAmrCore::RemakeLevel(const int level, const amrex::Real time,
                                const amrex::BoxArray &ba,
                                const amrex::DistributionMapping &dm) {
  DECLARE_CCTK_PARAMETERS;

  if (verbose)
#pragma omp critical
    CCTK_VINFO("RemakeLevel patch %d level %d", patch, level);

  auto &patchdata = ghext->patch(patch);
  auto &leveldata = patchdata.level(level);
  assert(leveldata.level > 0);
  auto &coarseleveldata = patchdata.level(level - 1);
  const active_levels_t active_levels(level, level + 1, patch, patch + 1);
  const active_levels_t active_coarse_levels(level - 1, level, patch,
                                             patch + 1);

  // only allow modfication of levels when the source and target are aligned in
  // time
  if (leveldata.iteration != coarseleveldata.iteration) {
    std::ostringstream msg;
    msg << "Coarse (rl=" << (level - 1) << ", it=" << coarseleveldata.iteration
        << ") and fine (rl=" << level << ", it=" << leveldata.iteration
        << ") grid do not align in time when regridding";
#pragma omp critical
    CCTK_VERROR(msg.str().c_str());
  }

  const auto transports =
      [](const GHExt::PatchData::LevelData::GroupData &coarsegroupdata,
         const GHExt::PatchData::LevelData::GroupData &oldgroupdata,
         const int tl) {
        return tl < regrid_prolongate_tls(oldgroupdata) &&
               (oldgroupdata.record.do_evolve ||
                (regrid_source_valid(coarsegroupdata, tl) &&
                 regrid_source_valid(oldgroupdata, tl)));
      };

  for (const auto &groupdata : leveldata.groups()) {
    const gi_t gi = groupdata.record.groupindex;
    const auto &coarsegroupdata = coarseleveldata.group(gi);
    assert(&coarsegroupdata.record == &groupdata.record);

    const int ntls = groupdata.num_timelevels();
    for (int tl = 0; tl < ntls; ++tl) {
      if (!transports(coarsegroupdata, groupdata, tl))
        continue;
      require(active_coarse_levels,
              {.gi = gi,
               .tl = tl,
               .where = make_valid_all(),
               .event = "RemakeLevel before prolongation"});
      require(active_levels, {.gi = gi,
                              .tl = tl,
                              .where = make_valid_all(),
                              .event = "RemakeLevel before prolongation"});
    } // for tl
  }

  const amrex::IntVect nghostzones = {
      ghost_size >= 0 ? ghost_size : ghost_size_x,
      ghost_size >= 0 ? ghost_size : ghost_size_y,
      ghost_size >= 0 ? ghost_size : ghost_size_z};
  leveldata.fab = std::make_unique<amrex::FabArrayBase>(ba, dm, 1, nghostzones);
  assert(ba.ixType() == amrex::IndexType(amrex::IndexType::CELL,
                                         amrex::IndexType::CELL,
                                         amrex::IndexType::CELL));

  // We assume that this level is at the same time as the next coarser level
  assert(leveldata.iteration == coarseleveldata.iteration);

  GHExt::PatchData::LevelData oldleveldata = patchdata.replace_level(
      level, GHExt::PatchData::LevelData(patch, level, ba, dm,
                                         []() { return "RemakeLevel"; }));

  for (auto &groupdata : leveldata.groups()) {
    const gi_t gi = groupdata.record.groupindex;
    auto &oldgroupdata = oldleveldata.group(gi);
    auto &coarsegroupdata = coarseleveldata.group(gi);
    assert(&coarsegroupdata.record == &groupdata.record);
    amrex::Interpolater *const interpolator = groupdata.interpolator;

    const valid_t fills = regrid_fills(groupdata);
    assert(fills == make_valid_all());

    const int ntls = groupdata.num_timelevels();
    const int prolongate_tl = regrid_prolongate_tls(groupdata);

    for (int tl = 0; tl < ntls; ++tl) {
      if (transports(coarsegroupdata, oldgroupdata, tl)) {
        const write_scope_t scope = open_write(
            active_levels, {.gi = gi,
                            .tl = tl,
                            .fills = fills,
                            .keeps = {},
                            .event = "RemakeLevel after prolongation"});
        FillPatch_RemakeLevel(
            groupdata, coarsegroupdata, *groupdata.timelevel(tl).mfab,
            *coarsegroupdata.timelevel(tl).mfab,
            *oldgroupdata.timelevel(tl).mfab,
            patchdata.amrcore->Geom(level - 1), patchdata.amrcore->Geom(level),
            interpolator, groupdata.bcrecs);
      } else {
        invalidate(active_levels,
                   {.gi = gi,
                    .tl = tl,
                    .where = make_valid_all(),
                    .event = tl < prolongate_tl
                                 ? "RemakeLevel: not prolongated because "
                                   "source was invalid at regrid time"
                                 : "RemakeLevel"});
      }
    } // for tl
  }

  level_modified.at(level) = true;
  carpetx_epoch.fetch_add(1, std::memory_order_relaxed); // Increment epoch

  if (verbose)
#pragma omp critical
    CCTK_VINFO("RemakeLevel patch %d level %d done.", patch, level);
}

void CactusAmrCore::ClearLevel(const int level) {
  DECLARE_CCTK_PARAMETERS;

  if (verbose)
#pragma omp critical
    CCTK_VINFO("ClearLevel patch %d level %d", patch, level);
  ghext->patch(patch).erase_levels_from(level);

  level_modified.resize(level);
  carpetx_epoch.fetch_add(1, std::memory_order_relaxed); // Increment epoch

  if (verbose)
#pragma omp critical
    CCTK_VINFO("ClearLevel patch %d level %d done.", patch, level);
}

////////////////////////////////////////////////////////////////////////////////

template <typename T, size_t N>
inline std::vector<T> seq(const std::array<T, N> &v) {
  std::vector<T> r;
  for (const auto &x : v)
    r.push_back(x);
  return r;
}

template <typename T, size_t N>
inline std::vector<std::vector<T> >
seqs(const std::vector<std::array<T, N> > &v) {
  std::vector<std::vector<T> > r;
  for (const auto &x : v)
    r.push_back(seq(x));
  return r;
}

} // namespace CarpetX

namespace std {
template <typename T, size_t N>
YAML::Emitter &operator<<(YAML::Emitter &yaml, const std::array<T, N> &arr) {
  yaml << YAML::Flow << YAML::BeginSeq;
  for (const auto &elt : arr)
    yaml << elt;
  yaml << YAML::EndSeq;
  return yaml;
}
} // namespace std

namespace amrex {

YAML::Emitter &operator<<(YAML::Emitter &yaml, const amrex::IntVect &iv) {
  yaml << YAML::Flow << YAML::BeginSeq;
  for (int d = 0; d < AMREX_SPACEDIM; ++d)
    yaml << iv[d];
  yaml << YAML::EndSeq;
  return yaml;
}

YAML::Emitter &operator<<(YAML::Emitter &yaml, const amrex::Box &box) {
  yaml << YAML::LocalTag("box-1.0.0");
  yaml << YAML::Flow << YAML::BeginMap;
  yaml << YAML::Key << "small" << YAML::Value << box.smallEnd();
  yaml << YAML::Key << "big" << YAML::Value << box.bigEnd();
  yaml << YAML::EndMap;
  return yaml;
}

YAML::Emitter &operator<<(YAML::Emitter &yaml, const amrex::RealBox &rbox) {
  yaml << YAML::LocalTag("realbox-1.0.0");
  yaml << YAML::Flow << YAML::BeginMap;
  yaml << YAML::Key << "xlo" << YAML::Value
       << std::vector<amrex::Real>(rbox.lo(), rbox.lo() + AMREX_SPACEDIM);
  yaml << YAML::Key << "xhi" << YAML::Value
       << std::vector<amrex::Real>(rbox.hi(), rbox.hi() + AMREX_SPACEDIM);
  yaml << YAML::EndMap;
  return yaml;
}

YAML::Emitter &operator<<(YAML::Emitter &yaml, const amrex::BoxArray &ba) {
  yaml << YAML::LocalTag("boxarray-1.0.0");
  yaml << YAML::BeginSeq;
  for (int n = 0; n < ba.size(); ++n)
    yaml << ba[n];
  yaml << YAML::EndSeq;
  return yaml;
}

YAML::Emitter &operator<<(YAML::Emitter &yaml, const amrex::Geometry &geom) {
  yaml << YAML::LocalTag("geometry-1.0.0");
  yaml << YAML::BeginMap;
  yaml << YAML::Key << "prob_domain" << YAML::Value << geom.ProbDomain();
  yaml << YAML::Key << "domain" << YAML::Value << geom.Domain();
  yaml << YAML::Key << "is_periodic" << YAML::Value << geom.isPeriodic();
  yaml << YAML::EndMap;
  return yaml;
}

YAML::Emitter &operator<<(YAML::Emitter &yaml,
                          const amrex::DistributionMapping &dm) {
  yaml << YAML::LocalTag("distributionmapping-1.0.0");
  yaml << YAML::BeginMap;
  yaml << YAML::Key << "processorMap" << YAML::Value << YAML::Flow
       << dm.ProcessorMap();
  yaml << YAML::EndMap;
  return yaml;
}

YAML::Emitter &operator<<(YAML::Emitter &yaml, const amrex::FabArrayBase &fab) {
  yaml << YAML::LocalTag("fabarraybase-1.0.0");
  yaml << YAML::BeginMap;
  yaml << YAML::Key << "ixType" << YAML::Value << fab.ixType().toIntVect();
  yaml << YAML::Key << "nGrowVect" << YAML::Value << fab.nGrowVect();
  yaml << YAML::Key << "boxArray" << YAML::Value << fab.boxArray();
  yaml << YAML::EndMap;
  return yaml;
}

YAML::Emitter &operator<<(YAML::Emitter &yaml, const amrex::AmrCore &amrcore) {
  yaml << YAML::LocalTag("amrcore-1.0.0");
  yaml << YAML::BeginMap;
  yaml << YAML::Key << "maxLevel" << YAML::Value << amrcore.maxLevel();
  yaml << YAML::Key << "finestLevel" << YAML::Value << amrcore.finestLevel();
  yaml << YAML::Key << "geometry" << YAML::Value << amrcore.Geom();
  yaml << YAML::Key << "distributionMapping" << YAML::Value
       << amrcore.DistributionMap();
  yaml << YAML::Key << "boxArray" << YAML::Value << amrcore.boxArray();
  yaml << YAML::EndMap;
  return yaml;
}

} // namespace amrex
namespace CarpetX {

std::ostream &
operator<<(std::ostream &os,
           const GHExt::GlobalData::AnyTypeVector::AnyTypeScalarRef &scalar) {
  const char sep = '\t';
  switch (scalar._vect.type()) {
  case CCTK_VARIABLE_REAL:
    os << *(CCTK_REAL *)scalar._vect.data_at(scalar._idx);
    break;
  case CCTK_VARIABLE_INT:
    os << *(CCTK_INT *)scalar._vect.data_at(scalar._idx);
    break;
  case CCTK_VARIABLE_COMPLEX: {
    CCTK_COMPLEX value = *(CCTK_COMPLEX *)scalar._vect.data_at(scalar._idx);
    os << value.real() << sep << value.imag();
  } break;
  default:
    assert(0 && "Unexpected variable type");
    break;
  }
  return os;
}

YAML::Emitter &operator<<(YAML::Emitter &yaml,
                          const GHExt::GroupRecord &group) {
  yaml << YAML::LocalTag("commongroupdata-1.0.0");
  yaml << YAML::BeginMap;
  yaml << YAML::Key << "groupname" << YAML::Value
       << CCTK_FullGroupName(group.groupindex.index());
  yaml << YAML::Key << "numvars" << YAML::Value << group.numvars;
  yaml << YAML::Key << "varnames" << YAML::Value << YAML::Flow
       << YAML::BeginSeq;
  for (int vi = 0; vi < group.numvars; ++vi)
    yaml << CCTK_VarName(group.firstvarindex + vi);
  yaml << YAML::EndSeq;
  yaml << YAML::Key << "do_checkpoint" << YAML::Value << group.do_checkpoint;
  yaml << YAML::Key << "do_evolve" << YAML::Value << group.do_evolve;
  yaml << YAML::Key << "do_restrict" << YAML::Value << group.do_restrict;
  yaml << YAML::Key << "restrict_on_sync" << YAML::Value
       << group.restrict_on_sync;
  yaml << YAML::Key << "active_timelevels" << YAML::Value
       << (ghext ? ghext->active_timelevels.at(group.groupindex.index()) : -1);
  yaml << YAML::EndMap;
  return yaml;
}

YAML::Emitter &operator<<(YAML::Emitter &yaml, const CCTK_COMPLEX &cval) {
  yaml << YAML::Flow << YAML::BeginSeq << cval.real() << cval.imag()
       << YAML::EndSeq;
  return yaml;
};

YAML::Emitter &
operator<<(YAML::Emitter &yaml,
           const GHExt::GlobalData::AnyTypeVector::AnyTypeScalarRef
               &anytypescalarref) {
  switch (anytypescalarref._vect.type()) {
  case CCTK_VARIABLE_COMPLEX:
    yaml << *(const CCTK_COMPLEX *)anytypescalarref._vect.data_at(
        anytypescalarref._idx);
    break;
  case CCTK_VARIABLE_REAL:
    yaml << *(const CCTK_REAL *)anytypescalarref._vect.data_at(
        anytypescalarref._idx);
    break;
  case CCTK_VARIABLE_INT:
    yaml << *(const CCTK_INT *)anytypescalarref._vect.data_at(
        anytypescalarref._idx);
    break;
  default:
    // missed to implement a type
    CCTK_VERROR("Cannot handle type %d", anytypescalarref._vect.type());
    break;
  }
  return yaml;
}

YAML::Emitter &
operator<<(YAML::Emitter &yaml,
           const GHExt::GlobalData::AnyTypeVector &anytypevector) {
  yaml << YAML::BeginSeq;
  for (size_t i = 0; i < anytypevector.size(); ++i)
    yaml << anytypevector[i];
  yaml << YAML::EndSeq;
  return yaml;
}

YAML::Emitter &
operator<<(YAML::Emitter &yaml,
           const GHExt::GlobalData::ArrayGroupData &arraygroupdata) {
  yaml << YAML::LocalTag("arraygroupdata-1.0.0");
  yaml << YAML::BeginMap;
  yaml << YAML::Key << "commongroupdata" << YAML::Value
       << arraygroupdata.record;
  yaml << YAML::Key << "valid" << YAML::Value << YAML::BeginSeq;
  for (const auto &timelevel : arraygroupdata.tls) {
    yaml << YAML::BeginSeq;
    for (int vi = 0; vi < arraygroupdata.record.numvars; ++vi)
      yaml << timelevel.why_valid(vi);
    yaml << YAML::EndSeq;
  }
  yaml << YAML::EndSeq;
  yaml << YAML::Key << "data" << YAML::Value << YAML::BeginSeq;
  for (const auto &timelevel : arraygroupdata.tls)
    yaml << timelevel.data;
  yaml << YAML::EndSeq;
  yaml << YAML::EndMap;
  return yaml;
}

YAML::Emitter &
operator<<(YAML::Emitter &yaml,
           const GHExt::PatchData::LevelData::GroupData &groupdata) {
  yaml << YAML::LocalTag("groupdata-1.1.0");
  yaml << YAML::BeginMap;
  yaml << YAML::Key << "commongroupdata" << YAML::Value << groupdata.record;
  yaml << YAML::Key << "valid" << YAML::Value << YAML::BeginSeq;
  for (const auto &timelevel : groupdata.tls) {
    yaml << YAML::BeginSeq;
    for (int vi = 0; vi < groupdata.record.numvars; ++vi)
      yaml << timelevel.why_valid(vi);
    yaml << YAML::EndSeq;
  }
  yaml << YAML::EndSeq;
  yaml << YAML::Key << "patch" << YAML::Value << groupdata.patch;
  yaml << YAML::Key << "level" << YAML::Value << groupdata.level;
  yaml << YAML::Key << "indextype" << YAML::Value << YAML::Flow
       << seq(groupdata.indextype);
  yaml << YAML::Key << "nghostzones" << YAML::Value << YAML::Flow
       << seq(groupdata.nghostzones);
  yaml << YAML::Key << "parities" << YAML::Value << YAML::Flow
       << seqs(groupdata.parities);
  yaml << YAML::Key << "fluxes" << YAML::Value << YAML::Flow << YAML::BeginSeq;
  for (int d = 0; d < dim; ++d)
    yaml << (groupdata.fluxes ? CCTK_FullGroupName((*groupdata.fluxes)[d].index())
                              : "");
  yaml << YAML::EndSeq;
  yaml << YAML::EndMap;
  return yaml;
}

YAML::Emitter &operator<<(YAML::Emitter &yaml,
                          const GHExt::PatchData::LevelData &leveldata) {
  yaml << YAML::LocalTag("leveldata-1.1.0");
  yaml << YAML::BeginMap;
  yaml << YAML::Key << "patch" << YAML::Value << leveldata.patch;
  yaml << YAML::Key << "level" << YAML::Value << leveldata.level;
  yaml << YAML::Key << "is_subcycling_level" << YAML::Value
       << leveldata.is_subcycling_level;
  yaml << YAML::Key << "iteration" << YAML::Value << leveldata.iteration;
  yaml << YAML::Key << "delta_iteration" << YAML::Value
       << leveldata.delta_iteration;
  yaml << YAML::Key << "fab" << YAML::Value << *leveldata.fab;
  yaml << YAML::Key << "groupdata" << YAML::Value << YAML::BeginSeq;
  for (const auto &groupdata : leveldata.groupdata)
    if (groupdata)
      yaml << *groupdata;
  yaml << YAML::EndSeq;
  yaml << YAML::EndMap;
  return yaml;
}

YAML::Emitter &operator<<(YAML::Emitter &yaml,
                          const GHExt::PatchData &patchdata) {
  yaml << YAML::LocalTag("patchdata-1.0.0");
  yaml << YAML::BeginMap;
  yaml << YAML::Key << "patch" << YAML::Value << patchdata.patch;
  yaml << YAML::Key << "amrcore" << YAML::Value << *patchdata.amrcore;
  yaml << YAML::Key << "leveldata" << YAML::Value << YAML::BeginSeq;
  for (const auto &leveldata : patchdata.levels())
    yaml << leveldata;
  yaml << YAML::EndSeq;
  yaml << YAML::EndMap;
  return yaml;
}

YAML::Emitter &operator<<(YAML::Emitter &yaml, const GHExt &ghext) {
  yaml << YAML::LocalTag("ghext-2.0.0");
  yaml << YAML::BeginMap;
  yaml << YAML::Key << "globaldata" << YAML::Value;
  yaml << YAML::LocalTag("globaldata-1.0.0");
  yaml << YAML::BeginMap;
  yaml << YAML::Key << "arraygroupdata" << YAML::Value << YAML::BeginSeq;
  for (const auto &group : ghext.group_records)
    if (group.kind != group_kind::grid_function)
      yaml << group.array();
  yaml << YAML::EndSeq;
  yaml << YAML::EndMap;
  yaml << YAML::Key << "patchdata" << YAML::Value << YAML::BeginSeq;
  for (const auto &patchdata : ghext.patches())
    yaml << patchdata;
  yaml << YAML::EndSeq;
  yaml << YAML::EndMap;
  return yaml;
}

std::ostream &operator<<(std::ostream &os, const GHExt &ghext) {
  YAML::Emitter yaml;
  yaml << ghext;
  os << yaml.c_str() << "\n";
  return os;
}

////////////////////////////////////////////////////////////////////////////////

extern "C" int CarpetX_Startup() {
  DECLARE_CCTK_PARAMETERS;

  if (verbose)
#pragma omp critical
    CCTK_VINFO("Startup");

  const std::vector<std::string> features{
#ifdef AMREX_USE_MPI
      "MPI",
#else
      "no MPI",
#endif
#ifdef AMREX_USE_OMP
      "OpenMP",
#else
      "no OpenMP",
#endif
#ifdef AMREX_USE_GPU
      "accelerators",
#else
      "no accelerators",
#endif
#ifdef AMREX_USE_ASSERTION
      "debug",
#else
      "optimized",
#endif
  };
  std::ostringstream buf;
  buf << logo();
  buf << "   AMR driver using AMReX " << amrex::Version() << "\n   ";
  auto sep = "";
  for (const auto &feature : features) {
    buf << sep << feature;
    sep = ", ";
  }
  int ierr = CCTK_RegisterBanner(buf.str().c_str());
  assert(!ierr);

  // Register a GH extension
  ghext_handle = CCTK_RegisterGHExtension("CarpetX");
  assert(ghext_handle >= 0);
  int iret = CCTK_RegisterGHExtensionSetupGH(ghext_handle, SetupGH);
  assert(iret);
  iret = CCTK_RegisterGHExtensionInitGH(ghext_handle, InitGH);
  assert(iret);
  iret = CCTK_RegisterGHExtensionScheduleTraverseGH(ghext_handle,
                                                    ScheduleTraverseGH);
  assert(iret);

  CCTK_OverloadInitialise(Initialise);
  CCTK_OverloadEvolve(Evolve);
  CCTK_OverloadShutdown(Shutdown);
  CCTK_OverloadOutputGH(OutputGH);

  CCTK_OverloadMyProc(MyProc);
  CCTK_OverloadnProcs(nProcs);
  CCTK_OverloadExit(Exit);
  CCTK_OverloadAbort(Abort);
  CCTK_OverloadBarrier(Barrier);

  CCTK_OverloadEnableGroupStorage(EnableGroupStorage);
  CCTK_OverloadDisableGroupStorage(DisableGroupStorage);
  CCTK_OverloadGroupStorageIncrease(GroupStorageIncrease);
  CCTK_OverloadGroupStorageDecrease(GroupStorageDecrease);
  CCTK_OverloadQueryGroupStorageB(QueryGroupStorageB);

  CCTK_OverloadSyncGroupsByDirI(SyncGroupsOnActiveLevels);

  CCTK_OverloadInterpGridArrays(CarpetX_InterpGridArrays);
  CCTK_OverloadArrayGroupSizeB(ArrayGroupSizeB);
  CCTK_OverloadGroupDynamicData(GroupDynamicData);

  iret = CCTK_InterpRegisterOpLocalUniform(InterpLocalUniform, "CarpetX",
                                           CCTK_THORNSTRING);
  assert(!iret && "CCTK_InterpRegisterOpLocalUniform failed");

  return 0;
}

// Set up GH extension
void *SetupGH(tFleshConfig *fc, int convLevel, cGH *restrict cctkGH) {
  DECLARE_CCTK_PARAMETERS;

  if (verbose)
#pragma omp critical
    CCTK_VINFO("SetupGH");

  assert(fc);
  assert(convLevel == 0);
  assert(cctkGH);

  // Initialize AMReX
  amrex::ParmParse pp;

  // Don't catch Unix signals. If signals are caught, we don't get
  // core files.
  pp.add("amrex.signal_handling", 0);
  // Throw exceptions for failing AMReX assertions. With exceptions,
  // we get core files.
  pp.add("amrex.throw_exception", 1);

  // Enable managed memory (i.e. copy data automatically between
  // accelerator and host)
#ifndef AMREX_USE_HIP
  // This option would render the code inefficient on AMD GPUs.
  // However, it is required to enable some thorns (TOVSolverX,
  // TwoPuncturesX, ...) to work on NVIDIA GPUs.
  // We might need to update those thorns to git rid of this option
  // completely at some point.
  pp.add("amrex.the_arena_is_managed", 1);
#endif

  // Set blocking factors via parameter table since AmrMesh needs to
  // know them when its constructor is running, but there are no
  // constructor arguments for them
  amrex::Vector<int> max_grid_size_x_vec(20, -1), max_grid_size_y_vec(20, -1),
      max_grid_size_z_vec(20, -1);
  for (int level = 0; level < 20; ++level) {
    int size_x = max_grid_sizes_x[level];
    int size_y = max_grid_sizes_y[level];
    int size_z = max_grid_sizes_z[level];
    if (size_x == -1)
      size_x = max_grid_size_x;
    if (size_y == -1)
      size_y = max_grid_size_y;
    if (size_z == -1)
      size_z = max_grid_size_z;
    max_grid_size_x_vec.at(level) = size_x;
    max_grid_size_y_vec.at(level) = size_y;
    max_grid_size_z_vec.at(level) = size_z;
  }
  pp.addarr("amr.max_grid_size_x", max_grid_size_x_vec);
  pp.addarr("amr.max_grid_size_y", max_grid_size_y_vec);
  pp.addarr("amr.max_grid_size_z", max_grid_size_z_vec);

  pp.add("amr.refine_grid_layout", refine_grid_layout);
  pp.add("amr.blocking_factor_x", blocking_factor_x);
  pp.add("amr.blocking_factor_y", blocking_factor_y);
  pp.add("amr.blocking_factor_z", blocking_factor_z);
  pp.add("amr.grid_eff", grid_efficiency);
  if (poison_undefined_values) {
    // Tell AMReX to initialize FArrayBoxes with nans
    pp.add("fab.do_initval", true);
    pp.add("fab.init_snan", true);
    amrex::FArrayBox::set_do_initval(true);
    amrex::FArrayBox::set_initval(
        std::numeric_limits<amrex::Real>::signaling_NaN());
  }
  // Set tile size
  pp.addarr(
      "fabarray.mfiter_tile_size",
      std::vector<int>{max_tile_size_x, max_tile_size_y, max_tile_size_z});

  std::vector<char *> args;
  for (std::size_t n = 0; n < 100; ++n)
    if (amrex_parameters[n][0])
      args.push_back(strdup(amrex_parameters[n]));
  int argc = args.size();
  args.push_back(nullptr);
  char **argv = args.data();
  if (verbose) {
#pragma omp critical
    {
      CCTK_VINFO("  Run-time AMReX parameters:");
      for (int n = 0; n < argc; ++n)
        CCTK_VINFO("    [%d]: %s", n, argv[n]);
    }
  }
  pamrex = amrex::Initialize(argc, argv, true, MPI_COMM_WORLD);
  for (auto arg : args)
    free(arg);
  args.clear();

  if (CCTK_IsFunctionAliased("MultiPatch_Setup")) {
    const int ierr = MultiPatch_Setup();
    assert(!ierr);
  }

  // Create grid structure
  ghext = std::make_unique<GHExt>();

  ghext->active_timelevels.assign(CCTK_NumGroups(), 0);
  ghext->storage_frozen = false;

  return ghext.get();
}

int InitGH(cGH *restrict cctkGH) {
  DECLARE_CCTK_PARAMETERS;

  if (verbose)
#pragma omp critical
    CCTK_VINFO("InitGH");

  assert(cctkGH);

  assert(ghext->num_patches() == 0);
  CCTK_INT num_patches = 1;
  if (CCTK_IsFunctionAliased("MultiPatch_GetSystemSpecification")) {
    const int ierr = MultiPatch_GetSystemSpecification(&num_patches);
    assert(!ierr);
  }
  for (int patch = 0; patch < num_patches; ++patch)
    ghext->add_patch(patch);

  ghext->use_subcycling = use_subcycling;
  ghext->do_reflux = do_reflux;

  return 0; // unused
}

// Traverse schedule
int ScheduleTraverseGH(cGH *restrict cctkGH, const char *where) {
  DECLARE_CCTK_PARAMETERS;

  if (verbose)
#pragma omp critical
    CCTK_VINFO("ScheduleTraverseGH iteration %d %s", cctkGH->cctk_iteration,
               where);

  int ierr = CCTK_ScheduleTraverse(where, cctkGH, CallFunction);
  assert(ierr == 0 || ierr == 2);

  return 0; // unused
}

// Shut down driver
void ShutdownADIOS2();
void ShutdownOpenPMD();
extern "C" int CarpetX_Shutdown() {
  DECLARE_CCTK_PARAMETERS;

  if (verbose)
#pragma omp critical
    CCTK_VINFO("Shutdown");

  // Shut down ADIOS2
  ShutdownADIOS2();
  ShutdownOpenPMD();

  if (false) {
    // Should we really do this? Cactus's extension handling mechanism
    // becomes inconsistent once extensions have been unregistered.
    int iret = CCTK_UnregisterGHExtension("CarpetX");
    assert(iret == 0);
  }

  // Deallocate grid hierarchy
  ghext = nullptr;

  // Finalize AMReX
  amrex::Finalize(pamrex);
  pamrex = nullptr;

  return 0;
}

////////////////////////////////////////////////////////////////////////////////

int MyProc(const cGH *restrict cctkGH) {
  if (pamrex) {
    return amrex::ParallelDescriptor::MyProc();
  } else {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    return rank;
  }
}

int nProcs(const cGH *restrict cctkGH) {
  if (pamrex) {
    return amrex::ParallelDescriptor::NProcs();
  } else {
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    return size;
  }
}

int Exit(cGH *cctkGH, int retval) {
  if (pamrex) {
    amrex::ParallelDescriptor::Abort(retval);
  } else {
    MPI_Abort(MPI_COMM_WORLD, retval);
  }
  return 0; // unreachable
}

int Abort(cGH *cctkGH, int retval) {
  if (pamrex) {
    amrex::ParallelDescriptor::Abort(retval);
  } else {
    MPI_Abort(MPI_COMM_WORLD, retval);
  }
  return 0; // unreachable
}

int Barrier(const cGH *restrict cctkGH) {
  assert(pamrex);
  amrex::ParallelDescriptor::Barrier();
  return 0;
}

void CarpetX_CallScheduleGroup(void *cctkGH_, const char *groupname) {
  cGH *cctkGH = static_cast<cGH *>(cctkGH_);
  static Timer timer("CallScheduleGroup");
  Interval interval(timer);
  int ierr = CCTK_ScheduleTraverse(groupname, cctkGH, CallFunction);
  assert(!ierr);
}

CCTK_INT
CarpetX_GetDomainSpecification(const CCTK_INT patch, const CCTK_INT size,
                               CCTK_REAL *restrict const physical_min,
                               CCTK_REAL *restrict const physical_max,
                               CCTK_REAL *restrict const interior_min,
                               CCTK_REAL *restrict const interior_max,
                               CCTK_REAL *restrict const exterior_min,
                               CCTK_REAL *restrict const exterior_max,
                               CCTK_REAL *restrict const spacing) {
  assert(size == dim);
  const auto &patchdata = ghext->patch(patch);
  const auto &fab = *patchdata.level(0).fab;
  const auto &geom = patchdata.amrcore->Geom(0);
  for (int d = 0; d < dim; ++d) {
    // the domain itself (physical faces)
    physical_min[d] = geom.ProbDomain().lo(d);
    physical_max[d] = geom.ProbDomain().hi(d);
    // domain without boundary points (last interior cell centre)
    interior_min[d] =
        geom.ProbDomain().lo(d) + CCTK_REAL(0.5) * geom.CellSize(d);
    interior_max[d] =
        geom.ProbDomain().hi(d) - CCTK_REAL(0.5) * geom.CellSize(d);
    // domain including boundary/ghost points (last boundary/ghost cell centre)
    exterior_min[d] = geom.ProbDomain().lo(d) -
                      (fab.nGrow(d) - CCTK_REAL(0.5)) * geom.CellSize(d);
    exterior_max[d] = geom.ProbDomain().hi(d) +
                      (fab.nGrow(d) - CCTK_REAL(0.5)) * geom.CellSize(d);
    spacing[d] = geom.CellSize(d);
  }
  return 0;
}

CCTK_INT CarpetX_GetBoundarySizesAndTypes(
    const void *const cctkGH_, const CCTK_INT patch, const CCTK_INT size,
    CCTK_INT *restrict const bndsize, CCTK_INT *restrict const is_ghostbnd,
    CCTK_INT *restrict const is_symbnd, CCTK_INT *restrict const is_physbnd) {
  DECLARE_CCTK_PARAMETERS;

  const cGH *restrict const cctkGH = static_cast<const cGH *>(cctkGH_);
  assert(size == 2 * dim);

  const std::array<std::array<bool, 3>, 2> is_periodic{{
      {{bool(periodic_x), bool(periodic_y), bool(periodic_z)}},
      {{bool(periodic_x), bool(periodic_y), bool(periodic_z)}},
  }};
  const std::array<std::array<bool, 3>, 2> is_reflection{{
      {{bool(reflection_x), bool(reflection_y), bool(reflection_z)}},
      {{bool(reflection_upper_x), bool(reflection_upper_y),
        bool(reflection_upper_z)}},
  }};
  const std::array<std::array<bool, 3>, 2> is_boundary{{
      {{
          bool(!CCTK_EQUALS(boundary_x, "none")),
          bool(!CCTK_EQUALS(boundary_y, "none")),
          bool(!CCTK_EQUALS(boundary_z, "none")),
      }},
      {{
          bool(!CCTK_EQUALS(boundary_upper_x, "none")),
          bool(!CCTK_EQUALS(boundary_upper_y, "none")),
          bool(!CCTK_EQUALS(boundary_upper_z, "none")),
      }},
  }};

  for (int d = 0; d < dim; ++d) {
    for (int f = 0; f < 2; ++f) {
      bndsize[2 * d + f] = cctkGH->cctk_nghostzones[d];
      is_ghostbnd[2 * d + f] = cctkGH->cctk_bbox[2 * d + f];
      is_symbnd[2 * d + f] =
          !is_ghostbnd[2 * d + f] &&
          (is_periodic[f][d] || is_reflection[f][d] || is_boundary[f][d]);
      is_physbnd[2 * d + f] = !is_ghostbnd[2 * d + f] && !is_symbnd[2 * d + f];
    }
  }
  return 0;
}

const int *ArrayGroupSizeB(const cGH *cctkGH, int dir, int gi,
                           const char *groupname) {
  if (groupname) {
    gi = CCTK_GroupIndex(groupname);
  }
  assert(gi >= 0 and gi < CCTK_NumGroups());

  // TODO: handle case of no storage allocated

  cGroup group;
  int ierr = CCTK_GroupData(gi, &group);
  assert(!ierr);
  assert(dir >= 0 && dir < group.dim);
  if (group.grouptype == CCTK_GF) {
    return &cctkGH->cctk_ash[dir];
  } else if (group.grouptype == CCTK_SCALAR or group.grouptype == CCTK_ARRAY) {
    return &ghext->record(gi_t{gi}).array().ash[dir];
  } else {
    CCTK_VERROR("Internal error: unexpected group type %d for group '%s'",
                (int)group.grouptype, CCTK_FullGroupName(gi));
  }
}

int GroupDynamicData(const cGH *cctkGH, int gi, cGroupDynamicData *data) {
  if (not cctkGH)
    return -3;
  if (not(gi >= 0 and gi < CCTK_NumGroups()))
    return -78;
  if (not data)
    return -1;
  cGroup group;
  int ierr = CCTK_GroupData(gi, &group);
  assert(!ierr);
  if (group.grouptype == CCTK_GF) {
    data->dim = group.dim;
    data->lsh = cctkGH->cctk_lsh;
    data->ash = cctkGH->cctk_ash;
    data->gsh = cctkGH->cctk_gsh;
    data->lbnd = cctkGH->cctk_lbnd;
    data->ubnd = cctkGH->cctk_ubnd;
    data->tile_min = cctkGH->cctk_tile_min;
    data->tile_max = cctkGH->cctk_tile_max;
    data->bbox = cctkGH->cctk_bbox;
    data->nghostzones = cctkGH->cctk_nghostzones;
    data->activetimelevels = CCTK_ActiveTimeLevelsGI(cctkGH, gi);
  } else if (group.grouptype == CCTK_SCALAR or group.grouptype == CCTK_ARRAY) {
    const auto &arraygroupdata = ghext->record(gi_t{gi}).array();
    data->dim = arraygroupdata.dimension;
    data->lsh = arraygroupdata.lsh;
    data->ash = arraygroupdata.ash;
    data->gsh = arraygroupdata.gsh;
    data->lbnd = arraygroupdata.lbnd;
    data->ubnd = arraygroupdata.ubnd;
    data->bbox = arraygroupdata.bbox;
    data->nghostzones = arraygroupdata.nghostzones;
    data->activetimelevels = arraygroupdata.record.num_timelevels();
  } else {
    CCTK_VERROR("Internal error: unexpected group type %d for group '%s'",
                (int)group.grouptype, CCTK_FullGroupName(gi));
  }
  return 0;
}

} // namespace CarpetX
