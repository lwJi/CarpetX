#include "hierarchy/group_tags.hxx"

#include "hierarchy/driver.hxx"

#include <cctk_Parameters.h>
#include <util_Table.h>

#include <cassert>
#include <cctype>
#include <cstddef>
#include <ostream>
#include <string>

namespace CarpetX {

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

std::array<std::array<symmetry_t, dim>, 2>
get_symmetries(const std::optional<patch_t> patch) {
  DECLARE_CCTK_PARAMETERS;

  std::array<std::array<bool, 3>, 2> is_interpatch{
      {{{false, false, false}}, {{false, false, false}}}};
  if (patch && CCTK_IsFunctionAliased("MultiPatch_GetBoundarySpecification2")) {
    CCTK_INT is_interpatch_boundary[2 * dim];
    const int ierr = MultiPatch_GetBoundarySpecification2(
        patch->index(), 2 * dim, is_interpatch_boundary);
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
      get_symmetries(std::nullopt);
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

std::array<std::array<boundary_t, dim>, 2> get_group_boundaries(const gi_t gi) {
  DECLARE_CCTK_PARAMETERS;

  const std::array<std::array<symmetry_t, dim>, 2> symmetries =
      get_symmetries(std::nullopt);
  std::array<std::array<bool, 3>, 2> is_symmetry;
  for (int f = 0; f < 2; ++f)
    for (int d = 0; d < dim; ++d)
      is_symmetry[f][d] = symmetries[f][d] != symmetry_t::none;

  std::array<std::array<boundary_t, dim>, 2> boundaries =
      get_default_boundaries();

  const auto override_group_boundary = [&](const char *const var_set_string,
                                           const int dir, const int face,
                                           const boundary_t boundary) {
    struct arg_t {
      const int gi;
      const std::array<std::array<bool, dim>, 2> &is_symmetry;
      std::array<std::array<boundary_t, dim>, 2> &boundaries;
      const int dir, face;
      const boundary_t boundary;
    } arg{gi.index(), is_symmetry, boundaries, dir, face, boundary};
    const int ierr = CCTK_TraverseString(
        var_set_string,
        [](const int vi, const char *const optstring, void *argptr) {
          auto &arg = *static_cast<arg_t *>(argptr);
          if (arg.is_symmetry[arg.face][arg.dir])
            return;
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

std::array<int, dim> get_group_indextype(const gi_t gi) {
  DECLARE_CCTK_PARAMETERS;

  assert(gi.index() >= 0);

  const int tags = CCTK_GroupTagsTableI(gi.index());
  assert(tags >= 0);
  std::array<CCTK_INT, dim> index;

  int iret = Util_TableGetIntArray(tags, dim, index.data(), "index");
  if (iret != UTIL_ERROR_TABLE_NO_SUCH_KEY)
    CCTK_VERROR(
        "The grid function group %s has a tag `index=...`. This is not "
        "supported any more; use a `CENTERING{...}` declaration instead.",
        CCTK_FullGroupName(gi.index()));

  const int centering = CCTK_GroupCenteringTableI(gi.index());
  assert(centering >= 0);
  iret = Util_TableGetIntArray(centering, dim, index.data(), "centering");
  if (iret == UTIL_ERROR_TABLE_NO_SUCH_KEY) {
    index = {0, 0, 0};
  } else if (iret >= 0) {
    assert(iret == dim);
  } else {
    assert(0);
  }

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
    fluxes_buf[0] = '\0';
  } else if (iret >= 0) {
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

  assert(strs.size() == dim);
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
      assert(fluxes[d] != fluxes[d1]);
  return fluxes;
}

bool group_is_integrated(const gi_t gi) {
  assert(gi.index() >= 0 && gi.index() < CCTK_NumGroups());
  const std::optional<rk_config_t> &integrator = driver().rk();
  return integrator && integrator->is_integrated(gi);
}

int num_rk_stages() {
  const std::optional<rk_config_t> &integrator = driver().rk();
  return integrator ? integrator->nstages() : max_num_rk_stages;
}

rk_dense_output_t registered_dense_output() {
  const std::optional<rk_config_t> &integrator = driver().rk();
  return integrator ? integrator->dense_output : rk_dense_output_t::none;
}

bool group_has_flux_register(const gi_t gi) {
  assert(gi.index() >= 0 && gi.index() < CCTK_NumGroups());
  return driver().do_reflux && CCTK_GroupTypeI(gi.index()) == CCTK_GF &&
         group_is_integrated(gi) && get_group_fluxes(gi).has_value();
}

void warn_inert_flux_tags() {
  DECLARE_CCTK_PARAMETERS;
  const GHExt &ext = driver();
  const std::optional<rk_config_t> &integrator = ext.rk();
  if (max_num_levels <= 1)
    return;
  if (CCTK_MyProc(nullptr) != 0)
    return;
  if (!ext.do_reflux) {
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

std::array<int, dim> get_group_nghostzones(const gi_t gi) {
  DECLARE_CCTK_PARAMETERS;
  assert(gi.index() >= 0);
  const int tags = CCTK_GroupTagsTableI(gi.index());
  assert(tags >= 0);
  std::array<CCTK_INT, dim> nghostzones;
  int iret =
      Util_TableGetIntArray(tags, dim, nghostzones.data(), "nghostzones");
  if (iret == UTIL_ERROR_TABLE_NO_SUCH_KEY) {
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

std::vector<std::array<int, dim> > get_group_parities(const gi_t gi) {
  DECLARE_CCTK_PARAMETERS;
  assert(gi.index() >= 0);
  const int tags = CCTK_GroupTagsTableI(gi.index());
  assert(tags >= 0);
  const int nelems = Util_TableGetIntArray(tags, 0, nullptr, "parities");
  if (nelems == UTIL_ERROR_TABLE_NO_SUCH_KEY) {
    return {};
  } else if (nelems >= 0) {
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

std::vector<CCTK_REAL> get_group_dirichlet_values(const gi_t gi) {
  DECLARE_CCTK_PARAMETERS;
  assert(gi.index() >= 0);
  const int tags = CCTK_GroupTagsTableI(gi.index());
  assert(tags >= 0);
  const int nelems =
      Util_TableGetRealArray(tags, 0, nullptr, "dirichlet_values");
  if (nelems == UTIL_ERROR_TABLE_NO_SUCH_KEY) {
    return {};
  } else if (nelems >= 0) {
  } else {
    assert(0);
  }
  std::vector<CCTK_REAL> dirichlet_values(nelems);
  const int iret = Util_TableGetRealArray(tags, nelems, dirichlet_values.data(),
                                          "dirichlet_values");
  assert(iret == nelems);
  return dirichlet_values;
}

std::vector<CCTK_REAL> get_group_robin_values(const gi_t gi) {
  DECLARE_CCTK_PARAMETERS;
  assert(gi.index() >= 0);
  const int tags = CCTK_GroupTagsTableI(gi.index());
  assert(tags >= 0);
  const int nelems = Util_TableGetRealArray(tags, 0, nullptr, "robin_values");
  if (nelems == UTIL_ERROR_TABLE_NO_SUCH_KEY) {
    return {};
  } else if (nelems >= 0) {
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
    return prolongation_type;
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
    group_prolongation_order = prolongation_order;
  } else if (iret >= 0) {
    assert(iret == 1);
  } else {
    assert(0);
  }
  return group_prolongation_order;
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

} // namespace CarpetX
