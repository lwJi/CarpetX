#include "config/groups.hxx"

#include "config/boundaries.hxx"
#include "config/indices.hxx"
#include <carpetx.hxx>

#include <cctk.h>
#include <cctk_Parameters.h>
#include <util_Table.h>

#include <array>
#include <cassert>
#include <cctype>
#include <cstddef>
#include <cstdlib>
#include <optional>
#include <string>
#include <vector>

namespace CarpetX {

static std::array<std::array<boundary_t, dim>, 2>
get_group_boundaries(const gi_t gi) {
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
      const gi_t gi;
      const std::array<std::array<bool, dim>, 2> &is_symmetry;
      std::array<std::array<boundary_t, dim>, 2> &boundaries;
      const int dir, face;
      const boundary_t boundary;
    } arg{gi, is_symmetry, boundaries, dir, face, boundary};
    const int ierr = CCTK_TraverseString(
        var_set_string,
        [](const int vi, const char *const optstring, void *argptr) {
          auto &arg = *static_cast<arg_t *>(argptr);
          if (arg.is_symmetry[arg.face][arg.dir])
            return;
          if (gi_t{CCTK_GroupIndexFromVarI(vi)} == arg.gi)
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

static std::array<int, dim> get_group_indextype(const gi_t gi) {
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

static std::optional<std::array<gi_t, dim> > get_group_fluxes(const gi_t gi) {
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

static std::array<int, dim> get_group_nghostzones(const gi_t gi) {
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

static std::vector<std::array<int, dim> > get_group_parities(const gi_t gi) {
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

static std::vector<CCTK_REAL> get_group_dirichlet_values(const gi_t gi) {
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

static std::vector<CCTK_REAL> get_group_robin_values(const gi_t gi) {
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

grid_function_t declared_grid_function(const gi_t gi, const int numvars) {
  grid_function_t result;
  result.indextype = get_group_indextype(gi);
  result.nghostzones = get_group_nghostzones(gi);

  result.boundaries = get_group_boundaries(gi);
  result.parities = get_group_parities(gi);
  if (result.parities.empty()) {
    std::array<int, dim> parity;
    for (int d = 0; d < dim; ++d)
      parity[d] = +1;
    result.parities.resize(numvars, parity);
  }
  assert(int(result.parities.size()) == numvars);
  for (const std::array<int, dim> &parity : result.parities)
    for (int d = 0; d < dim; ++d)
      assert(abs(parity[d]) == 1);

  result.dirichlet_values = get_group_dirichlet_values(gi);
  if (result.dirichlet_values.empty())
    result.dirichlet_values.resize(numvars, 0);
  result.robin_values = get_group_robin_values(gi);
  if (result.robin_values.empty())
    result.robin_values.resize(numvars, 0);

  result.fluxes = get_group_fluxes(gi);
  if (result.fluxes)
    assert((result.indextype == std::array<int, dim>{1, 1, 1}));
  return result;
}

std::string get_group_prolongation_type(const gi_t gi) {
  DECLARE_CCTK_PARAMETERS;
  assert(gi.index() >= 0);
  const int tags = CCTK_GroupTagsTableI(gi.index());
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

int get_group_prolongation_order(const gi_t gi) {
  DECLARE_CCTK_PARAMETERS;
  assert(gi.index() >= 0);
  const int tags = CCTK_GroupTagsTableI(gi.index());
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

} // namespace CarpetX
