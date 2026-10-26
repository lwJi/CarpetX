#include <carpetx.hxx>
#include <loop.hxx>

#include <cctk.h>
#include <cctk_Arguments.h>
#include <cctk_Parameters.h>

#include <array>
#include <cassert>
#include <cmath>
#include <sstream>
#include <string>
#include <vector>

namespace TestRestrictTags {
using namespace Loop;

// The groups in the order interface.ccl declares them
constexpr int ngroups = 6;
const std::array<std::string, ngroups> group_names{
    "untagged", "ckpt_no", "nr", "nr_ckpt_no", "ry_ckpt_no", "nr_explicit"};

// Which groups the parameter restricted_groups names
std::array<bool, ngroups> parse_restricted_groups(const char *const names) {
  std::array<bool, ngroups> restricted;
  restricted.fill(false);
  std::istringstream buf(names);
  std::string name;
  while (buf >> name) {
    bool found = false;
    for (int n = 0; n < ngroups; ++n)
      if (name == group_names[n]) {
        restricted[n] = true;
        found = true;
      }
    if (!found)
      CCTK_VERROR("TestRestrictTags::restricted_groups names \"%s\", which is "
                  "not a group of this thorn",
                  name.c_str());
  }
  return restricted;
}

extern "C" void TestRestrictTags_Init(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTSX_TestRestrictTags_Init;
  DECLARE_CCTK_PARAMETERS;

  if (fine_value == coarse_value)
    CCTK_VERROR("TestRestrictTags::fine_value (%g) must differ from "
                "TestRestrictTags::coarse_value (%g)",
                double(fine_value), double(coarse_value));
  if (cctk_level > 1)
    CCTK_VERROR("The test expects 2 levels, but level %d exists", cctk_level);

  const CCTK_REAL value = cctk_level == 0 ? coarse_value : fine_value;
  grid.loop_int<0, 0, 0>(grid.nghostzones, [&](const PointDesc &p) {
    untagged(p.I) = value;
    ckpt_no(p.I) = value;
    nr(p.I) = value;
    nr_ckpt_no(p.I) = value;
    ry_ckpt_no(p.I) = value;
    nr_explicit(p.I) = value;
  });
}

extern "C" void TestRestrictTags_RestrictExplicit(CCTK_ARGUMENTS) {
  const CarpetX::gi_t gi{CCTK_GroupIndex("TestRestrictTags::nr_explicit")};
  assert(gi.index() >= 0);
  const std::array groups{gi};
  CarpetX::restrict_no_poison(CarpetX::active_window(), groups);
}

extern "C" void TestRestrictTags_Sync(CCTK_ARGUMENTS) {
  // Do nothing; the schedule SYNCs
}

extern "C" void TestRestrictTags_Check(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTSX_TestRestrictTags_Check;
  DECLARE_CCTK_PARAMETERS;

  if (CarpetX::num_levels(CarpetX::patch_t{0}) != 2)
    CCTK_VERROR("The test expects 2 levels, but there are %d",
                CarpetX::num_levels(CarpetX::patch_t{0}));
  assert(cctk_level == 0 || cctk_level == 1);
  const bool is_coarse = cctk_level == 0;

  const std::array<bool, ngroups> restricted =
      parse_restricted_groups(restricted_groups);

  // Whether level 1 covers this point
  const auto in_fine_box = [&](const PointDesc &p) {
    for (int d = 0; d < dim; ++d)
      if (std::fabs(p.X[d]) > fine_box_radius + p.DX[d] / 4)
        return false;
    return true;
  };

  // Level 1 must lie in the cube; level 0 then shows that it fills it
  if (!is_coarse) {
    int noutside = 0;
    vect<CCTK_REAL, dim> outside_X{0, 0, 0};
    grid.loop_int<0, 0, 0>(grid.nghostzones, [&](const PointDesc &p) {
      if (!in_fine_box(p)) {
        if (noutside == 0)
          outside_X = p.X;
        ++noutside;
      }
    });
    if (noutside != 0) {
#pragma omp critical(TestRestrictTags_Check)
      CCTK_VERROR("Level 1 has %d interior points on this tile outside the "
                  "cube of half-width TestRestrictTags::fine_box_radius = %g, "
                  "the first at [%g,%g,%g]",
                  noutside, double(fine_box_radius), double(outside_X[0]),
                  double(outside_X[1]), double(outside_X[2]));
    }
  }

  // The first mismatch
  int nbad = 0;
  int bad_group = -1;
  const char *bad_where = nullptr;
  vect<CCTK_REAL, dim> bad_X{0, 0, 0};
  CCTK_REAL bad_value = 0, bad_expected = 0;
  const auto expect = [&](const int n, const char *const where,
                          const PointDesc &p, const CCTK_REAL value,
                          const CCTK_REAL expected) {
    if (!(value == expected)) {
      if (nbad == 0) {
        bad_group = n;
        bad_where = where;
        bad_X = p.X;
        bad_value = value;
        bad_expected = expected;
      }
      ++nbad;
    }
  };

  const auto check = [&](const int n, const auto &gf) {
    grid.loop_int<0, 0, 0>(grid.nghostzones, [&](const PointDesc &p) {
      expect(n, "interior", p, gf(p.I),
             !is_coarse || (restricted[n] && in_fine_box(p)) ? fine_value
                                                             : coarse_value);
    });
    if (!restricted[n])
      grid.loop_all<0, 0, 0>(grid.nghostzones, [&](const PointDesc &p) {
        expect(n, "everywhere", p, gf(p.I),
               !is_coarse && in_fine_box(p) ? fine_value : coarse_value);
      });
  };

  check(0, untagged);
  check(1, ckpt_no);
  check(2, nr);
  check(3, nr_ckpt_no);
  check(4, ry_ckpt_no);
  check(5, nr_explicit);

  if (nbad != 0) {
    // The routine runs once per tile from an OpenMP task list; let only one
    // tile report and abort
#pragma omp critical(TestRestrictTags_Check)
    CCTK_VERROR("Group %s on level %d holds %.17g at [%g,%g,%g] (%s), "
                "expected %.17g (%s is %sin TestRestrictTags::"
                "restricted_groups); %d mismatches on this tile",
                group_names[bad_group].c_str(), cctk_level, double(bad_value),
                double(bad_X[0]), double(bad_X[1]), double(bad_X[2]), bad_where,
                double(bad_expected), group_names[bad_group].c_str(),
                restricted[bad_group] ? "" : "not ", nbad);
  }
}

} // namespace TestRestrictTags
