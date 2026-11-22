#include "integrated_groups.hxx"
#include "rk_methods.hxx"

#include <carpetx_rk.hxx>

#include <cctk.h>
#include <cctk_Arguments.h>
#include <cctk_Parameters.h>

#include <algorithm>
#include <cassert>

namespace ODESolvers {

extern "C" void ODESolvers_CheckTimelevels(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTS_ODESolvers_CheckTimelevels;
  DECLARE_CCTK_PARAMETERS;

  for (const CarpetX::rk_pair_t &pair : integrated_groups().pairs) {
    const int gi = pair.state.index();
    const int ntls = CCTK_ActiveTimeLevelsGI(cctkGH, gi);
    if (ntls >= 2)
      CCTK_VERROR("ODESolvers subcycling requires evolved groups to have a "
                  "single timelevel, but group \"%s\" has %d active "
                  "timelevels. Subcycling does not support timelevels >= 2 "
                  "for evolution variables.",
                  CCTK_FullGroupName(gi), ntls);
  }
}

extern "C" void ODESolvers_CheckMethod(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTS_ODESolvers_CheckMethod;
  DECLARE_CCTK_PARAMETERS;

  const cParamData *const method_param =
      CCTK_ParameterData("method", "ODESolvers");
  assert(method_param);
  for (const t_range *keyword = method_param->range; keyword;
       keyword = keyword->next) {
    if (!CCTK_Equals(keyword->origin, "ODESolvers"))
      continue;
    const bool has_row = std::any_of(
        rk_methods.begin(), rk_methods.end(), [&](const rk_method_t &row) {
          return CCTK_Equals(keyword->range, row.name);
        });
    if (!has_row)
      CCTK_VERROR("ODESolvers::method keyword \"%s\" (ODESolvers/param.ccl) "
                  "has no row in the RK method table "
                  "(ODESolvers/src/rk_methods.hxx)",
                  keyword->range);
  }
  for (const rk_method_t &row : rk_methods) {
    bool has_keyword = false;
    for (const t_range *keyword = method_param->range; keyword;
         keyword = keyword->next)
      if (CCTK_Equals(keyword->origin, "ODESolvers") &&
          CCTK_Equals(keyword->range, row.name))
        has_keyword = true;
    if (!has_keyword)
      CCTK_VERROR("The RK method table (ODESolvers/src/rk_methods.hxx) has a "
                  "row \"%s\" that is not a keyword of ODESolvers::method "
                  "(ODESolvers/param.ccl)",
                  row.name);
  }
}

} // namespace ODESolvers
