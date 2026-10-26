#include "driver.hxx"

#include <cctk.h>
#include <cctk_Arguments.h>
#include <cctk_Parameters.h>

#include <AMReX.H>
#include <AMReX_MLMG.H>
#include <AMReX_MLNodeLaplacian.H>

#include <cmath>
#include <optional>
#include <vector>

namespace CarpetX {

namespace {

void solve_poisson(const gi_t gi_sol, const gi_t gi_rhs,
                   const std::optional<gi_t> gi_res, const CCTK_REAL reltol,
                   const CCTK_REAL abstol, CCTK_REAL *restrict const res_initial,
                   CCTK_REAL *restrict const res_final) {
  const bool have_res = gi_res.has_value();
  const int tl = 0;
  const int vi = 0;

  // Create operator

  assert(ghext->num_patches() == 1);
  const int patch = 0;
  const auto &patchdata = ghext->patch(patch);
  amrex::Vector<amrex::Geometry> geoms(patchdata.num_levels());
  amrex::Vector<amrex::BoxArray> grids(patchdata.num_levels());
  amrex::Vector<amrex::DistributionMapping> dmaps(patchdata.num_levels());
  for (int level = 0; level < patchdata.num_levels(); ++level) {
    geoms.at(level) = patchdata.amrcore->Geom(level);
    grids.at(level) = patchdata.amrcore->boxArray(level);
    dmaps.at(level) = patchdata.amrcore->DistributionMap(level);
  }

  amrex::MLNodeLaplacian mlnodelaplacian(geoms, grids, dmaps);

  // TODO: Look at Geometry for this
  mlnodelaplacian.setDomainBC(
      {amrex::LinOpBCType::Dirichlet, amrex::LinOpBCType::Dirichlet,
       amrex::LinOpBCType::Dirichlet},
      {amrex::LinOpBCType::Dirichlet, amrex::LinOpBCType::Dirichlet,
       amrex::LinOpBCType::Dirichlet});

  std::vector<amrex::MultiFab> sigmas(patchdata.num_levels());
  for (int level = 0; level < patchdata.num_levels(); ++level) {
    auto &sigma = sigmas.at(level);
    sigma.define(patchdata.amrcore->boxArray(level),
                 patchdata.amrcore->DistributionMap(level), 1, 0);
    sigma.setVal(1.0);
    mlnodelaplacian.setSigma(level, sigma);
  }

  mlnodelaplacian.setVerbose(10);

  // Create solver

  amrex::MLMG mlmg(mlnodelaplacian);

  amrex::Vector<amrex::MultiFab *> ress(patchdata.num_levels());
  amrex::Vector<amrex::MultiFab *> sols(patchdata.num_levels());
  amrex::Vector<const amrex::MultiFab *> rhss(patchdata.num_levels());
  for (int level = 0; level < patchdata.num_levels(); ++level) {
    const auto &restrict leveldata = patchdata.level(level);
    const auto &restrict groupdata_rhs = leveldata.group(gi_rhs);
    rhss.at(level) = groupdata_rhs.timelevel(tl).mfab.get();
    const auto &restrict groupdata_sol = leveldata.group(gi_sol);
    sols.at(level) = groupdata_sol.timelevel(tl).mfab.get();
    if (have_res) {
      const auto &restrict groupdata_res = leveldata.group(*gi_res);
      ress.at(level) = groupdata_res.timelevel(tl).mfab.get();
    }
  }

  mlmg.setVerbose(10);
  mlmg.setBottomVerbose(10);

  // Solve

  if (have_res) {
    mlmg.compResidual(ress, sols, rhss);
    *res_initial = 0;
    for (int level = 0; level < patchdata.num_levels(); ++level)
      *res_initial =
          fmax(*res_initial, ress.at(level)->norminf(vi, 0, false, true));
  } else {
    *res_initial = NAN;
  }

#pragma omp critical
  {
    CCTK_VINFO("Before solving:");
    for (int level = 0; level < patchdata.num_levels(); ++level)
      CCTK_VINFO("norm_inf rhs[%d]: %g", level,
                 double(rhss.at(level)->norminf(vi, 0, false, true)));
    for (int level = 0; level < patchdata.num_levels(); ++level)
      CCTK_VINFO("norm_inf sol[%d]: %g", level,
                 double(sols.at(level)->norminf(vi, 0, false, true)));
    if (have_res)
      for (int level = 0; level < patchdata.num_levels(); ++level)
        CCTK_VINFO("norm_inf res[%d]: %g", level,
                   double(ress.at(level)->norminf(vi, 0, false, true)));
  }

  const CCTK_REAL maxerr = mlmg.solve(sols, rhss, reltol, abstol);
#pragma omp critical
  CCTK_VINFO("Solution error (norm_inf): %g", double(maxerr));

  if (have_res) {
    mlmg.compResidual(ress, sols, rhss);
    *res_final = 0;
    for (int level = 0; level < patchdata.num_levels(); ++level)
      *res_final =
          fmax(*res_final, ress.at(level)->norminf(vi, 0, false, true));
  } else {
    *res_final = NAN;
  }

#pragma omp critical
  {
    CCTK_VINFO("After solving:");
    for (int level = 0; level < patchdata.num_levels(); ++level)
      CCTK_VINFO("norm_inf rhs[%d]: %g", level,
                 double(rhss.at(level)->norminf(vi, 0, false, true)));
    for (int level = 0; level < patchdata.num_levels(); ++level)
      CCTK_VINFO("norm_inf sol[%d]: %g", level,
                 double(sols.at(level)->norminf(vi, 0, false, true)));
    if (have_res)
      for (int level = 0; level < patchdata.num_levels(); ++level)
        CCTK_VINFO("norm_inf res[%d]: %g", level,
                   double(ress.at(level)->norminf(vi, 0, false, true)));
  }
}

} // namespace

extern "C" void CarpetX_SolvePoisson(const CCTK_INT gi_sol,
                                     const CCTK_INT gi_rhs,
                                     const CCTK_INT gi_res,
                                     const CCTK_REAL reltol,
                                     const CCTK_REAL abstol,
                                     CCTK_REAL *restrict const res_initial,
                                     CCTK_REAL *restrict const res_final) {
  solve_poisson(gi_t{gi_sol}, gi_t{gi_rhs},
                gi_res >= 0 ? std::optional<gi_t>(gi_t{gi_res}) : std::nullopt,
                reltol, abstol, res_initial, res_final);
}

} // namespace CarpetX
