#include "hierarchy/driver.hxx"
#include "ops/linsolve.hxx"

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

void solve_poisson(const gi_t gi_sol, const gi_t gi_rhs,
                   const std::optional<gi_t> gi_res, const CCTK_REAL reltol,
                   const CCTK_REAL abstol, CCTK_REAL *restrict const res_initial,
                   CCTK_REAL *restrict const res_final) {
  const bool have_res = gi_res.has_value();
  const tl_t tl{0};
  const vi_t vi{0};

  // Create operator

  const GHExt &ext = driver();
  assert(ext.num_patches() == 1);
  const patch_t patch{0};
  const auto &patchdata = ext.patch(patch);
  amrex::Vector<amrex::Geometry> geoms(patchdata.num_levels());
  amrex::Vector<amrex::BoxArray> grids(patchdata.num_levels());
  amrex::Vector<amrex::DistributionMapping> dmaps(patchdata.num_levels());
  for (const level_t level : patchdata.level_handles()) {
    geoms.at(level.index()) = patchdata.amrcore->Geom(level.index());
    grids.at(level.index()) = patchdata.amrcore->boxArray(level.index());
    dmaps.at(level.index()) = patchdata.amrcore->DistributionMap(level.index());
  }

  amrex::MLNodeLaplacian mlnodelaplacian(geoms, grids, dmaps);

  // TODO: Look at Geometry for this
  mlnodelaplacian.setDomainBC(
      {amrex::LinOpBCType::Dirichlet, amrex::LinOpBCType::Dirichlet,
       amrex::LinOpBCType::Dirichlet},
      {amrex::LinOpBCType::Dirichlet, amrex::LinOpBCType::Dirichlet,
       amrex::LinOpBCType::Dirichlet});

  std::vector<amrex::MultiFab> sigmas(patchdata.num_levels());
  for (const level_t level : patchdata.level_handles()) {
    auto &sigma = sigmas.at(level.index());
    sigma.define(patchdata.amrcore->boxArray(level.index()),
                 patchdata.amrcore->DistributionMap(level.index()), 1, 0);
    sigma.setVal(1.0);
    mlnodelaplacian.setSigma(level.index(), sigma);
  }

  mlnodelaplacian.setVerbose(10);

  // Create solver

  amrex::MLMG mlmg(mlnodelaplacian);

  amrex::Vector<amrex::MultiFab *> ress(patchdata.num_levels());
  amrex::Vector<amrex::MultiFab *> sols(patchdata.num_levels());
  amrex::Vector<const amrex::MultiFab *> rhss(patchdata.num_levels());
  for (const level_t level : patchdata.level_handles()) {
    const auto &restrict leveldata = patchdata.level(level);
    const auto &restrict groupdata_rhs = leveldata.group(gi_rhs);
    rhss.at(level.index()) = groupdata_rhs.timelevel(tl).mfab.get();
    const auto &restrict groupdata_sol = leveldata.group(gi_sol);
    sols.at(level.index()) = groupdata_sol.timelevel(tl).mfab.get();
    if (have_res) {
      const auto &restrict groupdata_res = leveldata.group(*gi_res);
      ress.at(level.index()) = groupdata_res.timelevel(tl).mfab.get();
    }
  }

  mlmg.setVerbose(10);
  mlmg.setBottomVerbose(10);

  // Solve

  if (have_res) {
    mlmg.compResidual(ress, sols, rhss);
    *res_initial = 0;
    for (const level_t level : patchdata.level_handles())
      *res_initial =
          fmax(*res_initial,
               ress.at(level.index())->norminf(vi.index(), 0, false, true));
  } else {
    *res_initial = NAN;
  }

#pragma omp critical
  {
    CCTK_VINFO("Before solving:");
    for (const level_t level : patchdata.level_handles())
      CCTK_VINFO(
          "norm_inf rhs[%d]: %g", level.index(),
          double(rhss.at(level.index())->norminf(vi.index(), 0, false, true)));
    for (const level_t level : patchdata.level_handles())
      CCTK_VINFO(
          "norm_inf sol[%d]: %g", level.index(),
          double(sols.at(level.index())->norminf(vi.index(), 0, false, true)));
    if (have_res)
      for (const level_t level : patchdata.level_handles())
        CCTK_VINFO(
            "norm_inf res[%d]: %g", level.index(),
            double(
                ress.at(level.index())->norminf(vi.index(), 0, false, true)));
  }

  const CCTK_REAL maxerr = mlmg.solve(sols, rhss, reltol, abstol);
#pragma omp critical
  CCTK_VINFO("Solution error (norm_inf): %g", double(maxerr));

  if (have_res) {
    mlmg.compResidual(ress, sols, rhss);
    *res_final = 0;
    for (const level_t level : patchdata.level_handles())
      *res_final =
          fmax(*res_final,
               ress.at(level.index())->norminf(vi.index(), 0, false, true));
  } else {
    *res_final = NAN;
  }

#pragma omp critical
  {
    CCTK_VINFO("After solving:");
    for (const level_t level : patchdata.level_handles())
      CCTK_VINFO(
          "norm_inf rhs[%d]: %g", level.index(),
          double(rhss.at(level.index())->norminf(vi.index(), 0, false, true)));
    for (const level_t level : patchdata.level_handles())
      CCTK_VINFO(
          "norm_inf sol[%d]: %g", level.index(),
          double(sols.at(level.index())->norminf(vi.index(), 0, false, true)));
    if (have_res)
      for (const level_t level : patchdata.level_handles())
        CCTK_VINFO(
            "norm_inf res[%d]: %g", level.index(),
            double(
                ress.at(level.index())->norminf(vi.index(), 0, false, true)));
  }
}

} // namespace CarpetX
