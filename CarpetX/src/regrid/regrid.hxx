#ifndef CARPETX_CARPETX_REGRID_REGRID_HXX
#define CARPETX_CARPETX_REGRID_REGRID_HXX

#include "internal.hxx"

#include "hierarchy/driver.hxx"
#include <carpetx.hxx>

#include <AMReX_AmrCore.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_RealBox.H>
#include <AMReX_TagBox.H>
#include <AMReX_Vector.H>

#include <functional>
#include <memory>
#include <string>

namespace CarpetX {

class CactusAmrCore final : public amrcore_t {
  patch_t patch;

public:
  CactusAmrCore(patch_t patch, const amrex::RealBox &rb, int max_level_in,
                const amrex::Vector<int> &n_cell_in, int coord,
                amrex::Vector<amrex::IntVect> const &ref_ratios,
                amrex::Array<int, AMREX_SPACEDIM> const &is_per);
  CactusAmrCore(const amrex::AmrCore &rhs) = delete;
  CactusAmrCore &operator=(const amrex::AmrCore &rhs) = delete;

  virtual ~CactusAmrCore() override;

  virtual void ErrorEst(int level, amrex::TagBoxArray &tags, amrex::Real time,
                        int ngrow) override;
  void SetupLevel(level_t level, const amrex::BoxArray &ba,
                  const amrex::DistributionMapping &dm,
                  const std::function<std::string()> &why);
  virtual void make_level(level_t level, const amrex::BoxArray &ba,
                          const amrex::DistributionMapping &dm,
                          const std::string &why) override;
  virtual void
  MakeNewLevelFromScratch(int level, amrex::Real time,
                          const amrex::BoxArray &ba,
                          const amrex::DistributionMapping &dm) override;
  virtual void
  MakeNewLevelFromCoarse(int level, amrex::Real time, const amrex::BoxArray &ba,
                         const amrex::DistributionMapping &dm) override;
  virtual void RemakeLevel(int level, amrex::Real time,
                           const amrex::BoxArray &ba,
                           const amrex::DistributionMapping &dm) override;
  virtual void ClearLevel(int level) override;
};

std::unique_ptr<amrcore_t>
make_cactus_amrcore(patch_t patch, const amrex::RealBox &domain, int,
                    const amrex::Vector<int> &ncells,
                    const amrex::Array<int, dim> &is_periodic);

} // namespace CarpetX

#endif
