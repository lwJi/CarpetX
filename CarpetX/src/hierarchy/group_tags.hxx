#ifndef CARPETX_CARPETX_HIERARCHY_GROUP_TAGS_HXX
#define CARPETX_CARPETX_HIERARCHY_GROUP_TAGS_HXX

#include "internal.hxx"

#include "hierarchy/driver.hxx"

#include <carpetx.hxx>
#include <carpetx_rk.hxx>

#include <cctk.h>

#include <AMReX_Interpolater.H>

#include <array>
#include <optional>
#include <string>
#include <vector>

namespace CarpetX {

std::array<std::array<symmetry_t, dim>, 2>
get_symmetries(std::optional<patch_t> patch);
std::array<std::array<boundary_t, dim>, 2> get_default_boundaries();
std::array<std::array<boundary_t, dim>, 2> get_group_boundaries(gi_t gi);

bool get_group_checkpoint_flag(gi_t gi);
std::optional<bool> get_group_restrict_tag(gi_t gi);
bool get_group_evolve_flag(gi_t gi);
std::array<int, dim> get_group_indextype(gi_t gi);
std::optional<std::array<gi_t, dim> > get_group_fluxes(gi_t gi);

bool group_is_integrated(gi_t gi);
int num_rk_stages();
rk_dense_output_t registered_dense_output();
bool group_has_flux_register(gi_t gi);
void warn_inert_flux_tags();

std::array<int, dim> get_group_nghostzones(gi_t gi);
std::vector<std::array<int, dim> > get_group_parities(gi_t gi);
std::vector<CCTK_REAL> get_group_dirichlet_values(gi_t gi);
std::vector<CCTK_REAL> get_group_robin_values(gi_t gi);

amrex::Interpolater *get_interpolator(std::string prolongation_type,
                                      int prolongation_order,
                                      std::array<int, dim> indextype);
amrex::Interpolater *
get_interpolator(const GHExt::PatchData::LevelData::GroupData &groupdata);

} // namespace CarpetX

#endif
