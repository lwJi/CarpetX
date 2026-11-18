#include "comm/reflux.hxx"

#include "hierarchy/model.hxx"
#include "hierarchy/scopes.hxx"
#include "comm/buffers.hxx"
#include "comm/integrator.hxx"
#include "comm/level_pairs.hxx"
#include <carpetx.hxx>

#include <cctk.h>
#include <cctk_Parameters.h>

#include <AMReX_FluxRegister.H>
#include <AMReX_Geometry.H>
#include <AMReX_IndexType.H>
#include <AMReX_IntVect.H>
#include <AMReX_MultiFab.H>

#include <array>
#include <cassert>
#include <optional>
#include <sstream>

namespace CarpetX {

static bool group_has_flux_tag(const GHExt::GroupRecord &record) {
  return record.kind == group_kind::grid_function &&
         record.grid_function().fluxes.has_value();
}

bool group_has_flux_register(const gi_t gi) {
  const GHExt &ext = driver();
  return ext.do_reflux && group_is_integrated(gi) &&
         group_has_flux_tag(ext.record(gi));
}

void warn_inert_flux_tags() {
  DECLARE_CCTK_PARAMETERS;
  const GHExt &ext = driver();
  const std::optional<GHExt::integrator_t> &integrator = ext.integrator();
  if (max_num_levels <= 1)
    return;
  if (CCTK_MyProc(nullptr) != 0)
    return;
  if (!ext.do_reflux) {
    for (const gi_t gi : index_range<gi_t>(gi_t{0}, gi_t{CCTK_NumGroups()})) {
      if (integrator && integrator->rhs.contains(gi) &&
          group_has_flux_tag(ext.record(gi))) {
        CCTK_INFO("Grid functions integrated by ODESolvers carry fluxes= "
                  "tags, but refluxing at coarse-fine boundaries is off; set "
                  "CarpetX::do_reflux = yes to enable it");
        break;
      }
    }
    return;
  }
  for (const gi_t gi : index_range<gi_t>(gi_t{0}, gi_t{CCTK_NumGroups()})) {
    if (!group_has_flux_tag(ext.record(gi)) || group_has_flux_register(gi))
      continue;
    CCTK_VWARN(CCTK_WARN_ALERT,
               "Group \"%s\" has a fluxes= tag but is not integrated by "
               "ODESolvers: no flux register is allocated for it, and it is "
               "not refluxed at coarse-fine boundaries (see "
               "CarpetX::do_reflux)",
               CCTK_FullGroupName(gi.index()));
  }
}

static std::array<CCTK_REAL, dim> face_areas(const amrex::Geometry &geom) {
  const CCTK_REAL *const dx = geom.CellSize();
  std::array<CCTK_REAL, dim> area;
  for (int d = 0; d < dim; ++d) {
    area[d] = 1;
    for (int j = 0; j < dim; ++j)
      if (j != d)
        area[d] *= dx[j];
  }
  return area;
}

void AccumulateFluxes(const level_t level, const stage_t stage,
                      const CCTK_REAL weight) {
  const GHExt &ext = driver();
  if (!ext.do_reflux)
    return;
  const stage_range stages = registered_stages();
  assert(stages.contains(stage));

  static Timer timer("AccumulateFluxes");
  Interval interval(timer);

  const auto &patchdata = ext.patch(level.patch());
  const auto &leveldata = patchdata.level(level);
  const std::optional<level_t> child = level.child();
  const auto *const childleveldata = child ? &patchdata.level(*child) : nullptr;
  const tl_t tl{0};
  const window_t this_level = window_t::one_level(level);
  const std::array<CCTK_REAL, dim> area =
      face_areas(patchdata.amrcore().Geom(level.index()));
  const flux_register_writer_t writing;

  for (const auto &groupdata : leveldata.groups()) {
    const std::optional<std::array<gi_t, dim>> &fluxes =
        groupdata.grid_function.fluxes;
    if (!fluxes)
      continue;
    const gi_t gi = groupdata.record.groupindex;

    const auto opened =
        [&](const GHExt::PatchData::LevelData &holder) -> flux_buffers_t * {
      const std::optional<sealed_flux_buffers_t> &flux =
          buffers_of(holder, gi).flux;
      return flux ? &writing.open(*flux) : nullptr;
    };
    flux_buffers_t *const child_flux =
        childleveldata ? opened(*childleveldata) : nullptr;
    flux_buffers_t *const own_flux = opened(leveldata);
    if (!child_flux && !own_flux)
      continue;

    for (int d = 0; d < dim; ++d) {
      std::ostringstream buf;
      buf << "AccumulateFluxes: flux of " << groupdata.record.groupname
          << " in direction " << d << " at RK stage " << stage;
      require(this_level, {.gi = fluxes->at(d),
                           .tl = tl,
                           .where = make_valid_int(),
                           .event = buf.str()});
    }

    if (child_flux) {
      amrex::FluxRegister &child_freg = child_flux->reg;
      amrex::FluxRegister &child_scratch = child_flux->scratch;
      if (stage == stages.first())
        child_freg.setVal(0);
      const int numvars = groupdata.record.numvars;
      const CCTK_REAL mult = -weight;
      for (int d = 0; d < dim; ++d) {
        const amrex::MultiFab &flux =
            leveldata.group(fluxes->at(d)).timelevel(tl).mfab();
        const CCTK_REAL area_d = area[d];
        for (const auto side :
             {amrex::Orientation::low, amrex::Orientation::high}) {
          const amrex::Orientation face(d, side);
          amrex::FabSet &scratch = child_scratch[face];
          amrex::FabSet &reg = child_freg[face];
          assert(scratch.boxArray() == reg.boxArray());
          assert(scratch.DistributionMap() == reg.DistributionMap());
          assert(scratch.nComp() == numvars && reg.nComp() == numvars);
          scratch.setVal(0);
          scratch.copyFrom(flux, 0, 0, 0, numvars);
          for (amrex::FabSetIter fsi(scratch); fsi.isValid(); ++fsi) {
            const amrex::Box &bx = fsi.validbox();
            const auto sfab = scratch.array(fsi);
            amrex::ParallelFor(
                bx, numvars,
                [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) noexcept {
                  sfab(i, j, k, n) = sfab(i, j, k, n) * mult * area_d;
                });
          }
          reg.plusFrom(scratch, 0, 0, numvars);
        }
      }
    }

    if (own_flux) {
      amrex::FluxRegister &own_freg = own_flux->reg;
      for (int d = 0; d < dim; ++d) {
        const amrex::MultiFab &flux =
            leveldata.group(fluxes->at(d)).timelevel(tl).mfab();
        assert(flux.ixType() ==
               amrex::IndexType(amrex::IntVect::TheDimensionVector(d)));
        assert(flux.DistributionMap() == own_freg.DistributionMap());
        own_freg.FineAdd(flux, d, 0, 0, groupdata.record.numvars,
                         weight * area[d]);
      }
    }
  }
}

namespace {
void correct_from_register(const amrex::FluxRegister &reg,
                           amrex::MultiFab &state,
                           const amrex::MultiFab &volume, const int numvars,
                           const amrex::Geometry &geom) {
  const_cast<amrex::FluxRegister &>(reg).Reflux(state, volume, 1.0, 0, 0,
                                                numvars, geom);
}
} // namespace

static void reflux_level(const level_t level) {
  DECLARE_CCTK_PARAMETERS;

  static Timer timer("Reflux");
  Interval interval(timer);

  const std::optional<level_t> fine = level.child();
  if (!fine)
    return;
  const auto &patchdata = driver().patch(level.patch());
  const auto &leveldata = patchdata.level(level);
  const auto &fineleveldata = patchdata.level(*fine);
  const amrex::Geometry &geom = patchdata.amrcore().Geom(level.index());
  const window_t coarse_level = window_t::one_level(level);
  const tl_t tl{0};

  for (const auto &groupdata : leveldata.groups()) {
    const gi_t gi = groupdata.record.groupindex;
    const std::optional<sealed_flux_buffers_t> &fineflux =
        buffers_of(fineleveldata, gi).flux;
    if (!fineflux)
      continue;

    if (verbose)
      CCTK_VINFO("Reflux: applying the flux register of levels (%d, %d) to "
                 "%s at iteration %d",
                 level.index(), fine->index(),
                 groupdata.record.groupname.c_str(), current_iteration());

    require(coarse_level,
            {.gi = gi,
             .tl = tl,
             .where = make_valid_int(),
             .event = "Reflux before refluxing: Coarse level data"});

    write_scope_t scope = open_write(coarse_level, {.gi = gi,
                                                    .tl = tl,
                                                    .fills = make_valid_int(),
                                                    .keeps = make_valid_all(),
                                                    .event = "Reflux"});
    amrex::MultiFab &state = scope.mfab(level);
    const std::optional<amrex::MultiFab> &cell_volume =
        buffers_of(leveldata).cell_volume;
    assert(cell_volume);
    const amrex::MultiFab &volume = *cell_volume;
    assert(volume.ixType() == state.ixType());
    assert(volume.DistributionMap() == state.DistributionMap());
    correct_from_register(fineflux->get().reg, state, volume,
                          groupdata.record.numvars, geom);
  }
}

void Reflux(const window_t window) {
  if (!driver().do_reflux)
    return;
  loop_pairs_fine_to_coarse(window, reflux_level);
}

} // namespace CarpetX
