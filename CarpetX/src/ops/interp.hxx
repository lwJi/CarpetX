#ifndef CARPETX_CARPETX_OPS_INTERP_HXX
#define CARPETX_CARPETX_OPS_INTERP_HXX

#include "internal.hxx"

#include <cctk.h>

#include <defs.hxx>

#include <AMReX_AmrParticles.H>
#include <AMReX_Particles.H>

#include <vector>
#include <vect.hxx>

namespace CarpetX {

class InterpolationSetup {
private:
  using Container = amrex::ParticleContainer<3, 2>;
  using Particle = Container::ParticleType;
  using PinnedParticleTile = amrex::ParticleContainer_impl<
      Particle, 0, 0, amrex::PinnedArenaAllocator>::ParticleTileType;
  using ParticleTile = Container::ParticleTileType;

  const int npoints{0};
  std::vector<bool> symmetry_reflected_z;
  std::vector<Container> containers{}; // [patch]

public:

  InterpolationSetup(CCTK_ATTRIBUTE_UNUSED const cGH *restrict const cctkGH,
                     const CCTK_INT npoints,
                     const CCTK_REAL *restrict const globalsx,
                     const CCTK_REAL *restrict const globalsy,
                     const CCTK_REAL *restrict const globalsz);

  void Interpolate(CCTK_ATTRIBUTE_UNUSED const cGH *restrict const cctkGH,
                   const CCTK_INT nvars, const CCTK_INT *restrict const varinds,
                   const CCTK_INT *restrict const operations,
                   const std::vector<Arith::vect<Arith::vect<bool, 3>, 2> >
                       &allowed_boundaries, //  [patch][face][direction]
                   const CCTK_POINTER resultptrs_) const;
};

// a dummy routine for now
// TODO: implement this for actual local interpolation
int InterpLocalUniform(int N_dims, int param_table_handle,
                       /***** coordinate system *****/
                       const CCTK_REAL coord_origin[],
                       const CCTK_REAL coord_delta[],
                       /***** interpolation points *****/
                       int N_interp_points, int interp_coords_type_code,
                       const void *const interp_coords[],
                       /***** input arrays *****/
                       int N_input_arrays, const CCTK_INT input_array_dims[],
                       const CCTK_INT input_array_type_codes[],
                       const void *const input_arrays[],
                       /***** output arrays *****/
                       int N_output_arrays,
                       const CCTK_INT output_array_type_codes[],
                       void *const output_arrays[]);
} // namespace CarpetX

#endif
