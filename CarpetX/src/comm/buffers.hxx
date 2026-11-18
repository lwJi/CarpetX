#ifndef CARPETX_CARPETX_COMM_BUFFERS_HXX
#define CARPETX_CARPETX_COMM_BUFFERS_HXX

#include "internal.hxx"

#include "hierarchy/model.hxx"
#include <carpetx.hxx>

#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>

namespace CarpetX {

level_buffers_ptr make_level_buffers(level_t level, const amrex::BoxArray &ba,
                                     const amrex::DistributionMapping &dm);

} // namespace CarpetX

#endif
