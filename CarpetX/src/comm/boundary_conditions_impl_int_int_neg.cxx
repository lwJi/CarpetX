#include "comm/boundary_conditions_impl.hxx"

namespace CarpetX {

template void BoundaryCondition::apply_on_face<INT, INT, NEG>() const;

} // namespace CarpetX
