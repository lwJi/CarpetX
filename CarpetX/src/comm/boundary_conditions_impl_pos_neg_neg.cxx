#include "comm/boundary_conditions_impl.hxx"

namespace CarpetX {

template void BoundaryCondition::apply_on_face<POS, NEG, NEG>() const;

} // namespace CarpetX
