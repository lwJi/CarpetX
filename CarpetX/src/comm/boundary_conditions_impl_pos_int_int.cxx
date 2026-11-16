#include "comm/boundary_conditions_impl.hxx"

namespace CarpetX {

template void BoundaryCondition::apply_on_face<POS, INT, INT>() const;

} // namespace CarpetX
