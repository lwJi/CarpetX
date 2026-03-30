#ifndef CARPETX_CARPETX_IO_HXX
#define CARPETX_CARPETX_IO_HXX

#include <cctk.h>

namespace CarpetX {

void RecoverGridStructure(cGH *cctkGH);
void RecoverGH(const cGH *cctkGH);
void InputGH(const cGH *cctkGH);

int OutputGH(const cGH *cctkGH);

bool HasRecoveredLevelIterations();
void SetRecoveredLevelIterations(bool value);

} // namespace CarpetX

#endif // #ifndef CARPETX_CARPETX_IO_HXX
