#ifndef CARPETX_CARPETX_IO_HXX
#define CARPETX_CARPETX_IO_HXX

#include <cctk.h>

#include <cstdint>

namespace CarpetX {

void RecoverGridStructure(cGH *cctkGH);
void RecoverGH(const cGH *cctkGH);
void InputGH(const cGH *cctkGH);

int OutputGH(const cGH *cctkGH);

bool HasRecoveredLevelIterations();
void SetRecoveredLevelIterations(bool value);

void StoreRecoveredLevelIteration(int patch, int level, int64_t num,
                                  int64_t den);
void ApplyRecoveredLevelIterations();

} // namespace CarpetX

#endif // #ifndef CARPETX_CARPETX_IO_HXX
