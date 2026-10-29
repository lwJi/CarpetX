#ifndef CARPETX_CARPETX_IO_IO_HXX
#define CARPETX_CARPETX_IO_IO_HXX

#include "internal.hxx"

#include "io/checkpoint.hxx"

#include <cctk.h>

#include <string>

namespace CarpetX {

std::string get_simulation_name();

checkpoint_record_t RecoverGridStructure(cGH *cctkGH);
void RecoverGH(const cGH *cctkGH, const checkpoint_record_t &record);
void InputGH(const cGH *cctkGH);

int OutputGH(const cGH *cctkGH);

void Checkpoint(const cGH *cctkGH);
bool CheckWalltimeTrigger(int runtime);
std::string DescribeTriggers(const char *name1, bool fired1, const char *name2,
                             bool fired2);

} // namespace CarpetX

#endif
