#ifndef CARPETX_CARPETX_SCHEDULE_SCHEDULE_HXX
#define CARPETX_CARPETX_SCHEDULE_SCHEDULE_HXX

#include "internal.hxx"

#include <cctk.h>
#include <cctk_Schedule.h>

namespace CarpetX {

int Initialise(tFleshConfig *config);
int Evolve(tFleshConfig *config);
int Shutdown(tFleshConfig *config);

int CallFunction(void *function, cFunctionData *attribute, void *data);

int GroupStorageIncrease(const cGH *cctkGH, int n_groups, const int *groups,
                         const int *tls, int *status);
int GroupStorageDecrease(const cGH *cctkGH, int n_groups, const int *groups,
                         const int *tls, int *status);
int EnableGroupStorage(const cGH *cctkGH, const char *groupname);
int DisableGroupStorage(const cGH *cctkGH, const char *groupname);
int QueryGroupStorageB(const cGH *cctkGH, int gi, const char *groupname);

} // namespace CarpetX

#endif
