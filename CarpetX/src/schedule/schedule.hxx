#ifndef CARPETX_CARPETX_SCHEDULE_SCHEDULE_HXX
#define CARPETX_CARPETX_SCHEDULE_SCHEDULE_HXX

#include "internal.hxx"

#include "writers/writers.hxx"

#include <cctk.h>
#include <cctk_Schedule.h>

namespace CarpetX {

int Initialise(tFleshConfig *config, clock_writer_t &clock,
               structure_writer_t &structure, registry_writer_t &registry);
int Evolve(tFleshConfig *config, clock_writer_t &clock,
           structure_writer_t &structure);
int Shutdown(tFleshConfig *config);

int CallFunction(void *function, cFunctionData *attribute, void *data);

int GroupStorageIncrease(const cGH *cctkGH, int n_groups, const int *groups,
                         const int *tls, int *status,
                         registry_writer_t &registry);
int GroupStorageDecrease(const cGH *cctkGH, int n_groups, const int *groups,
                         const int *tls, int *status,
                         registry_writer_t &registry);
int EnableGroupStorage(const cGH *cctkGH, const char *groupname,
                       registry_writer_t &registry);
int DisableGroupStorage(const cGH *cctkGH, const char *groupname,
                        registry_writer_t &registry);
int QueryGroupStorageB(const cGH *cctkGH, int gi, const char *groupname);

} // namespace CarpetX

#endif
