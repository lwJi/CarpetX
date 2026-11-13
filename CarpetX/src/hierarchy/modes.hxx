#ifndef CARPETX_CARPETX_HIERARCHY_MODES_HXX
#define CARPETX_CARPETX_HIERARCHY_MODES_HXX

#include "internal.hxx"

#include "hierarchy/driver.hxx"

#include <cctk.h>

#include <vector>

namespace CarpetX {

GHExt::cctkGHptr copy_cctkGH(const cGH *restrict const sourceGH);

enum class mode_t { unknown, local, patch, level, global, meta };

bool in_local_mode(const cGH *restrict cctkGH);
bool in_patch_mode(const cGH *restrict cctkGH);
bool in_level_mode(const cGH *restrict cctkGH);
bool in_global_mode(const cGH *restrict cctkGH);
bool in_meta_mode(const cGH *restrict cctkGH);

void setup_cctkGH(cGH *restrict cctkGH);
void update_cctkGH(cGH *restrict cctkGH, const cGH *restrict sourceGH);
void enter_global_mode(cGH *restrict cctkGH);
void leave_global_mode(cGH *restrict cctkGH);

GHExt::cctkGHptr make_level_cctkGH(cGH *global_cctkGH, level_t level);

struct level_cctkGHs_t {
  GHExt::cctkGHptr patch;
  std::vector<GHExt::cctkGHptr> local;
};
level_cctkGHs_t make_level_cctkGHs(cGH *level_cctkGH,
                                   GHExt::PatchData::LevelData &leveldata);

} // namespace CarpetX

#endif
