#ifndef CARPETX_CARPETX_HIERARCHY_MODES_HXX
#define CARPETX_CARPETX_HIERARCHY_MODES_HXX

#include "internal.hxx"

#include "hierarchy/grid.hxx"
#include "hierarchy/model.hxx"

#include <carpetx.hxx>

#include <cctk.h>

namespace CarpetX {

enum class mode_t { unknown, local, patch, level, global, meta };

bool in_local_mode(const cGH *restrict cctkGH);
bool in_patch_mode(const cGH *restrict cctkGH);
bool in_level_mode(const cGH *restrict cctkGH);
bool in_global_mode(const cGH *restrict cctkGH);
bool in_meta_mode(const cGH *restrict cctkGH);

void setup_cctkGH(cGH *restrict cctkGH);
void update_cctkGH(cGH *restrict cctkGH, const cGH *restrict sourceGH);
void update_cctkGHs(cGH *restrict cctkGH);
void enter_global_mode(cGH *restrict cctkGH);
void leave_global_mode(cGH *restrict cctkGH);
void enter_level_mode(cGH *restrict cctkGH, level_t level);
void leave_level_mode(cGH *restrict cctkGH, level_t level);
void enter_patch_mode(cGH *restrict cctkGH,
                      const GHExt::PatchData::LevelData &restrict leveldata);
void leave_patch_mode(cGH *restrict cctkGH,
                      const GHExt::PatchData::LevelData &restrict leveldata);
void enter_local_mode(cGH *restrict cctkGH,
                      const GHExt::PatchData::LevelData &restrict leveldata,
                      const MFPointer &mfp);
void leave_local_mode(cGH *restrict cctkGH,
                      const GHExt::PatchData::LevelData &restrict leveldata,
                      const MFPointer &mfp);

} // namespace CarpetX

#endif
