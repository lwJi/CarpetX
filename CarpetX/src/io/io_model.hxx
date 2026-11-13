#ifndef CARPETX_CARPETX_IO_IO_MODEL_HXX
#define CARPETX_CARPETX_IO_IO_MODEL_HXX

#include "internal.hxx"

#include <yaml-cpp/yaml.h>

namespace CarpetX {

struct GHExt;

YAML::Emitter &operator<<(YAML::Emitter &yaml, const GHExt &ext);

} // namespace CarpetX

#endif
