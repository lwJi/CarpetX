#ifndef CARPETX_CARPETX_HIERARCHY_WHY_VALID_HXX
#define CARPETX_CARPETX_HIERARCHY_WHY_VALID_HXX

#include "internal.hxx"

#include <carpetx.hxx>

#include <yaml-cpp/yaml.h>

#include <functional>
#include <ostream>
#include <string>

namespace CarpetX {

class write_scope_t;
class array_write_scope_t;

YAML::Emitter &operator<<(YAML::Emitter &yaml, valid_t v);

class why_valid_t {
  valid_t valid;
  std::function<std::string()> why_int, why_outer, why_same_level,
      why_coarse_fine;

  void set(const valid_t &which, const valid_t &val,
           const std::function<std::string()> &why) {
    valid = (valid & ~which) | (val & which);
    if (which.valid_int)
      why_int = why;
    if (which.valid_outer)
      why_outer = why;
    if (which.valid_same_level)
      why_same_level = why;
    if (which.valid_coarse_fine)
      why_coarse_fine = why;
  }
  void set_invalid(const valid_t &which,
                   const std::function<std::string()> &why) {
    set(which, valid_t(false), why);
  }
  void set_valid(const valid_t &which,
                 const std::function<std::string()> &why) {
    set(which, valid_t(true), why);
  }
  friend class write_scope_t;
  friend class array_write_scope_t;

public:
  why_valid_t() = delete;
  why_valid_t(const std::function<std::string()> &why)
      : why_valid_t(false, why) {}
  why_valid_t(bool b, const std::function<std::string()> &why)
      : why_valid_t(valid_t(b), why) {}
  why_valid_t(const valid_t &val, const std::function<std::string()> &why)
      : valid(val), why_int(why), why_outer(why), why_same_level(why),
        why_coarse_fine(why) {}

  const valid_t &get() const { return valid; }

  std::string explanation() const;

  friend std::ostream &operator<<(std::ostream &os, const why_valid_t &why);
  operator std::string() const;
  friend YAML::Emitter &operator<<(YAML::Emitter &yaml, const why_valid_t &why);
};

} // namespace CarpetX

#endif
