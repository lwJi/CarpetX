#include "hierarchy/why_valid.hxx"

#include <carpetx.hxx>

#include <yaml-cpp/yaml.h>

#include <ostream>
#include <sstream>
#include <string>

namespace CarpetX {

std::string valid_t::explanation() const {
  const auto valstr = [](bool v) { return v ? "valid" : "invalid"; };
  std::ostringstream buf;
  buf << "\n"
      << "  The interior is " << valstr(valid_int) << ".\n"
      << "  The outer boundary is " << valstr(valid_outer) << ".\n"
      << "  The same-level ghosts are " << valstr(valid_same_level) << ".\n"
      << "  The coarse-fine ghosts are " << valstr(valid_coarse_fine) << ".\n";
  return buf.str();
}

std::ostream &operator<<(std::ostream &os, const valid_t v) {
  auto str = [](bool v) { return v ? "VAL" : "INV"; };
  return os << "[int:" << str(v.valid_int) << ",outer:" << str(v.valid_outer)
            << ",same_level:" << str(v.valid_same_level)
            << ",coarse_fine:" << str(v.valid_coarse_fine) << "]";
}

valid_t::operator std::string() const {
  std::ostringstream buf;
  buf << *this;
  return buf.str();
}

YAML::Emitter &operator<<(YAML::Emitter &yaml, const valid_t v) {
  yaml << YAML::LocalTag("valid-2.0.0");
  yaml << YAML::Flow << YAML::BeginMap;
  yaml << YAML::Key << "int" << YAML::Value << v.valid_int;
  yaml << YAML::Key << "outer" << YAML::Value << v.valid_outer;
  yaml << YAML::Key << "same_level" << YAML::Value << v.valid_same_level;
  yaml << YAML::Key << "coarse_fine" << YAML::Value << v.valid_coarse_fine;
  yaml << YAML::EndMap;
  return yaml;
}

std::string why_valid_t::explanation() const {
  const auto valstr = [](bool v) { return v ? "valid" : "invalid"; };
  std::ostringstream buf;
  buf << "\n"
      << "  The interior is " << valstr(valid.valid_int)
      << " because: " << why_int() << ".\n"
      << "  The outer boundary is " << valstr(valid.valid_outer)
      << " because: " << why_outer() << ".\n"
      << "  The same-level ghosts are " << valstr(valid.valid_same_level)
      << " because: " << why_same_level() << ".\n"
      << "  The coarse-fine ghosts are " << valstr(valid.valid_coarse_fine)
      << " because: " << why_coarse_fine() << ".\n";
  return buf.str();
}

std::ostream &operator<<(std::ostream &os, const why_valid_t &why) {
  return os << why.valid << ","
            << "why{int:" << why.why_int() << ","
            << "outer:" << why.why_outer() << ","
            << "same_level:" << why.why_same_level() << ","
            << "coarse_fine:" << why.why_coarse_fine() << "}";
}

why_valid_t::operator std::string() const {
  std::ostringstream buf;
  buf << *this;
  return buf.str();
}

YAML::Emitter &operator<<(YAML::Emitter &yaml, const why_valid_t &why) {
  yaml << YAML::LocalTag("why_valid-2.0.0");
  yaml << YAML::BeginMap;
  yaml << YAML::Key << "valid" << YAML::Value << why.valid;
  yaml << YAML::Key << "why" << YAML::Value << YAML::BeginMap;
  yaml << YAML::Key << "int" << YAML::Value << why.why_int();
  yaml << YAML::Key << "outer" << YAML::Value << why.why_outer();
  yaml << YAML::Key << "same_level" << YAML::Value << why.why_same_level();
  yaml << YAML::Key << "coarse_fine" << YAML::Value << why.why_coarse_fine();
  yaml << YAML::EndMap;
  yaml << YAML::EndMap;
  return yaml;
}

} // namespace CarpetX
