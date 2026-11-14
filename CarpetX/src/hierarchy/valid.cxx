#include "hierarchy/valid.hxx"

#include <carpetx.hxx>

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

} // namespace CarpetX
