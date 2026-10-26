#ifndef CARPETX_CARPETX_IO_SILO_NAMES_HXX
#define CARPETX_CARPETX_IO_SILO_NAMES_HXX

#include "internal.hxx"

#include <carpetx.hxx>

#include <cctk.h>

#ifdef HAVE_CAPABILITY_Silo

#include <silo.hxx>

#include <array>
#include <cassert>
#include <cctype>
#include <iomanip>
#include <optional>
#include <regex>
#include <sstream>
#include <string>
#include <type_traits>

namespace CarpetX {

template <typename T> struct db_datatype;
template <> struct db_datatype<char> : std::integral_constant<int, DB_CHAR> {};
template <>
struct db_datatype<short> : std::integral_constant<int, DB_SHORT> {};
template <> struct db_datatype<int> : std::integral_constant<int, DB_INT> {};
template <> struct db_datatype<long> : std::integral_constant<int, DB_LONG> {};
template <>
struct db_datatype<long long> : std::integral_constant<int, DB_LONG_LONG> {};
template <>
struct db_datatype<float> : std::integral_constant<int, DB_FLOAT> {};
template <>
struct db_datatype<double> : std::integral_constant<int, DB_DOUBLE> {};
template <typename T> constexpr int db_datatype_v = db_datatype<T>::value;

inline const std::string driver_name = "CarpetX";

inline std::string make_subdirname(const std::string &file_name,
                                   const int iteration) {
  std::ostringstream buf;
  buf << file_name << ".it" << std::setw(8) << std::setfill('0') << iteration
      << ".silo.dir";
  return buf.str();
}

inline std::string make_filename(const std::string &file_name,
                                 const int iteration, const int ioserver = -1) {
  std::ostringstream buf;
  buf << file_name << ".it" << std::setw(8) << std::setfill('0') << iteration;
  if (ioserver >= 0)
    buf << ".p" << std::setw(6) << std::setfill('0') << ioserver;
  buf << ".silo";
  return buf.str();
}

inline int match_filename(const std::string &file_name) {
  const std::regex filename_regex("[.]it(\\d+)[.]silo$",
                                  std::regex_constants::ECMAScript);
  std::smatch sm;
  std::regex_search(file_name, sm, filename_regex);
  if (sm.empty())
    return -1;
  return std::stoi(sm.str(1));
}

inline std::string silo_spelling(std::string name) {
  name = std::regex_replace(name, std::regex("::"), "-");
  for (auto &ch : name)
    ch = std::tolower(ch);
  return name;
}

inline void put_block_suffix(std::ostringstream &buf, const level_t level,
                             const int component) {
  buf << ".m" << std::setw(4) << std::setfill('0') << level.patch().index()
      << ".rl" << std::setw(2) << std::setfill('0') << level.index() << ".c"
      << std::setw(8) << std::setfill('0') << component;
}

inline std::string
make_meshname(const std::array<int, Loop::dim> &nghosts,
              const std::optional<level_t> level = std::nullopt,
              const int component = -1,
              const std::string &band = std::string()) {
  assert(bool(level) == (component >= 0));
  std::ostringstream buf;
  buf << (level ? "box" : "gh") << ".ghosts";
  for (int d = 0; d < Loop::dim; ++d) {
    if (d > 0)
      buf << "_";
    buf << std::setw(2) << std::setfill('0') << nghosts[d];
  }
  if (level)
    put_block_suffix(buf, *level, component);
  if (!band.empty())
    buf << ".band." << band;
  return DB::legalize_name(buf.str());
}

inline std::string make_varname(
    const std::string &var, const std::optional<level_t> level = std::nullopt,
    const int component = -1, const std::string &band = std::string()) {
  assert(bool(level) == (component >= 0));
  std::ostringstream buf;
  buf << silo_spelling(var);
  if (level)
    put_block_suffix(buf, *level, component);
  if (!band.empty())
    buf << ".band." << band;
  return DB::legalize_name(buf.str());
}

inline std::string make_fabarraybasename(const level_t level) {
  std::ostringstream buf;
  buf << "FabArrayBase.m" << std::setw(4) << std::setfill('0')
      << level.patch().index() << ".rl" << std::setw(2) << std::setfill('0')
      << level.index();
  return DB::legalize_name(buf.str());
}

} // namespace CarpetX

#endif

#endif
