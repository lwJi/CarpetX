#ifndef CARPETX_CARPETX_IO_OPENPMD_NAMES_HXX
#define CARPETX_CARPETX_IO_OPENPMD_NAMES_HXX

#include "internal.hxx"

#include <carpetx.hxx>

#include <cctk.h>
#include <cctk_Parameters.h>

#ifdef HAVE_CAPABILITY_openPMD_api

#include <openPMD/openPMD.hpp>

#ifdef HAVE_CAPABILITY_ADIOS2
#include <adios2.h>
#endif

#include <cctype>
#include <iomanip>
#include <optional>
#include <regex>
#include <sstream>
#include <string>

namespace CarpetX {

inline openPMD::Format get_format() {
  DECLARE_CCTK_PARAMETERS;
  if (CCTK_EQUALS(openpmd_format, "HDF5"))
    return openPMD::Format::HDF5;
#if OPENPMDAPI_VERSION_GE(0, 16, 0)
#else
  if (CCTK_EQUALS(openpmd_format, "ADIOS1"))
    return openPMD::Format::ADIOS1;
#endif
  if (CCTK_EQUALS(openpmd_format, "ADIOS2_auto"))
#if OPENPMDAPI_VERSION_GE(0, 15, 0)
    return openPMD::Format::ADIOS2_BP5;
#else
    return openPMD::Format::ADIOS2;
#endif
#if OPENPMDAPI_VERSION_GE(0, 15, 0)
  if (CCTK_EQUALS(openpmd_format, "ADIOS2_BP"))
    return openPMD::Format::ADIOS2_BP;
  if (CCTK_EQUALS(openpmd_format, "ADIOS2_BP4"))
    return openPMD::Format::ADIOS2_BP4;
  if (CCTK_EQUALS(openpmd_format, "ADIOS2_BP5"))
    return openPMD::Format::ADIOS2_BP5;
#else
  if (CCTK_EQUALS(openpmd_format, "ADIOS2"))
    return openPMD::Format::ADIOS2;
#endif
  if (CCTK_EQUALS(openpmd_format, "ADIOS2_SST"))
    return openPMD::Format::ADIOS2_SST;
  if (CCTK_EQUALS(openpmd_format, "ADIOS2_SSC"))
    return openPMD::Format::ADIOS2_SSC;
  if (CCTK_EQUALS(openpmd_format, "JSON"))
    return openPMD::Format::JSON;
#if OPENPMDAPI_VERSION_GE(0, 16, 0)
  if (CCTK_EQUALS(openpmd_format, "TOML"))
    return openPMD::Format::TOML;
#endif
#if OPENPMDAPI_VERSION_GE(0, 16, 0)
  if (CCTK_EQUALS(openpmd_format, "GENERIC"))
    return openPMD::Format::GENERIC;
#endif
  CCTK_VERROR("The openPMD format \"%s\" is not supported in version %d.%d.%d "
              "of the openPMD_api library",
              openpmd_format, OPENPMDAPI_VERSION_MAJOR,
              OPENPMDAPI_VERSION_MINOR, OPENPMDAPI_VERSION_PATCH);
}

constexpr openPMD::IterationEncoding iterationEncoding =
    openPMD::IterationEncoding::fileBased;

#ifdef ADIOS2_HAVE_BLOSC2
inline const std::string options = R"EOS(
  {
    "adios2": {
      "dataset": {
        "operators": [
          {
            "type": "blosc",
            "parameters": {
              "clevel": "9",
              "doshuffle": "BLOSC_SHUFFLE"
            }
          }
        ]
      }
    }
  }
)EOS";
#else
inline const std::string options = R"EOS(
  {
    "adios2": {
      "dataset": {
        "operators": [
        ]
      }
    }
  }
)EOS";
#endif

struct Const {
  static constexpr CCTK_REAL c = 299792458;
  static constexpr CCTK_REAL G = 6.67430e-11;
  static constexpr CCTK_REAL M_solar = 1.98847e+30;
};

struct Unit {
  static constexpr CCTK_REAL velocity = Const::c;
  static constexpr CCTK_REAL mass = Const::M_solar;
  static constexpr CCTK_REAL length = Const::G * mass / Arith::pow2(Const::c);
  static constexpr CCTK_REAL time = length / velocity;
};

inline std::string openpmd_spelling(std::string name) {
  name = std::regex_replace(name, std::regex("::"), "_");
  for (auto &ch : name)
    ch = std::tolower(ch);
  return name;
}

inline std::string make_meshname(const std::string &group,
                                 const std::optional<level_t> level,
                                 const tl_t tl = tl_t{0},
                                 const std::string &band = std::string()) {
  std::ostringstream buf;
  buf << openpmd_spelling(group);
  if (level) {
    buf << "_patch" << std::setw(2) << std::setfill('0')
        << level->patch().index();
    buf << "_lev" << std::setw(2) << std::setfill('0') << level->index();
  }
  if (tl > tl_t{0})
    buf << "_tl" << std::setw(2) << std::setfill('0') << tl.index();
  if (!band.empty())
    buf << "_band_" << band;
  return buf.str();
}

inline std::string make_componentname(const std::string &var) {
  return openpmd_spelling(var);
}

} // namespace CarpetX

#endif

#endif
