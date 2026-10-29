#include "io/io_meta.hxx"

#include "hierarchy/driver.hxx"
#include "util/refuse.hxx"

#include <cctk.h>
#include <cctk_Parameters.h>

#include <cassert>
#include <fstream>
#include <iomanip>
#include <utility>

namespace CarpetX {

const std::array<int, 3> cactus_metadata_version{1, 0, 0};

YAML::Emitter &operator<<(YAML::Emitter &yaml, const group_type_t &group_type) {
  yaml << YAML::LocalTag("group_type-1.0.0");
  switch (group_type) {
  case group_type_t::none:
    yaml << "none";
    break;
  case group_type_t::gf:
    yaml << "gf";
    break;
  case group_type_t::array:
    yaml << "array";
    break;
  default:
    assert(0);
  }
  return yaml;
}

YAML::Emitter &operator<<(YAML::Emitter &yaml, const reduction_t &reduction) {
  yaml << YAML::LocalTag("reduction-1.0.0");
  switch (reduction) {
  case reduction_t::volume:
    yaml << "volume";
    break;
  case reduction_t::sum:
    yaml << "sum";
    break;
  case reduction_t::sum_abs:
    yaml << "sum_abs";
    break;
  case reduction_t::sum_squared:
    yaml << "sum_squared";
    break;
  case reduction_t::average:
    yaml << "average";
    break;
  case reduction_t::standard_deviation:
    yaml << "standard_deviation";
    break;
  case reduction_t::minimum:
    yaml << "minimum";
    break;
  case reduction_t::maximum:
    yaml << "maximum";
    break;
  case reduction_t::maximum_abs:
    yaml << "maximum_abs";
    break;
  case reduction_t::norm1:
    yaml << "norm1";
    break;
  case reduction_t::norm2:
    yaml << "norm2";
    break;
  case reduction_t::norm_inf:
    yaml << "norm_inf";
    break;
  case reduction_t::minimum_location:
    yaml << "minimum_location";
    break;
  case reduction_t::maximum_location:
    yaml << "maximum_location";
    break;
  case reduction_t::sum_location:
    yaml << "sum_location";
    break;
  default:
    assert(0);
  }
  return yaml;
}

struct real_output_file_description_t {
  std::string filename;                //
  std::string description;             // human readable
  std::string writer_thorn;            //
  std::vector<std::string> variables;  //
  std::vector<int> iterations;         //
  group_type_t group_type;             // set automatically
  int variable_dimensions;             // set automatically
  std::vector<int> output_directions;  //
  std::vector<reduction_t> reductions; //
  std::string format_name;             //
  std::array<int, 3> format_version;   //

  real_output_file_description_t() = delete;

  real_output_file_description_t(const real_output_file_description_t &) =
      default;
  real_output_file_description_t(real_output_file_description_t &&) = default;
  real_output_file_description_t &
  operator=(const real_output_file_description_t &) = default;
  real_output_file_description_t &
  operator=(real_output_file_description_t &&) = default;

  real_output_file_description_t(
      output_file_description_t output_file_description,
      std::source_location site);
};

real_output_file_description_t::real_output_file_description_t(
    output_file_description_t ofd, const std::source_location site)
    : filename(std::move(ofd.filename)),
      description(std::move(ofd.description)),
      writer_thorn(std::move(ofd.writer_thorn)),
      variables(std::move(ofd.variables)),
      iterations(std::move(ofd.iterations)), group_type(group_type_t::none),
      variable_dimensions(-1),
      output_directions(std::move(ofd.output_directions)),
      reductions(std::move(ofd.reductions)),
      format_name(std::move(ofd.format_name)),
      format_version(ofd.format_version) {
  if (filename.empty())
    refuse(site, "an output file description has an empty filename");

  if (CCTK_IsThornActive(writer_thorn.c_str()) <= 0)
    refuse(site, "output file %s: writer thorn \"%s\" is not active",
           filename.c_str(), writer_thorn.c_str());
  for (auto &ch : writer_thorn)
    ch = std::tolower(ch);

  for (const int iteration : iterations)
    if (iteration < 0)
      refuse(site, "output file %s: iteration %d is negative", filename.c_str(),
             iteration);

  if (!variables.empty()) {
    const GHExt &ext = driver();
    const auto record_named =
        [&](const std::string &var) -> const GHExt::GroupRecord & {
      const int varindex = CCTK_VarIndex(var.c_str());
      if (varindex < 0)
        refuse(site, "output file %s: variable \"%s\" does not exist",
               filename.c_str(), var.c_str());
      return ext.record_of(var_t{varindex}, site);
    };
    const auto group_type_of = [](const GHExt::GroupRecord &record) {
      return record.kind == group_kind::grid_function ? group_type_t::gf
                                                      : group_type_t::array;
    };
    const auto dimension_of = [](const GHExt::GroupRecord &record) {
      return record.kind == group_kind::grid_function
                 ? dim
                 : record.array().dimension;
    };

    const GHExt::GroupRecord &first = record_named(variables.front());
    group_type = group_type_of(first);
    variable_dimensions = dimension_of(first);
    for (const std::string &var : variables) {
      const GHExt::GroupRecord &record = record_named(var);
      if (group_type_of(record) != group_type)
        refuse(site,
               "output file %s: of the variables \"%s\" and \"%s\", only one "
               "is a grid function",
               filename.c_str(), variables.front().c_str(), var.c_str());
      if (dimension_of(record) != variable_dimensions)
        refuse(site,
               "output file %s: variable \"%s\" has %d dimensions, but "
               "variable \"%s\" has %d",
               filename.c_str(), var.c_str(), dimension_of(record),
               variables.front().c_str(), variable_dimensions);
    }
    for (std::string &var : variables)
      for (auto &ch : var)
        ch = std::tolower(ch);

    for (const int dir : output_directions)
      if (dir < 0 || dir >= variable_dimensions)
        refuse(site,
               "output file %s: output direction %d is out of range for "
               "%d-dimensional variables",
               filename.c_str(), dir, variable_dimensions);
  }

  if (!output_directions.empty() && !reductions.empty())
    refuse(site, "output file %s has both output directions and reductions",
           filename.c_str());

  if (format_name.empty())
    refuse(site, "output file %s has an empty format name", filename.c_str());

  for (const int version : format_version)
    if (version < 0)
      refuse(site,
             "output file %s has format version %d.%d.%d, which is negative",
             filename.c_str(), format_version[0], format_version[1],
             format_version[2]);
}

YAML::Emitter &operator<<(YAML::Emitter &yaml,
                          const real_output_file_description_t &rofd) {
  yaml << YAML::LocalTag("cactus-metadata-1.0.0");
  yaml << YAML::BeginMap;
  yaml << YAML::Key << "filename" << YAML::Value << rofd.filename;
  yaml << YAML::Key << "description" << YAML::Value << rofd.description;
  yaml << YAML::Key << "writer_thorn" << YAML::Value << rofd.writer_thorn;
  yaml << YAML::Key << "variables" << YAML::Value << rofd.variables;
  if (!rofd.variables.empty()) {
    yaml << YAML::Key << "iterations" << YAML::Value << YAML::Flow
         << rofd.iterations;
    yaml << YAML::Key << "group_type" << YAML::Value << rofd.group_type;
    yaml << YAML::Key << "variable_dimensions" << YAML::Value
         << rofd.variable_dimensions;
    yaml << YAML::Key << "real_output_directions" << YAML::Value << YAML::Flow
         << rofd.output_directions;
    yaml << YAML::Key << "reductions" << YAML::Value << YAML::Flow
         << rofd.reductions;
  } // if !rofd.variables.empty
  yaml << YAML::Key << "format_name" << YAML::Value << rofd.format_name;
  yaml << YAML::Key << "format_version" << YAML::Value << YAML::Flow
       << YAML::BeginSeq;
  for (const auto &elt : rofd.format_version)
    yaml << elt;
  yaml << YAML::EndSeq;
  yaml << YAML::EndMap;
  return yaml;
}

////////////////////////////////////////////////////////////////////////////////

std::vector<real_output_file_description_t> real_output_file_descriptions;

void OutputMeta_RegisterOutputFile(
    output_file_description_t output_file_description,
    const std::source_location site) {
  // Only the root process outputs metadata
  if (CCTK_MyProc(nullptr) != 0)
    return;

  real_output_file_descriptions.emplace_back(std::move(output_file_description),
                                             site);
}

void OutputMeta(const cGH *restrict const cctkGH) {
  DECLARE_CCTK_PARAMETERS;

  // Only the root process outputs metadata
  if (CCTK_MyProc(nullptr) != 0)
    return;

  // Skip metadata output if no files were written
  if (real_output_file_descriptions.empty())
    return;

  if (out_metadata) {
    YAML::Emitter yaml;
    yaml << YAML::Comment("Cactus Metadata");
    yaml << YAML::BeginDoc;
    yaml << real_output_file_descriptions;
    yaml << YAML::EndDoc;

    std::ostringstream buf;
    buf << out_dir << "/cactus-metadata"
        << ".it" << std::setw(8) << std::setfill('0') << cctkGH->cctk_iteration
        << ".yaml";
    const std::string filename = buf.str();
    std::ofstream file(filename);
    file << yaml.c_str() << "\n";
    file.close();
  }

  real_output_file_descriptions.clear();
}

} // namespace CarpetX
