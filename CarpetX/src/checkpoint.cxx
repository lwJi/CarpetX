#include "checkpoint.hxx"

#include "driver.hxx"
#include "valid.hxx"

#include <CactusBase/IOUtil/src/ioutil_CheckpointRecovery.h>
#include <cctk.h>
#include <cctk_Parameters.h>

#include <AMReX_BoxArray.H>
#include <AMReX_BoxList.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Geometry.H>
#include <AMReX_Vector.H>

#include <cassert>
#include <cstdlib>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <variant>
#include <vector>

namespace CarpetX {

std::string buffer_id_t::name() const {
  std::ostringstream buf;
  buf << group;
  if (level)
    buf << ".p" << level->patch().index() << ".l" << level->index();
  if (const std::string *const tag = std::get_if<std::string>(&which))
    buf << ".band." << *tag;
  else if (level)
    buf << ".tl" << std::get<tl_t>(which).index();
  return buf.str();
}

namespace {

std::string describe(const buffer_id_t &id) {
  std::ostringstream buf;
  buf << id.group;
  if (!id.level) {
    buf << " (grid array or scalar)";
    return buf.str();
  }
  buf << " on patch " << id.level->patch().index() << " level "
      << id.level->index();
  if (const std::string *const tag = std::get_if<std::string>(&id.which))
    buf << ", band " << *tag;
  else
    buf << ", time level " << std::get<tl_t>(id.which).index();
  return buf.str();
}

template <typename Ghosts> bool has_ghosts(const Ghosts &nghostzones) {
  for (int d = 0; d < dim; ++d)
    if (nghostzones[d] != 0)
      return true;
  return false;
}

std::vector<std::string> variable_names(const GHExt::GroupRecord &group) {
  std::vector<std::string> vars;
  for (const vi_t vi : group.variables())
    vars.push_back(CCTK_FullVarName(group.var(vi).index()));
  return vars;
}

void refuse_unless_carried(const std::vector<bool> &mask,
                           const codec_limits_t &limits) {
  DECLARE_CCTK_PARAMETERS;
  std::ostringstream uncarried;
  int num_uncarried = 0;
  for (const GHExt::GroupRecord &group : ghext->records()) {
    if (!mask.at(group.groupindex.index()))
      continue;
    std::vector<std::string> reasons;
    if (group.kind != group_kind::grid_function && !limits.carries_arrays)
      reasons.push_back("grid scalars and arrays are not carried");
    if (group.num_timelevels() > limits.max_timelevels)
      reasons.push_back(std::to_string(group.num_timelevels()) +
                        " time levels, at most " +
                        std::to_string(limits.max_timelevels) + " carried");
    if (reasons.empty())
      continue;
    ++num_uncarried;
    uncarried << "\n  " << group.groupname << ": " << reasons.front();
    for (std::size_t i = 1; i < reasons.size(); ++i)
      uncarried << "; " << reasons.at(i);
  }
  if (num_uncarried > 0)
    CCTK_VERROR("Checkpoint method \"%s\" cannot carry %d checkpointed "
                "group%s:%s\nUse \"openpmd\", or tag the groups "
                "checkpoint=\"no\" if they are rebuilt at startup.",
                checkpoint_method, num_uncarried, num_uncarried == 1 ? "" : "s",
                uncarried.str().c_str());
}

const amrex::MultiFab *vertex_coords(const level_t level) {
  if (ghext->patch(level.patch()).is_cartesian)
    return nullptr;
  const int gi = CCTK_GroupIndex("CoordinatesX::vertex_coords");
  if (gi < 0 || !ghext->record(gi_t{gi}).has_storage())
    return nullptr;
  return ghext->level(level).group(gi_t{gi}).timelevel(tl_t{0}).mfab.get();
}

void snapshot_geometry(level_record_t &record, const level_t level) {
  record.geom = ghext->patch(level.patch()).amrcore->Geom(level.index());
  record.vertex_coords = vertex_coords(level);
}

} // namespace

std::unique_ptr<checkpoint_codec_t> make_codec(const char *const method) {
  if (CCTK_EQUALS(method, "openpmd")) {
#ifdef HAVE_CAPABILITY_openPMD_api
    return make_openpmd_codec();
#else
    CCTK_VERROR("The I/O method \"%s\" needs openPMD_api, which is not enabled",
                method);
#endif
  }
  if (CCTK_EQUALS(method, "silo")) {
#ifdef HAVE_CAPABILITY_Silo
    return make_silo_codec();
#else
    CCTK_VERROR("The I/O method \"%s\" needs Silo, which is not enabled",
                method);
#endif
  }
  CCTK_VERROR("Unknown I/O method \"%s\"", method);
}

void refuse_unless_recoverable(const codec_limits_t &limits) {
  DECLARE_CCTK_PARAMETERS;
  if (rechop_on_recovery && !limits.rechops_on_recovery)
    CCTK_VERROR("CarpetX::rechop_on_recovery is set, but the recover method "
                "\"%s\" recovers only the grid the checkpoint was written on. "
                "Recover with \"openpmd\", or set rechop_on_recovery = no.",
                recover_method);
}

bool finer_level_is_behind(const checkpoint_record_t &record,
                           const level_t level) {
  const std::vector<level_record_t> &levels =
      record.levels.at(level.patch().index());
  const std::optional<rat64> &self = levels.at(level.index()).iteration;
  if (!self)
    return false;
  for (int finer = level.index() + 1; finer < int(levels.size()); ++finer) {
    const std::optional<rat64> &finer_iteration = levels.at(finer).iteration;
    if (finer_iteration && *finer_iteration < *self)
      return true;
  }
  return false;
}

void build_buffers(checkpoint_record_t &record, const std::vector<bool> &mask,
                   const record_purpose purpose, const codec_limits_t &limits) {
  for (auto &patchdata : ghext->patches()) {
    for (auto &leveldata : patchdata.levels()) {
      const level_t level = leveldata.level;
      const bool bands = purpose == record_purpose::checkpoint &&
                         finer_level_is_behind(record, level);
      for (auto &groupdata : leveldata.groups()) {
        const GHExt::GroupRecord &group = groupdata.record;
        const gi_t gi = group.groupindex;
        if (!mask.at(gi.index()))
          continue;
        const std::vector<std::string> vars = variable_names(group);

        const tl_range tls = purpose == record_purpose::checkpoint
                                 ? group.timelevels()
                                 : tl_range{tl_t{0}, tl_t{1}};
        for (const tl_t tl : tls)
          record.buffers.push_back(
              {.id = {.group = group.groupname,
                      .vars = vars,
                      .level = level,
                      .which = tl},
               .data = grid_buffer_t{groupdata.timelevel(tl).mfab.get()}});

        if (!bands)
          continue;
        const auto add_band = [&](amrex::MultiFab *const band,
                                  const band_kind kind, const int stage) {
          if (!band || band->empty())
            return;
          assert(band->nGrowVect() == 0);
          record.buffers.push_back(
              {.id = {.group = group.groupname,
                      .vars = vars,
                      .level = level,
                      .which = subcycling_band_tag(kind, stage)},
               .data = grid_buffer_t{band}});
        };
        for (int s = 0; s < max_num_rk_stages; ++s)
          add_band(rk_source_band(level, gi, band_kind::ks_source, s),
                   band_kind::ks_source, s);
        add_band(rk_source_band(level, gi, band_kind::old_source),
                 band_kind::old_source, -1);
        for (int f = 0; f < 2 * dim; ++f)
          add_band(rk_source_band(level, gi, band_kind::flux_register, f),
                   band_kind::flux_register, f);
      }
    }
  }

  if (!limits.carries_arrays)
    return;
  for (const GHExt::GroupRecord &group : ghext->records()) {
    if (group.kind == group_kind::grid_function ||
        !mask.at(group.groupindex.index()))
      continue;
    GHExt::GlobalData::ArrayGroupData &arraydata =
        ghext->record(group.groupindex).array();
    assert(!has_ghosts(arraydata.nghostzones));
    GHExt::GlobalData::AnyTypeVector &data = arraydata.timelevel(tl_t{0}).data;
    record.buffers.push_back(
        {.id = {.group = group.groupname,
                .vars = variable_names(group),
                .level = std::nullopt,
                .which = tl_t{0}},
         .data = array_buffer_t{
             .data = data.data_at(0),
             .vartype = data.type(),
             .shape = {arraydata.gsh[0], arraydata.gsh[1], arraydata.gsh[2]}}});
  }
}

checkpoint_record_t build_record(const cGH *const cctkGH,
                                 const std::vector<bool> &mask,
                                 const record_purpose purpose,
                                 const codec_limits_t &limits) {
  if (purpose == record_purpose::checkpoint)
    refuse_unless_carried(mask, limits);

  checkpoint_record_t record;
  record.purpose = purpose;
  record.iteration = cctkGH->cctk_iteration;
  record.time = cctkGH->cctk_time;
  record.delta_time = cctkGH->cctk_delta_time;
  char *const parameters = IOUtil_GetAllParameters(cctkGH, 1);
  record.parameters = parameters;
  std::free(parameters);

  record.levels.resize(ghext->num_patches());
  for (const auto &patchdata : ghext->patches()) {
    std::vector<level_record_t> &levels =
        record.levels.at(patchdata.patch.index());
    levels.resize(patchdata.num_levels());
    for (const auto &leveldata : patchdata.levels()) {
      level_record_t &level = levels.at(leveldata.level.index());
      const amrex::FabArrayBase &fab = *leveldata.fab;
      for (int component = 0; component < fab.size(); ++component)
        level.boxes.push_back(fab.box(component));
      level.iteration = leveldata.iteration;
      level.last_step_iteration = leveldata.last_step_iteration;
      snapshot_geometry(level, leveldata.level);
    }
  }

  build_buffers(record, mask, purpose, limits);
  return record;
}

void apply_structure(cGH *const cctkGH, checkpoint_record_t &record) {
  DECLARE_CCTK_PARAMETERS;

  const int npatches = record.levels.size();
  if (npatches != ghext->num_patches())
    CCTK_VERROR(
        "Wrong number of patches: Expected %d, found %d in the checkpoint file",
        ghext->num_patches(), npatches);

  cctkGH->cctk_iteration = record.iteration;
  cctkGH->cctk_time = record.time;

  for (auto &patchdata : ghext->patches()) {
    std::vector<level_record_t> &levels =
        record.levels.at(patchdata.patch.index());
    const int nlevels = levels.size();
    assert(nlevels > 0);
    CactusAmrCore &amrcore = *patchdata.amrcore;
    amrcore.SetFinestLevel(nlevels - 1);

    for (int lev = 0; lev < nlevels; ++lev) {
      const level_t level{patchdata.patch, lev};

      if (lev > 0) {
        amrex::Geometry geom = amrcore.Geom(lev - 1);
        geom.refine({2, 2, 2});
        amrcore.SetGeometry(lev, geom);
      }

      const std::vector<amrex::Box> &boxes = levels.at(lev).boxes;
      amrex::BoxArray ba(amrex::BoxList(
          amrex::Vector<amrex::Box>(boxes.begin(), boxes.end())));
      if (rechop_on_recovery)
        ba = amrcore.RechopLevel(level, ba);
      amrcore.SetBoxArray(lev, ba);

      const amrex::DistributionMapping dm(ba);
      amrcore.SetDistributionMap(lev, dm);

      amrcore.make_level(level, ba, dm, "Recovering");
      snapshot_geometry(levels.at(lev), level);
    }
  }
}

void read_record(checkpoint_codec_t &codec, const checkpoint_record_t &record,
                 const std::string &dir, const std::string &file,
                 const std::string &event) {
  std::ostringstream missing;
  int num_missing = 0;
  bool band_missing = false;
  for (const buffer_t &buffer : record.buffers) {
    if (record.present.count(codec.key(buffer.id)))
      continue;
    ++num_missing;
    missing << "\n  " << describe(buffer.id);
    band_missing |= std::holds_alternative<std::string>(buffer.id.which);
  }
  if (num_missing > 0)
    CCTK_VERROR("The checkpoint \"%s/%s\" at iteration %d lacks %d of the %d "
                "buffers being read:%s%s",
                dir.c_str(), file.c_str(), record.iteration, num_missing,
                int(record.buffers.size()), missing.str().c_str(),
                band_missing
                    ? "\nA mid-cycle checkpoint holds the bands of every level "
                      "whose finer level is behind. This one was written by a "
                      "version that did not save them, or is incomplete. "
                      "Restart from a time-aligned checkpoint."
                    : "");

  std::vector<write_scope_t> scopes;
  std::vector<array_write_scope_t> array_scopes;
  for (const buffer_t &buffer : record.buffers) {
    const tl_t *const tl = std::get_if<tl_t>(&buffer.id.which);
    if (!tl)
      continue;
    const gi_t gi{CCTK_GroupIndex(buffer.id.group.c_str())};
    assert(gi.index() >= 0);
    if (buffer.id.level) {
      const auto &groupdata = ghext->level(*buffer.id.level).group(gi);
      scopes.push_back(open_write(window_t::one_level(*buffer.id.level),
                                  {.gi = gi,
                                   .tl = *tl,
                                   .fills = has_ghosts(groupdata.nghostzones)
                                                ? make_valid_int()
                                                : make_valid_all(),
                                   .keeps = valid_t(),
                                   .event = event}));
    } else {
      const auto &arraydata = ghext->record(gi).array();
      array_scopes.push_back(open_array_write(
          {.gi = gi,
           .tl = *tl,
           .fills = has_ghosts(arraydata.nghostzones) ? make_valid_int()
                                                      : make_valid_all(),
           .keeps = valid_t(),
           .event = event}));
    }
  }

  codec.decode_data(record, dir, file);

  scopes.clear();
  array_scopes.clear();
}

} // namespace CarpetX
