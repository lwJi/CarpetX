#include "io/io_model.hxx"

#include "hierarchy/model.hxx"
#include "hierarchy/valid.hxx"
#include "util/var_type.hxx"

namespace std {
template <typename T, size_t N>
YAML::Emitter &operator<<(YAML::Emitter &yaml, const std::array<T, N> &arr) {
  yaml << YAML::Flow << YAML::BeginSeq;
  for (const auto &elt : arr)
    yaml << elt;
  yaml << YAML::EndSeq;
  return yaml;
}
} // namespace std

namespace amrex {

YAML::Emitter &operator<<(YAML::Emitter &yaml, const amrex::IntVect &iv) {
  yaml << YAML::Flow << YAML::BeginSeq;
  for (int d = 0; d < AMREX_SPACEDIM; ++d)
    yaml << iv[d];
  yaml << YAML::EndSeq;
  return yaml;
}

YAML::Emitter &operator<<(YAML::Emitter &yaml, const amrex::Box &box) {
  yaml << YAML::LocalTag("box-1.0.0");
  yaml << YAML::Flow << YAML::BeginMap;
  yaml << YAML::Key << "small" << YAML::Value << box.smallEnd();
  yaml << YAML::Key << "big" << YAML::Value << box.bigEnd();
  yaml << YAML::EndMap;
  return yaml;
}

YAML::Emitter &operator<<(YAML::Emitter &yaml, const amrex::RealBox &rbox) {
  yaml << YAML::LocalTag("realbox-1.0.0");
  yaml << YAML::Flow << YAML::BeginMap;
  yaml << YAML::Key << "xlo" << YAML::Value
       << std::vector<amrex::Real>(rbox.lo(), rbox.lo() + AMREX_SPACEDIM);
  yaml << YAML::Key << "xhi" << YAML::Value
       << std::vector<amrex::Real>(rbox.hi(), rbox.hi() + AMREX_SPACEDIM);
  yaml << YAML::EndMap;
  return yaml;
}

YAML::Emitter &operator<<(YAML::Emitter &yaml, const amrex::BoxArray &ba) {
  yaml << YAML::LocalTag("boxarray-1.0.0");
  yaml << YAML::BeginSeq;
  for (int n = 0; n < ba.size(); ++n)
    yaml << ba[n];
  yaml << YAML::EndSeq;
  return yaml;
}

YAML::Emitter &operator<<(YAML::Emitter &yaml, const amrex::Geometry &geom) {
  yaml << YAML::LocalTag("geometry-1.0.0");
  yaml << YAML::BeginMap;
  yaml << YAML::Key << "prob_domain" << YAML::Value << geom.ProbDomain();
  yaml << YAML::Key << "domain" << YAML::Value << geom.Domain();
  yaml << YAML::Key << "is_periodic" << YAML::Value << geom.isPeriodic();
  yaml << YAML::EndMap;
  return yaml;
}

YAML::Emitter &operator<<(YAML::Emitter &yaml,
                          const amrex::DistributionMapping &dm) {
  yaml << YAML::LocalTag("distributionmapping-1.0.0");
  yaml << YAML::BeginMap;
  yaml << YAML::Key << "processorMap" << YAML::Value << YAML::Flow
       << dm.ProcessorMap();
  yaml << YAML::EndMap;
  return yaml;
}

YAML::Emitter &operator<<(YAML::Emitter &yaml, const amrex::FabArrayBase &fab) {
  yaml << YAML::LocalTag("fabarraybase-1.0.0");
  yaml << YAML::BeginMap;
  yaml << YAML::Key << "ixType" << YAML::Value << fab.ixType().toIntVect();
  yaml << YAML::Key << "nGrowVect" << YAML::Value << fab.nGrowVect();
  yaml << YAML::Key << "boxArray" << YAML::Value << fab.boxArray();
  yaml << YAML::EndMap;
  return yaml;
}

YAML::Emitter &operator<<(YAML::Emitter &yaml, const amrex::AmrCore &amrcore) {
  yaml << YAML::LocalTag("amrcore-1.0.0");
  yaml << YAML::BeginMap;
  yaml << YAML::Key << "maxLevel" << YAML::Value << amrcore.maxLevel();
  yaml << YAML::Key << "finestLevel" << YAML::Value << amrcore.finestLevel();
  yaml << YAML::Key << "geometry" << YAML::Value << amrcore.Geom();
  yaml << YAML::Key << "distributionMapping" << YAML::Value
       << amrcore.DistributionMap();
  yaml << YAML::Key << "boxArray" << YAML::Value << amrcore.boxArray();
  yaml << YAML::EndMap;
  return yaml;
}

} // namespace amrex

namespace CarpetX {

template <typename T, size_t N>
inline std::vector<T> seq(const std::array<T, N> &v) {
  std::vector<T> r;
  for (const auto &x : v)
    r.push_back(x);
  return r;
}

template <typename T, size_t N>
inline std::vector<std::vector<T> >
seqs(const std::vector<std::array<T, N> > &v) {
  std::vector<std::vector<T> > r;
  for (const auto &x : v)
    r.push_back(seq(x));
  return r;
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

YAML::Emitter &operator<<(YAML::Emitter &yaml, const why_valid_t &why) {
  const why_valid_t::reasons_t reasons = why.reasons();
  yaml << YAML::LocalTag("why_valid-2.0.0");
  yaml << YAML::BeginMap;
  yaml << YAML::Key << "valid" << YAML::Value << why.get();
  yaml << YAML::Key << "why" << YAML::Value << YAML::BeginMap;
  yaml << YAML::Key << "int" << YAML::Value << reasons.interior;
  yaml << YAML::Key << "outer" << YAML::Value << reasons.outer;
  yaml << YAML::Key << "same_level" << YAML::Value << reasons.same_level;
  yaml << YAML::Key << "coarse_fine" << YAML::Value << reasons.coarse_fine;
  yaml << YAML::EndMap;
  yaml << YAML::EndMap;
  return yaml;
}

YAML::Emitter &operator<<(YAML::Emitter &yaml,
                          const GHExt::GroupRecord &group) {
  yaml << YAML::LocalTag("commongroupdata-1.1.0");
  yaml << YAML::BeginMap;
  yaml << YAML::Key << "groupname" << YAML::Value
       << CCTK_FullGroupName(group.groupindex.index());
  yaml << YAML::Key << "numvars" << YAML::Value << group.numvars;
  yaml << YAML::Key << "varnames" << YAML::Value << YAML::Flow
       << YAML::BeginSeq;
  for (const vi_t vi : group.variables())
    yaml << CCTK_VarName(group.var(vi).index());
  yaml << YAML::EndSeq;
  yaml << YAML::Key << "do_checkpoint" << YAML::Value << group.do_checkpoint;
  yaml << YAML::Key << "do_evolve" << YAML::Value << group.do_evolve;
  yaml << YAML::Key << "do_restrict" << YAML::Value << group.do_restrict;
  yaml << YAML::Key << "restrict_on_sync" << YAML::Value
       << group.restrict_on_sync;
  if (group.kind == group_kind::grid_function) {
    const GHExt::GroupRecord::grid_function_t &grid_function =
        group.grid_function();
    yaml << YAML::Key << "grid_function" << YAML::Value << YAML::BeginMap;
    yaml << YAML::Key << "indextype" << YAML::Value << YAML::Flow
         << seq(grid_function.indextype);
    yaml << YAML::Key << "nghostzones" << YAML::Value << YAML::Flow
         << seq(grid_function.nghostzones);
    yaml << YAML::Key << "parities" << YAML::Value << YAML::Flow
         << seqs(grid_function.parities);
    yaml << YAML::Key << "fluxes" << YAML::Value << YAML::Flow
         << YAML::BeginSeq;
    for (int d = 0; d < dim; ++d)
      yaml << (grid_function.fluxes
                   ? CCTK_FullGroupName((*grid_function.fluxes)[d].index())
                   : "");
    yaml << YAML::EndSeq;
    yaml << YAML::EndMap;
  }
  yaml << YAML::EndMap;
  return yaml;
}

YAML::Emitter &operator<<(YAML::Emitter &yaml, const CCTK_COMPLEX &cval) {
  yaml << YAML::Flow << YAML::BeginSeq << cval.real() << cval.imag()
       << YAML::EndSeq;
  return yaml;
};

YAML::Emitter &
operator<<(YAML::Emitter &yaml,
           const GHExt::GlobalData::AnyTypeVector::AnyTypeScalarRef
               &anytypescalarref) {
  visit(anytypescalarref.type(), [&](const auto tag) {
    using T = typename decltype(tag)::type;
    yaml << *static_cast<const T *>(anytypescalarref.data());
  });
  return yaml;
}

YAML::Emitter &
operator<<(YAML::Emitter &yaml,
           const GHExt::GlobalData::AnyTypeVector &anytypevector) {
  yaml << YAML::BeginSeq;
  for (size_t i = 0; i < anytypevector.size(); ++i)
    yaml << anytypevector[i];
  yaml << YAML::EndSeq;
  return yaml;
}

YAML::Emitter &
operator<<(YAML::Emitter &yaml,
           const GHExt::GlobalData::ArrayGroupData &arraygroupdata) {
  yaml << YAML::LocalTag("arraygroupdata-1.0.0");
  yaml << YAML::BeginMap;
  yaml << YAML::Key << "commongroupdata" << YAML::Value
       << arraygroupdata.record;
  yaml << YAML::Key << "valid" << YAML::Value << YAML::BeginSeq;
  for (const tl_t tl : arraygroupdata.record.timelevels()) {
    yaml << YAML::BeginSeq;
    for (const vi_t vi : arraygroupdata.record.variables())
      yaml << arraygroupdata.timelevel(tl).why_valid(vi);
    yaml << YAML::EndSeq;
  }
  yaml << YAML::EndSeq;
  yaml << YAML::Key << "data" << YAML::Value << YAML::BeginSeq;
  for (const tl_t tl : arraygroupdata.record.timelevels())
    yaml << arraygroupdata.timelevel(tl).data;
  yaml << YAML::EndSeq;
  yaml << YAML::EndMap;
  return yaml;
}

YAML::Emitter &
operator<<(YAML::Emitter &yaml,
           const GHExt::PatchData::LevelData::GroupData &groupdata) {
  yaml << YAML::LocalTag("groupdata-1.1.0");
  yaml << YAML::BeginMap;
  yaml << YAML::Key << "commongroupdata" << YAML::Value << groupdata.record;
  yaml << YAML::Key << "valid" << YAML::Value << YAML::BeginSeq;
  for (const tl_t tl : tl_range{tl_t{0}, tl_t{groupdata.num_timelevels()}}) {
    yaml << YAML::BeginSeq;
    for (const vi_t vi : groupdata.record.variables())
      yaml << groupdata.timelevel(tl).why_valid(vi);
    yaml << YAML::EndSeq;
  }
  yaml << YAML::EndSeq;
  yaml << YAML::Key << "patch" << YAML::Value
       << groupdata.level.patch().index();
  yaml << YAML::Key << "level" << YAML::Value << groupdata.level.index();
  yaml << YAML::EndMap;
  return yaml;
}

YAML::Emitter &operator<<(YAML::Emitter &yaml,
                          const GHExt::PatchData::LevelData &leveldata) {
  yaml << YAML::LocalTag("leveldata-1.1.0");
  yaml << YAML::BeginMap;
  yaml << YAML::Key << "patch" << YAML::Value
       << leveldata.level.patch().index();
  yaml << YAML::Key << "level" << YAML::Value << leveldata.level.index();
  yaml << YAML::Key << "is_subcycling_level" << YAML::Value
       << (driver().use_subcycling && leveldata.level.index() > 0);
  yaml << YAML::Key << "iteration" << YAML::Value << leveldata.iteration;
  yaml << YAML::Key << "delta_iteration" << YAML::Value
       << leveldata.delta_iteration;
  yaml << YAML::Key << "fab" << YAML::Value << *leveldata.fab;
  yaml << YAML::Key << "groupdata" << YAML::Value << YAML::BeginSeq;
  for (const auto &groupdata : leveldata.all_groups())
    yaml << groupdata;
  yaml << YAML::EndSeq;
  yaml << YAML::EndMap;
  return yaml;
}

YAML::Emitter &operator<<(YAML::Emitter &yaml,
                          const GHExt::PatchData &patchdata) {
  yaml << YAML::LocalTag("patchdata-1.0.0");
  yaml << YAML::BeginMap;
  yaml << YAML::Key << "patch" << YAML::Value << patchdata.patch.index();
  yaml << YAML::Key << "amrcore" << YAML::Value << *patchdata.amrcore;
  yaml << YAML::Key << "leveldata" << YAML::Value << YAML::BeginSeq;
  for (const auto &leveldata : patchdata.levels())
    yaml << leveldata;
  yaml << YAML::EndSeq;
  yaml << YAML::EndMap;
  return yaml;
}

YAML::Emitter &operator<<(YAML::Emitter &yaml, const GHExt &ext) {
  yaml << YAML::LocalTag("ghext-2.0.0");
  yaml << YAML::BeginMap;
  yaml << YAML::Key << "globaldata" << YAML::Value;
  yaml << YAML::LocalTag("globaldata-1.0.0");
  yaml << YAML::BeginMap;
  yaml << YAML::Key << "arraygroupdata" << YAML::Value << YAML::BeginSeq;
  for (const gi_t gi : ext.group_handles()) {
    const GHExt::GroupRecord &group = ext.record(gi);
    if (group.kind != group_kind::grid_function)
      yaml << group.array();
  }
  yaml << YAML::EndSeq;
  yaml << YAML::EndMap;
  yaml << YAML::Key << "patchdata" << YAML::Value << YAML::BeginSeq;
  for (const auto &patchdata : ext.patches())
    yaml << patchdata;
  yaml << YAML::EndSeq;
  yaml << YAML::EndMap;
  return yaml;
}

} // namespace CarpetX
