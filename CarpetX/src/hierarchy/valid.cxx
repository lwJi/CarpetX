#include "hierarchy/driver.hxx"
#include "hierarchy/loops.hxx"
#include <carpetx.hxx>
#include "hierarchy/valid.hxx"
#include "util/var_type.hxx"

#include <loop_device.hxx>

#include <cctk.h>
#include <cctk_Parameters.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>
#include <limits>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace CarpetX {

using Arith::vect;
using Loop::where_t;

////////////////////////////////////////////////////////////////////////////////

std::string valid_t::explanation() const {
  const auto valstr = [](bool v) { return v ? "valid" : "invalid"; };
  std::ostringstream buf;
  buf << "\n"
      << "  The interior is " << valstr(valid_int) << ".\n"
      << "  The outer boundary is " << valstr(valid_outer) << ".\n"
      << "  The ghost zones are " << valstr(valid_ghosts) << ".\n";
  return buf.str();
}

std::ostream &operator<<(std::ostream &os, const valid_t v) {
  auto str = [](bool v) { return v ? "VAL" : "INV"; };
  return os << "[int:" << str(v.valid_int) << ",outer:" << str(v.valid_outer)
            << ",ghosts:" << str(v.valid_ghosts) << "]";
}

valid_t::operator std::string() const {
  std::ostringstream buf;
  buf << *this;
  return buf.str();
}

YAML::Emitter &operator<<(YAML::Emitter &yaml, const valid_t v) {
  yaml << YAML::LocalTag("valid-1.0.0");
  yaml << YAML::Flow << YAML::BeginMap;
  yaml << YAML::Key << "int" << YAML::Value << v.valid_int;
  yaml << YAML::Key << "outer" << YAML::Value << v.valid_outer;
  yaml << YAML::Key << "ghosts" << YAML::Value << v.valid_ghosts;
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
      << "  The ghost zones are " << valstr(valid.valid_ghosts)
      << " because: " << why_ghosts() << ".\n";
  return buf.str();
}

std::ostream &operator<<(std::ostream &os, const why_valid_t &why) {
  return os << why.valid << ","
            << "why{int:" << why.why_int() << ","
            << "outer:" << why.why_outer() << ","
            << "ghosts:" << why.why_ghosts() << "}";
}

why_valid_t::operator std::string() const {
  std::ostringstream buf;
  buf << *this;
  return buf.str();
}

YAML::Emitter &operator<<(YAML::Emitter &yaml, const why_valid_t &why) {
  yaml << YAML::LocalTag("why_valid-1.0.0");
  yaml << YAML::BeginMap;
  yaml << YAML::Key << "valid" << YAML::Value << why.valid;
  yaml << YAML::Key << "why" << YAML::Value << YAML::BeginMap;
  yaml << YAML::Key << "int" << YAML::Value << why.why_int();
  yaml << YAML::Key << "outer" << YAML::Value << why.why_outer();
  yaml << YAML::Key << "ghosts" << YAML::Value << why.why_ghosts();
  yaml << YAML::EndMap;
  yaml << YAML::EndMap;
  return yaml;
}

////////////////////////////////////////////////////////////////////////////////

namespace {

using GroupData = GHExt::PatchData::LevelData::GroupData;
using ArrayGroupData = GHExt::GlobalData::ArrayGroupData;

void error_if_invalid(const GroupData &groupdata, const vi_t vi, const tl_t tl,
                      const valid_t &required,
                      const std::function<std::string()> &msg,
                      const std::function<std::string()> &hint) {
  const valid_t have = groupdata.timelevel(tl).valid(vi);
  if (CCTK_BUILTIN_EXPECT((required & ~have).valid_any(), false))
    CCTK_VERROR("%s: Grid function \"%s\" is invalid on patch %d, refinement "
                "level %d, time level %d; required: %s, found: %s%s",
                msg().c_str(),
                CCTK_FullVarName(groupdata.record.var(vi).index()),
                groupdata.level.patch().index(), groupdata.level.index(),
                tl.index(), required.explanation().c_str(),
                groupdata.timelevel(tl).why_valid(vi).explanation().c_str(),
                hint ? hint().c_str() : "");
}

void warn_if_invalid(const GroupData &groupdata, const vi_t vi, const tl_t tl,
                     const valid_t &required,
                     const std::function<std::string()> &msg) {
  const valid_t have = groupdata.timelevel(tl).valid(vi);
  if (CCTK_BUILTIN_EXPECT((required & ~have).valid_any(), false))
    CCTK_VWARN(CCTK_WARN_ALERT,
               "%s: Grid function \"%s\" is invalid on patch %d, refinement "
               "level %d, time level %d; required: %s, found: %s",
               msg().c_str(),
               CCTK_FullVarName(groupdata.record.var(vi).index()),
               groupdata.level.patch().index(), groupdata.level.index(),
               tl.index(), required.explanation().c_str(),
               groupdata.timelevel(tl).why_valid(vi).explanation().c_str());
}

void error_if_invalid(const ArrayGroupData &groupdata, const vi_t vi,
                      const tl_t tl, const valid_t &required,
                      const std::function<std::string()> &msg) {
  const valid_t have = groupdata.timelevel(tl).valid(vi);
  if (CCTK_BUILTIN_EXPECT((required & ~have).valid_any(), false))
    CCTK_VERROR(
        "%s: Array \"%s\" is invalid on time level %d; required %s, found %s",
        msg().c_str(), CCTK_FullVarName(groupdata.record.var(vi).index()),
        tl.index(), required.explanation().c_str(),
        groupdata.timelevel(tl).why_valid(vi).explanation().c_str());
}

void poison_invalid_gf(const window_t window, const gi_t gi, const vi_t vi,
                       const tl_t tl, const valid_t where) {
  DECLARE_CCTK_PARAMETERS;
  if (!poison_undefined_values || !where.valid_any())
    return;

  static Timer timer("poison_invalid<GF>");
  Interval interval(timer);

  const poison_value_t<CCTK_REAL> poison_value;
  CCTK_REAL poison;
  poison_value.set_to_poison(poison);

  loop_parallel(window, [&](const GHExt::PatchData::LevelData &leveldata,
                            const int index, const int component,
                            const cGH *restrict const cctkGH) {
    const auto &restrict groupdata = leveldata.group(gi);

    const Loop::GridDescBaseDevice grid(cctkGH);
    const Loop::GF3D2layout layout(cctkGH, groupdata.indextype);
    const Loop::GF3D2<CCTK_REAL> gf(
        layout, static_cast<CCTK_REAL *>(CCTK_VarDataPtrI(
                    cctkGH, tl.index(), groupdata.record.var(vi).index())));

    const auto set_poison =
        [=] CCTK_DEVICE(const Loop::PointDesc &p)
            CCTK_ATTRIBUTE_ALWAYS_INLINE { gf(p.I) = poison; };

    if (where.valid_all()) {
      grid.loop_device_idx<where_t::everywhere>(
          groupdata.indextype, groupdata.nghostzones, set_poison);
    } else {
      if (where.valid_int)
        grid.loop_device_idx<where_t::interior>(
            groupdata.indextype, groupdata.nghostzones, set_poison);
      if (where.valid_outer)
        grid.loop_device_idx<where_t::boundary>(
            groupdata.indextype, groupdata.nghostzones, set_poison);
      if (where.valid_ghosts)
        grid.loop_device_idx<where_t::ghosts>(
            groupdata.indextype, groupdata.nghostzones, set_poison);
    }
  });

  synchronize();
}

void check_valid_gf(const window_t window, const gi_t gi, const vi_t vi,
                    const tl_t tl, const valid_t where,
                    const std::function<std::string()> &msg) {
  DECLARE_CCTK_PARAMETERS;
  if (!poison_undefined_values || !where.valid_any())
    return;

  static Timer timer("check_valid<GF>");
  Interval interval(timer);

  const auto is_poison = [] CCTK_DEVICE CCTK_HOST(const CCTK_REAL val)
                             CCTK_ATTRIBUTE_ALWAYS_INLINE {
                               poison_value_t<CCTK_REAL> const poison_value;
                               if (poison_value.is_poison(val))
                                 return true;
                               using std::isnan;
                               if (isnan(val))
                                 return true;
                               return false;
                             };

  amrex::FArrayBox poison_found(
      amrex::Box(amrex::IntVect(0, 0, 0), amrex::IntVect(0, 0, 0)), 1,
      amrex::The_Async_Arena());
#ifdef AMREX_USE_GPU
  constexpr auto run_on = amrex::RunOn::Device;
#else
  constexpr auto run_on = amrex::RunOn::Host;
#endif
  poison_found.operator= <run_on>(0.0);
  CCTK_REAL *restrict const poison_found_ptr = poison_found.dataPtr();

  loop_parallel(window, [&](const GHExt::PatchData::LevelData &leveldata,
                            const int index, const int component,
                            const cGH *restrict const cctkGH) {
    const auto &restrict groupdata = leveldata.group(gi);

    const Loop::GridDescBaseDevice grid(cctkGH);
    const Loop::GF3D2layout layout(cctkGH, groupdata.indextype);
    const Loop::GF3D2<const CCTK_REAL> gf(
        layout, static_cast<const CCTK_REAL *>(CCTK_VarDataPtrI(
                    cctkGH, tl.index(), groupdata.record.var(vi).index())));

    const auto update_poison_found =
        [=] CCTK_DEVICE(const Loop::PointDesc &p) CCTK_ATTRIBUTE_ALWAYS_INLINE {
          if (CCTK_BUILTIN_EXPECT(!is_poison(gf(p.I)), true))
            return;
#pragma omp atomic write
          *poison_found_ptr = 0.0 / 0.0;
        };

    if (where.valid_all()) {
      grid.loop_device_idx<where_t::everywhere>(
          groupdata.indextype, groupdata.nghostzones, update_poison_found);
    } else {
      if (where.valid_int)
        grid.loop_device_idx<where_t::interior>(
            groupdata.indextype, groupdata.nghostzones, update_poison_found);
      if (where.valid_outer)
        grid.loop_device_idx<where_t::boundary>(
            groupdata.indextype, groupdata.nghostzones, update_poison_found);
      if (where.valid_ghosts)
        grid.loop_device_idx<where_t::ghosts>(
            groupdata.indextype, groupdata.nghostzones, update_poison_found);
    }
  });
  synchronize();

  if (!poison_found.contains_nan<run_on>())
    return;

  std::size_t nan_count{0};
  std::array<int, 3> nan_imin, nan_imax;
  std::array<CCTK_REAL, 3> nan_xmin, nan_xmax;
  for (int d = 0; d < 3; ++d) {
    nan_imin[d] = std::numeric_limits<int>::max();
    nan_imax[d] = std::numeric_limits<int>::min();
    nan_xmin[d] = +1.0 / 0.0;
    nan_xmax[d] = -1.0 / 0.0;
  }

  struct info_t {
    where_t where;
    int patch, level, component;
    vect<int, dim> I;
    vect<CCTK_REAL, dim> X;
    CCTK_REAL val;
  };
  std::vector<info_t> infos;

  loop_serially(window, [&](const GHExt::PatchData::LevelData &leveldata,
                            const int index, const int component,
                            const cGH *restrict const cctkGH) {
    const auto &restrict groupdata = leveldata.group(gi);

    const Loop::GridDescBaseDevice grid(cctkGH);
    const Loop::GF3D2layout layout(cctkGH, groupdata.indextype);
    const Loop::GF3D2<const CCTK_REAL> gf(
        layout, static_cast<const CCTK_REAL *>(CCTK_VarDataPtrI(
                    cctkGH, tl.index(), groupdata.record.var(vi).index())));

    const auto update_nan_count = [&](const Loop::PointDesc &p,
                                      const where_t where) {
      if (CCTK_BUILTIN_EXPECT(!is_poison(gf(p.I)), true))
        return;

      using std::fmax, std::fmin, std::max, std::min;
      ++nan_count;
      nan_imin[0] = min(nan_imin[0], grid.lbnd[0] + p.i);
      nan_imin[1] = min(nan_imin[1], grid.lbnd[1] + p.j);
      nan_imin[2] = min(nan_imin[2], grid.lbnd[2] + p.k);
      nan_imax[0] = max(nan_imax[0], grid.lbnd[0] + p.i);
      nan_imax[1] = max(nan_imax[1], grid.lbnd[1] + p.j);
      nan_imax[2] = max(nan_imax[2], grid.lbnd[2] + p.k);
      nan_xmin[0] = fmin(nan_xmin[0], p.x);
      nan_xmin[1] = fmin(nan_xmin[1], p.y);
      nan_xmin[2] = fmin(nan_xmin[2], p.z);
      nan_xmax[0] = fmax(nan_xmax[0], p.x);
      nan_xmax[1] = fmax(nan_xmax[1], p.y);
      nan_xmax[2] = fmax(nan_xmax[2], p.z);

      infos.push_back(
          info_t{where, p.patch, p.level, p.component, p.I, p.X, gf(p.I)});
    };

    if (where.valid_int)
      grid.loop_idx(where_t::interior, groupdata.indextype,
                    groupdata.nghostzones, [&](const Loop::PointDesc &p) {
                      update_nan_count(p, where_t::interior);
                    });
    if (where.valid_outer)
      grid.loop_idx(where_t::boundary, groupdata.indextype,
                    groupdata.nghostzones, [&](const Loop::PointDesc &p) {
                      update_nan_count(p, where_t::boundary);
                    });
    if (where.valid_ghosts)
      grid.loop_idx(where_t::ghosts, groupdata.indextype, groupdata.nghostzones,
                    [&](const Loop::PointDesc &p) {
                      update_nan_count(p, where_t::ghosts);
                    });
  });

  const var_t var = driver().record(gi).var(vi);
  CCTK_VWARN(CCTK_WARN_ALERT,
             "%s: Grid function \"%s\" contains %td nans, infinities, or "
             "poison in box (%g,%g,%g):(%g,%g,%g); expected valid %s",
             msg().c_str(), CCTK_FullVarName(var.index()),
             std::size_t(nan_count), double(nan_xmin[0]), double(nan_xmin[1]),
             double(nan_xmin[2]), double(nan_xmax[0]), double(nan_xmax[1]),
             double(nan_xmax[2]), where.explanation().c_str());

  std::sort(infos.begin(), infos.end(), [](const info_t &a, const info_t &b) {
    if (a.level < b.level)
      return true;
    if (a.level > b.level)
      return false;
    if (a.patch < b.patch)
      return true;
    if (a.patch > b.patch)
      return false;
    if (a.component < b.component)
      return true;
    if (a.component > b.component)
      return false;
    const std::less<vect<int, dim> > lt;
    return lt(reversed(a.I), reversed(b.I));
  });

  std::ostringstream buf;
  buf << std::setprecision(std::numeric_limits<CCTK_REAL>::digits10 + 1);
  for (const auto &info : infos)
    buf << "\n"
        << info.where << " level " << info.level << " patch " << info.patch
        << " component " << info.component << " " << info.I << " " << info.X
        << " " << info.val;
  CCTK_WARN(CCTK_WARN_ALERT, buf.str().c_str());

  CCTK_VERROR("%s: Grid function \"%s\" contains nans, infinities, or poison; "
              "expected valid %s",
              msg().c_str(), CCTK_FullVarName(var.index()),
              where.explanation().c_str());
}

int array_elements(const ArrayGroupData &arraygroupdata) {
  int n_elems = 1;
  for (int i = 0; i < arraygroupdata.dimension; i++)
    n_elems *= arraygroupdata.gsh[i];
  return n_elems;
}

void poison_ga_interior(ArrayGroupData &arraygroupdata, const vi_t vi,
                        const tl_t tl) {
  static Timer timer("poison_invalid<GA>");
  Interval interval(timer);

  const int n_elems = array_elements(arraygroupdata);
  visit(arraygroupdata.record.var_type, [&](const auto tag) {
    using T = typename decltype(tag)::type;
    T *restrict const ptr = static_cast<T *>(
        arraygroupdata.timelevel(tl).data.data_at(vi.index() * n_elems));
    const poison_value_t<T> poison_value;
    poison_value.set_to_poison(ptr, n_elems);
  });
}

template <typename T> bool has_nan(const T &value) {
  using std::isnan;
  if constexpr (std::is_same_v<T, CCTK_COMPLEX>)
    return isnan(value.real()) || isnan(value.imag());
  else if constexpr (std::is_floating_point_v<T>)
    return isnan(value);
  else
    return false;
}

std::size_t count_ga_poison(const ArrayGroupData &arraygroupdata, const vi_t vi,
                            const tl_t tl) {
  static Timer timer("check_valid<GA>");
  Interval interval(timer);

  const int n_elems = array_elements(arraygroupdata);
  return visit(arraygroupdata.record.var_type, [&](const auto tag) {
    using T = typename decltype(tag)::type;
    const poison_value_t<T> poison_value;
    const T *restrict const ptr = static_cast<const T *>(
        arraygroupdata.timelevel(tl).data.data_at(vi.index() * n_elems));
    std::size_t nan_count{0};
    for (int i = 0; i < n_elems; i++)
      if (CCTK_BUILTIN_EXPECT(poison_value.is_poison(ptr[i]) || has_nan(ptr[i]),
                              false))
        ++nan_count;
    return nan_count;
  });
}

void poison_invalid_ga(const gi_t gi, const vi_t vi, const tl_t tl,
                       const valid_t where) {
  DECLARE_CCTK_PARAMETERS;
  if (!poison_undefined_arrays || !where.valid_int)
    return;
  poison_ga_interior(driver().record(gi).array(), vi, tl);
}

void check_valid_ga(const gi_t gi, const vi_t vi, const tl_t tl,
                    const valid_t where,
                    const std::function<std::string()> &msg) {
  DECLARE_CCTK_PARAMETERS;
  if (!poison_undefined_arrays || !where.valid_int)
    return;
  const auto &restrict arraygroupdata = driver().record(gi).array();
  const std::size_t nan_count = count_ga_poison(arraygroupdata, vi, tl);
  if (nan_count == 0)
    return;
  CCTK_VERROR("%s: Grid array \"%s\" has %td nans on time level %d; "
              "expected valid %s",
              msg().c_str(),
              CCTK_FullVarName(arraygroupdata.record.var(vi).index()),
              nan_count, tl.index(), where.explanation().c_str());
}

////////////////////////////////////////////////////////////////////////////////

int current_iteration() {
  const cGH *const cctkGH = driver().get_global_cctkGH();
  return cctkGH ? cctkGH->cctk_iteration : 0;
}

std::function<std::string()> event_reason(const std::string &event,
                                          const char *const what,
                                          const int iteration) {
  const std::string text =
      event + what + " (iteration " + std::to_string(iteration) + ")";
  return [text]() { return text; };
}

std::vector<vi_t> variables_of(const gi_t gi, const std::optional<vi_t> vi) {
  if (vi)
    return {*vi};
  std::vector<vi_t> vis;
  for (const vi_t v : driver().record(gi).variables())
    vis.push_back(v);
  return vis;
}

} // namespace

write_scope_t::write_scope_t(const window_t window, write_spec_t spec,
                             const int iteration)
    : window(window), spec(std::move(spec)), iteration(iteration),
      epoch(carpetx_epoch.load(std::memory_order_relaxed)), live(true) {
  const valid_t destroys = ~this->spec.keeps;
  const auto not_filled =
      event_reason(this->spec.event, " not filled", iteration);
  const std::vector<vi_t> vis = variables_of(this->spec.gi, this->spec.vi);
  loop_serially(window, [&](GHExt::PatchData::LevelData &leveldata) {
    GroupData::timelevel_t &timelevel =
        leveldata.group(this->spec.gi).timelevel(this->spec.tl);
    ++timelevel.open_writes;
    for (const vi_t vi : vis)
      timelevel.valid_.at(vi.index()).set_invalid(destroys, not_filled);
  });
  for (const vi_t vi : vis)
    poison_invalid_gf(window, this->spec.gi, vi, this->spec.tl, destroys);
}

write_scope_t::write_scope_t(write_scope_t &&other) noexcept
    : window(other.window),
      spec(std::move(other.spec)),
      iteration(other.iteration),
      epoch(other.epoch),
      live(other.live) {
  other.live = false;
}

write_scope_t open_write(const window_t window, write_spec_t spec) {
  return write_scope_t(window, std::move(spec), current_iteration());
}

void invalidate(const window_t window, invalidate_spec_t spec) {
  const write_scope_t scope =
      open_write(window, {.gi = spec.gi,
                          .tl = spec.tl,
                          .vi = spec.vi,
                          .fills = {},
                          .keeps = ~spec.where,
                          .event = std::move(spec.event)});
}

write_scope_t::~write_scope_t() {
  if (!live)
    return;
  if (epoch != carpetx_epoch.load(std::memory_order_relaxed))
    CCTK_VERROR("CarpetX write scope \"%s\" (iteration %d) on group %s, time "
                "level %d: the grid hierarchy was regridded while the write "
                "scope was open",
                spec.event.c_str(), iteration,
                driver().record(spec.gi).groupname.c_str(), spec.tl.index());
  const auto msg = event_reason(spec.event, "", iteration);
  const auto filled = event_reason(spec.event, " filled", iteration);
  const std::vector<vi_t> vis = variables_of(spec.gi, spec.vi);
  for (const vi_t vi : vis)
    check_valid_gf(window, spec.gi, vi, spec.tl, spec.fills, msg);
  loop_serially(window, [&](GHExt::PatchData::LevelData &leveldata) {
    GroupData::timelevel_t &timelevel =
        leveldata.group(spec.gi).timelevel(spec.tl);
    for (const vi_t vi : vis)
      timelevel.valid_.at(vi.index()).set_valid(spec.fills, filled);
    --timelevel.open_writes;
  });
}

void require(const window_t window, require_spec_t spec) {
  const auto msg = event_reason(spec.event, "", current_iteration());
  const std::vector<vi_t> vis = variables_of(spec.gi, spec.vi);
  loop_serially(window, [&](GHExt::PatchData::LevelData &leveldata) {
    const GroupData &groupdata = leveldata.group(spec.gi);
    for (const vi_t vi : vis)
      error_if_invalid(groupdata, vi, spec.tl, spec.where, msg, spec.hint);
  });
  for (const vi_t vi : vis)
    check_valid_gf(window, spec.gi, vi, spec.tl, spec.where, msg);
}

void warn(const window_t window, require_spec_t spec) {
  const auto msg = event_reason(spec.event, "", current_iteration());
  const std::vector<vi_t> vis = variables_of(spec.gi, spec.vi);
  loop_serially(window, [&](GHExt::PatchData::LevelData &leveldata) {
    const GroupData &groupdata = leveldata.group(spec.gi);
    for (const vi_t vi : vis)
      warn_if_invalid(groupdata, vi, spec.tl, spec.where, msg);
  });
}

array_write_scope_t::array_write_scope_t(write_spec_t spec, const int iteration)
    : spec(std::move(spec)), iteration(iteration), live(true) {
  const valid_t destroys = ~this->spec.keeps;
  const auto not_filled =
      event_reason(this->spec.event, " not filled", iteration);
  ArrayGroupData::timelevel_t &timelevel =
      driver().record(this->spec.gi).array().timelevel(this->spec.tl);
  for (const vi_t vi : variables_of(this->spec.gi, this->spec.vi)) {
    timelevel.valid_.at(vi.index()).set_invalid(destroys, not_filled);
    poison_invalid_ga(this->spec.gi, vi, this->spec.tl, destroys);
  }
}

array_write_scope_t::array_write_scope_t(array_write_scope_t &&other) noexcept
    : spec(std::move(other.spec)),
      iteration(other.iteration),
      live(other.live) {
  other.live = false;
}

array_write_scope_t open_array_write(write_spec_t spec) {
  return array_write_scope_t(std::move(spec), current_iteration());
}

void invalidate_array(invalidate_spec_t spec) {
  const array_write_scope_t scope =
      open_array_write({.gi = spec.gi,
                        .tl = spec.tl,
                        .vi = spec.vi,
                        .fills = {},
                        .keeps = ~spec.where,
                        .event = std::move(spec.event)});
}

array_write_scope_t::~array_write_scope_t() {
  if (!live)
    return;
  const auto msg = event_reason(spec.event, "", iteration);
  const auto filled = event_reason(spec.event, " filled", iteration);
  ArrayGroupData::timelevel_t &timelevel =
      driver().record(spec.gi).array().timelevel(spec.tl);
  for (const vi_t vi : variables_of(spec.gi, spec.vi)) {
    check_valid_ga(spec.gi, vi, spec.tl, spec.fills, msg);
    timelevel.valid_.at(vi.index()).set_valid(spec.fills, filled);
  }
}

void require_array(require_spec_t spec) {
  const auto msg = event_reason(spec.event, "", current_iteration());
  const ArrayGroupData &arraygroupdata = driver().record(spec.gi).array();
  for (const vi_t vi : variables_of(spec.gi, spec.vi)) {
    error_if_invalid(arraygroupdata, vi, spec.tl, spec.where, msg);
    check_valid_ga(spec.gi, vi, spec.tl, spec.where, msg);
  }
}

////////////////////////////////////////////////////////////////////////////////

// Checksums to catch illegal modifications

checksums_t calculate_checksums(
    const window_t window,
    const std::vector<std::vector<std::vector<valid_t> > > &will_write) {
  DECLARE_CCTK_PARAMETERS;

  checksums_t checksums;

  if (!poison_undefined_values)
    return checksums;

  static Timer timer("calculate_checksums");
  Interval interval(timer);

  loop_parallel(window, [&](const GHExt::PatchData::LevelData &leveldata,
                            const int index, const int component,
                            const cGH *restrict const cctkGH) {
    for (const auto &groupdata : leveldata.groups()) {
      const Loop::GridDescBaseDevice grid(cctkGH);
      const Loop::GF3D2layout layout(cctkGH, groupdata.indextype);

      for (const vi_t vi : groupdata.record.variables()) {
        for (const tl_t tl : groupdata.record.timelevels()) {
          const gi_t gi = groupdata.record.groupindex;
          const tiletag_t tiletag{leveldata.level, component, gi, vi, tl};

          const valid_t valid = groupdata.timelevel(tl).valid(vi);
          if (tl.index() >=
              int(will_write.at(gi.index()).at(vi.index()).size()))
            continue;
          const auto &wr =
              will_write.at(gi.index()).at(vi.index()).at(tl.index());
          valid_t to_check = valid & ~wr;

          // Check only those variables which are valid, and where
          // some part (but not everything) is written
          if (!(wr.valid_any() && to_check.valid_any()))
            continue;

          const Loop::GF3D2<const CCTK_REAL> gf(
              layout,
              static_cast<const CCTK_REAL *>(CCTK_VarDataPtrI(
                  cctkGH, tl.index(), groupdata.record.var(vi).index())));

          checksum_t checksum(to_check);
          checksum.add(tiletag);
          const auto add_point = [&](const Loop::PointDesc &p) {
            checksum.add(gf(p.I));
          };

          if (to_check.valid_int)
            grid.loop_idx(where_t::interior, groupdata.indextype,
                          groupdata.nghostzones, add_point);

          if (to_check.valid_outer)
            grid.loop_idx(where_t::boundary, groupdata.indextype,
                          groupdata.nghostzones, add_point);

          if (to_check.valid_ghosts)
            grid.loop_idx(where_t::ghosts, groupdata.indextype,
                          groupdata.nghostzones, add_point);

#pragma omp critical(CarpetX_calculate_checksums)
          checksums[tiletag] = checksum;
        }
      }
    }
  });

  return checksums;
}

void check_checksums(const window_t window, const checksums_t &checksums,
                     const std::function<std::string()> &where) {
  DECLARE_CCTK_PARAMETERS;

  if (!poison_undefined_values)
    return;
  if (checksums.empty())
    return;

  static Timer timer("check_checksums");
  Interval interval(timer);

  loop_parallel(window, [&](const GHExt::PatchData::LevelData &leveldata,
                            const int index, const int component,
                            const cGH *restrict const cctkGH) {
    for (const auto &groupdata : leveldata.groups()) {
      const Loop::GridDescBaseDevice grid(cctkGH);
      const Loop::GF3D2layout layout(cctkGH, groupdata.indextype);

      for (const vi_t vi : groupdata.record.variables()) {
        for (const tl_t tl : groupdata.record.timelevels()) {
          const gi_t gi = groupdata.record.groupindex;
          const tiletag_t tiletag{leveldata.level, component, gi, vi, tl};

          if (!checksums.count(tiletag))
            continue;

          const auto &old_checksum = checksums.at(tiletag);
          const auto &did_check = old_checksum.where;
          assert(did_check.valid_any());

          const Loop::GF3D2<const CCTK_REAL> gf(
              layout,
              static_cast<const CCTK_REAL *>(CCTK_VarDataPtrI(
                  cctkGH, tl.index(), groupdata.record.var(vi).index())));

          checksum_t checksum(did_check);
          checksum.add(tiletag);
          const auto add_point = [&](const Loop::PointDesc &p) {
            checksum.add(gf(p.I));
          };

          if (did_check.valid_int)
            grid.loop_idx(where_t::interior, groupdata.indextype,
                          groupdata.nghostzones, add_point);

          if (did_check.valid_outer)
            grid.loop_idx(where_t::boundary, groupdata.indextype,
                          groupdata.nghostzones, add_point);

          if (did_check.valid_ghosts)
            grid.loop_idx(where_t::ghosts, groupdata.indextype,
                          groupdata.nghostzones, add_point);

          if (checksum != old_checksum)
#pragma omp critical
            CCTK_VERROR(
                "%s: Checksum mismatch: variable %s, tile %s, "
                "int:%d,outer:%d,ghosts:%d, old checksum %s, new checksum %s",
                where().c_str(),
                CCTK_FullVarName(groupdata.record.var(tiletag.vi).index()),
                std::string(tiletag).c_str(), int(did_check.valid_int),
                int(did_check.valid_outer), int(did_check.valid_ghosts),
                std::string(old_checksum).c_str(),
                std::string(checksum).c_str());
        }
      }
    }
  });
}

} // namespace CarpetX
