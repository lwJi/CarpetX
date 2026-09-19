#include "subcycling_tally.hxx"

#include "driver.hxx"

#include <cctk.h>
#include <cctk_Parameters.h>

#include <AMReX_MultiFab.H>

#include <mpi.h>

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace CarpetX {

namespace subcycling_tally_detail {
bool tally_enabled = false;
} // namespace subcycling_tally_detail

namespace {

using subcycling_tally_detail::tally_enabled;

struct call_counts_t {
  int stream_waits = 0, device_waits = 0, temp_buffers = 0;
};

// One per level index; reset by end_coarse_step()
struct level_tally_t {
  int solver_calls = 0;
  call_counts_t worst_solver_call; // component-wise maximum over the calls
  int max_launches_per_lincomb = 0;
  int exchange_syncs = 0, worst_waits_per_exchange_sync = 0;
  int prolongate_syncs = 0;
};

struct scope_t {
  scope_kind_t kind;
  int level;
  call_counts_t counts;
};

// What one row of the counts file holds, apart from the step and the level
struct row_t {
  int solver_calls = 0;
  call_counts_t per_call;
  int launches_per_lincomb = 0;
  int waits_per_ghost_sync = 0;
  std::int64_t buffer_bytes = 0;

  void fold(const row_t &other) {
    using std::max;
    solver_calls = max(solver_calls, other.solver_calls);
    per_call.stream_waits =
        max(per_call.stream_waits, other.per_call.stream_waits);
    per_call.device_waits =
        max(per_call.device_waits, other.per_call.device_waits);
    per_call.temp_buffers =
        max(per_call.temp_buffers, other.per_call.temp_buffers);
    launches_per_lincomb =
        max(launches_per_lincomb, other.launches_per_lincomb);
    waits_per_ghost_sync =
        max(waits_per_ghost_sync, other.waits_per_ghost_sync);
    // Buffer memory is a state, not a count: keep the latest value
    buffer_bytes = other.buffer_bytes;
  }
};

std::vector<scope_t> scope_stack;
std::vector<level_tally_t> level_tallies;
bool regrid_flagged = false;

// Worst coarse step per level, for the shutdown summary
std::vector<row_t> worst_steady_rows; // steps without a regrid
std::vector<row_t> worst_rows;        // all steps
int num_steady_steps = 0;
int num_steps = 0;

std::ofstream counts_file;

level_tally_t &level_tally(const int level) {
  assert(level >= 0);
  if (level >= int(level_tallies.size()))
    level_tallies.resize(level + 1);
  return level_tallies.at(level);
}

// The scope that is charged, or nullptr if the event is dropped
scope_t *charged_scope() {
  if (scope_stack.empty())
    return nullptr;
  scope_t &scope = scope_stack.back();
  if (scope.kind == scope_kind_t::excluded)
    return nullptr;
  return &scope;
}

std::int64_t local_bytes(const amrex::MultiFab &mfab) {
  std::int64_t bytes = 0;
  for (const int index : mfab.IndexArray())
    bytes += std::int64_t(mfab.fabbox(index).numPts()) * mfab.nComp() *
             std::int64_t(sizeof(CCTK_REAL));
  return bytes;
}

// Memory held by subcycling's persistent buffers, attributed to the level that
// holds it, summed over all processes: source bands on levels with a child,
// RK fill buffers on refined levels
std::vector<std::int64_t> calc_buffer_bytes(const int nlevels) {
  std::vector<std::int64_t> bytes(nlevels, 0);
  for (const auto &patchdata : ghext->patchdata) {
    for (const auto &leveldata : patchdata.leveldata) {
      if (leveldata.level >= nlevels)
        continue;
      for (const auto &groupdataptr : leveldata.groupdata) {
        if (!groupdataptr)
          continue;
        const auto &groupdata = *groupdataptr;
        if (groupdata.old_source_band)
          bytes.at(leveldata.level) += local_bytes(*groupdata.old_source_band);
        for (const auto &band : groupdata.ks_source_band)
          if (band)
            bytes.at(leveldata.level) += local_bytes(*band);
        // The RK fill buffers are held by the (refined) level they fill
        if (groupdata.rk_crse_patch)
          bytes.at(leveldata.level) += local_bytes(*groupdata.rk_crse_patch);
        if (groupdata.rk_fine_patch)
          bytes.at(leveldata.level) += local_bytes(*groupdata.rk_fine_patch);
      }
    }
  }
  if (nlevels > 0)
    MPI_Allreduce(MPI_IN_PLACE, bytes.data(), nlevels, MPI_INT64_T, MPI_SUM,
                  MPI_COMM_WORLD);
  return bytes;
}

const char *method_name() {
  switch (ghext->num_rk_stages) {
  case 3:
    return "SSPRK3";
  case 4:
    return "RK4";
  default:
    return "unknown method";
  }
}

std::string format_mib(const std::int64_t bytes) {
  const double mib = double(bytes) / (1024.0 * 1024.0);
  char buf[100];
  if (mib >= 10)
    std::snprintf(buf, sizeof buf, "%.0f MiB", mib);
  else
    std::snprintf(buf, sizeof buf, "%.2f MiB", mib);
  return buf;
}

} // namespace

namespace subcycling_tally_detail {

void charge_stream_wait_slow() {
#pragma omp critical(CarpetX_subcycling_tally)
  if (scope_t *const scope = charged_scope())
    ++scope->counts.stream_waits;
}

void charge_device_wait_slow() {
#pragma omp critical(CarpetX_subcycling_tally)
  if (scope_t *const scope = charged_scope())
    ++scope->counts.device_waits;
}

void charge_temp_buffer_slow() {
#pragma omp critical(CarpetX_subcycling_tally)
  if (scope_t *const scope = charged_scope())
    ++scope->counts.temp_buffers;
}

} // namespace subcycling_tally_detail

TallyScope::TallyScope(const scope_kind_t kind, const int level)
    : active(tally_enabled) {
  if (!active)
    return;
  assert(kind == scope_kind_t::excluded || level >= 0);
#pragma omp critical(CarpetX_subcycling_tally)
  scope_stack.push_back(scope_t{kind, level, call_counts_t()});
}

TallyScope::~TallyScope() {
  if (!active)
    return;
#pragma omp critical(CarpetX_subcycling_tally)
  {
    assert(!scope_stack.empty());
    const scope_t scope = scope_stack.back();
    scope_stack.pop_back();

    using std::max;
    switch (scope.kind) {
    case scope_kind_t::solver: {
      level_tally_t &tally = level_tally(scope.level);
      ++tally.solver_calls;
      call_counts_t &worst = tally.worst_solver_call;
      worst.stream_waits = max(worst.stream_waits, scope.counts.stream_waits);
      worst.device_waits = max(worst.device_waits, scope.counts.device_waits);
      worst.temp_buffers = max(worst.temp_buffers, scope.counts.temp_buffers);
      break;
    }
    case scope_kind_t::ghost_sync_exchange: {
      level_tally_t &tally = level_tally(scope.level);
      ++tally.exchange_syncs;
      tally.worst_waits_per_exchange_sync =
          max(tally.worst_waits_per_exchange_sync,
              scope.counts.stream_waits + scope.counts.device_waits);
      break;
    }
    case scope_kind_t::ghost_sync_prolongate: {
      // A prolongating SYNC keeps its three phases; it is counted but not
      // reported
      ++level_tally(scope.level).prolongate_syncs;
      break;
    }
    case scope_kind_t::excluded:
      break;
    }
  }
}

void charge_launches(const int n) {
  if (!tally_enabled)
    return;
#pragma omp critical(CarpetX_subcycling_tally)
  if (const scope_t *const scope = charged_scope()) {
    using std::max;
    level_tally_t &tally = level_tally(scope->level);
    tally.max_launches_per_lincomb = max(tally.max_launches_per_lincomb, n);
  }
}

void init_tally() {
  DECLARE_CCTK_PARAMETERS;

  tally_enabled = out_subcycling_counts;
  if (!tally_enabled)
    return;

  scope_stack.clear();
  level_tallies.clear();
  regrid_flagged = false;

  if (CCTK_MyProc(nullptr) == 0) {
    std::ostringstream buf;
    buf << out_dir << "/subcycling-counts.tsv";
    const std::string filename = buf.str();
    counts_file.open(filename);
    if (!counts_file)
      CCTK_VERROR("Could not open \"%s\" for writing", filename.c_str());
    counts_file << "# subcycling-counts.tsv\n"
                << "# 1:coarse_step\t2:level\t3:solver_calls"
                << "\t4:stream_waits_per_call\t5:device_waits_per_call"
                << "\t6:temp_buffers_per_call\t7:launches_per_lincomb"
                << "\t8:waits_per_ghost_sync\t9:regrid\t10:buffer_bytes\n"
                << std::flush;
  }
}

void flag_regrid() {
  if (!tally_enabled)
    return;
  regrid_flagged = true;
}

void end_coarse_step() {
  if (!tally_enabled)
    return;
  assert(scope_stack.empty());

  const int nlevels = ghext->num_levels();
  const rat64 coarse_iteration =
      ghext->patchdata.at(0).leveldata.at(0).iteration;
  assert(coarse_iteration.den == 1);
  const std::int64_t coarse_step = coarse_iteration.num;

  // Collective
  const std::vector<std::int64_t> buffer_bytes = calc_buffer_bytes(nlevels);

  if (int(worst_rows.size()) < nlevels) {
    worst_rows.resize(nlevels);
    worst_steady_rows.resize(nlevels);
  }

  for (int level = 0; level < nlevels; ++level) {
    const level_tally_t &tally = level_tally(level);
    row_t row;
    row.solver_calls = tally.solver_calls;
    row.per_call = tally.worst_solver_call;
    row.launches_per_lincomb = tally.max_launches_per_lincomb;
    row.waits_per_ghost_sync = tally.worst_waits_per_exchange_sync;
    row.buffer_bytes = buffer_bytes.at(level);

    if (counts_file.is_open())
      counts_file << coarse_step << "\t" << level << "\t" << row.solver_calls
                  << "\t" << row.per_call.stream_waits << "\t"
                  << row.per_call.device_waits << "\t"
                  << row.per_call.temp_buffers << "\t"
                  << row.launches_per_lincomb << "\t"
                  << row.waits_per_ghost_sync << "\t" << int(regrid_flagged)
                  << "\t" << row.buffer_bytes << "\n";

    worst_rows.at(level).fold(row);
    if (!regrid_flagged)
      worst_steady_rows.at(level).fold(row);
  }
  if (counts_file.is_open())
    counts_file << std::flush;

  ++num_steps;
  if (!regrid_flagged)
    ++num_steady_steps;

  // Reset the per-level state. Levels that no longer exist are dropped.
  level_tallies.clear();
  regrid_flagged = false;
}

void print_summary() {
  DECLARE_CCTK_PARAMETERS;

  if (!tally_enabled)
    return;

  if (counts_file.is_open())
    counts_file.close();

  if (num_steps == 0) {
    CCTK_VINFO("Subcycling path counts: no coarse step was completed");
    return;
  }

  // Prefer steps without a regrid: buffers are legitimately rebuilt there
  const bool have_steady = num_steady_steps > 0;
  const std::vector<row_t> &rows = have_steady ? worst_steady_rows : worst_rows;
  assert(ghext);
  const int nlevels = ghext->num_levels();

  int num_evolved_groups = 0;
  for (const bool is_integrated : ghext->rk_integrated_group)
    num_evolved_groups += is_integrated;

  std::ostringstream buf;
  buf << "Subcycling path counts, worst "
      << (have_steady ? "steady-state coarse step"
                      : "coarse step (every coarse step included a regrid)")
      << " (" << nlevels << " levels, " << method_name()
      << ", G=" << num_evolved_groups << "):\n";
  buf << "  level  solver calls  stream waits/call  device waits/call  "
         "temp buffers/call  launches/lincomb  waits/ghost sync\n";
  for (int level = 0; level < std::min(nlevels, int(rows.size())); ++level) {
    const row_t &row = rows.at(level);
    char line[200];
    std::snprintf(line, sizeof line,
                  "  %-5d  %-12d  %-17d  %-17d  %-17d  %-16d  %d\n", level,
                  row.solver_calls, row.per_call.stream_waits,
                  row.per_call.device_waits, row.per_call.temp_buffers,
                  row.launches_per_lincomb, row.waits_per_ghost_sync);
    buf << line;
  }
  buf << "  subcycling buffers held (bands + fill buffers):";
  for (int level = 0; level < std::min(nlevels, int(rows.size())); ++level)
    buf << (level == 0 ? " " : ", ") << "level " << level << ": "
        << format_mib(rows.at(level).buffer_bytes);
  if (poison_undefined_values)
    buf << "\n  note: poison_undefined_values = yes - validity/poison/checksum "
           "waits are NOT included above;\n"
           "        set it to \"no\" for production GPU runs.";

  CCTK_VINFO("%s", buf.str().c_str());
}

} // namespace CarpetX
