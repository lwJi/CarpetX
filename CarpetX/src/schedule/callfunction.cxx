#include "schedule/schedule.hxx"
#include "schedule/clauses.hxx"
#include "hierarchy/driver.hxx"
#include "hierarchy/loops.hxx"
#include "hierarchy/valid.hxx"
#include "comm/comm.hxx"
#include "traversal/traversal.hxx"
#include <carpetx.hxx>

#include <cctk.h>
#include <cctk_Parameters.h>
#include <cctk_Schedule.h>

#include <cassert>
#include <map>
#include <set>
#include <string>
#include <tuple>
#include <unordered_set>
#include <vector>

namespace CarpetX {

int CallFunction(void *function, cFunctionData *restrict attribute,
                 void *data) {
  DECLARE_CCTK_PARAMETERS;

  assert(function);
  assert(attribute);
  assert(data);

  cGH *restrict const cctkGH = static_cast<cGH *>(data);

  if (verbose)
#pragma omp critical
    CCTK_VINFO("CallFunction iteration %d %s: %s::%s", cctkGH->cctk_iteration,
               attribute->where, attribute->thorn, attribute->routine);

  const routine_scope_t routine_scope(attribute);

  const std::string routine =
      std::string(attribute->thorn) + "::" + attribute->routine;
  const std::string event =
      std::string("CallFunction ") + attribute->where + ": " + routine;

  static std::map<cFunctionData *restrict, Timer> timers;

  std::map<cFunctionData *restrict, Timer>::iterator timer_iter;
#pragma omp critical(CarpetX_CallFunction)
  {
    timer_iter = timers.find(attribute);
    if (timer_iter == timers.end())
      timer_iter = std::get<0>(timers.emplace(attribute, event));
  }
  Timer &timer = timer_iter->second;
  Interval interval(timer);

  const window_t window = scheduled_window(routine.c_str());

  {
    static std::unordered_set<cFunctionData *> validated;
    cFunctionData *const key = attribute;
    bool needs_check;
#pragma omp critical(CarpetX_CallFunction)
    {
      needs_check = validated.insert(key).second;
    }
    if (needs_check)
      check_storage_clauses(attribute);
  }

  if (CCTK_EQUALS(presync_mode, "presync-only")) {
    const std::vector<clause_t> &reads =
        decode_clauses(attribute, rdwr_t::read);
    std::set<gi_t> sync_set;
    for (const auto &rd : reads) {
      if (CCTK_GroupTypeI(rd.gi.index()) == CCTK_GF) {

        loop_serially(window, [&](const auto &restrict leveldata) {
          const valid_t &need = rd.valid;
          const valid_t have =
              leveldata.group(rd.gi).timelevel(rd.tl).valid(rd.vi);
          if (need.valid_ghosts && !have.valid_ghosts && have.valid_int)
            sync_set.insert(rd.gi);
        });
      }
    }
    if (!sync_set.empty()) {
      const std::vector<gi_t> sync_vec(sync_set.begin(), sync_set.end());
      SyncGroupsByDirI(window, sync_vec);
    }
  }

  {
    std::set<gi_t> touched;
    for (const rdwr_t rdwr : {rdwr_t::read, rdwr_t::write})
      for (const auto &clause : decode_clauses(attribute, rdwr))
        touched.insert(clause.gi);
    for (int n = 0; n < attribute->n_SyncGroups; ++n)
      touched.insert(gi_t{attribute->SyncGroups[n]});
    for (const gi_t gi : touched) {
      const auto &group = driver().record(gi);
      if (group.kind != group_kind::grid_function || !group.has_storage())
        continue;
      loop_serially(window, [&](const auto &restrict leveldata) {
        const auto &restrict groupdata = leveldata.group(gi);
        const int open_writes = groupdata.open_write_count();
        if (open_writes > 0)
          CCTK_VERROR("CallFunction iteration %d %s: %s::%s reads, writes or "
                      "syncs group %s, which has %d open write scope%s "
                      "(CarpetX::write) on patch %d, refinement level %d; "
                      "destroy the write_t before scheduling the function",
                      cctkGH->cctk_iteration, attribute->where,
                      attribute->thorn, attribute->routine,
                      groupdata.record.groupname.c_str(), open_writes,
                      open_writes == 1 ? "" : "s",
                      groupdata.level.patch().index(), groupdata.level.index());
      });
    }
  }

  {
    const std::vector<clause_t> &reads =
        decode_clauses(attribute, rdwr_t::read);
    for (const auto &rd : reads) {
      if (!driver().record(rd.gi).has_storage())
        continue;
      const require_spec_t spec{.gi = rd.gi,
                                .tl = rd.tl,
                                .vi = rd.vi,
                                .where = rd.valid,
                                .event = event + " checking input"};
      if (CCTK_GroupTypeI(rd.gi.index()) == CCTK_GF)
        require(window, spec);
      else
        require_array(spec);
    }
  }

  std::vector<write_scope_t> write_scopes;
  std::vector<array_write_scope_t> array_write_scopes;
  {
    std::map<clause_t, valid_t> isread;
    const std::vector<clause_t> &reads =
        decode_clauses(attribute, rdwr_t::read);
    for (const auto &rd : reads) {
      clause_t cl = rd;
      cl.valid = valid_t();
      assert(isread.count(cl) == 0);
      isread[cl] = rd.valid;
    }
    const std::vector<clause_t> &writes =
        decode_clauses(attribute, rdwr_t::write);
    for (const auto &wr : writes) {
      const GHExt::GroupRecord &record = driver().record(wr.gi);
      if (!record.has_storage())
        continue;
      if (!record.timelevels().contains(wr.tl))
        CCTK_VERROR("Accessing non-existent timelevel %d of variable %s",
                    wr.tl.index(), CCTK_FullVarName(record.var(wr.vi).index()));
      clause_t cl = wr;
      cl.valid = valid_t();
      valid_t need;
      if (isread.count(cl) > 0)
        need = isread[cl];
      const write_spec_t spec{.gi = wr.gi,
                              .tl = wr.tl,
                              .vi = wr.vi,
                              .fills = wr.valid,
                              .keeps = ~(wr.valid & ~need),
                              .event = event};
      if (CCTK_GroupTypeI(wr.gi.index()) == CCTK_GF)
        write_scopes.push_back(open_write(window, spec));
      else
        array_write_scopes.push_back(open_array_write(spec));
    }
  }

  checksums_t checksums;
  if (poison_undefined_values) {
    const std::vector<clause_t> &writes =
        decode_clauses(attribute, rdwr_t::write);
    const int numgroups = CCTK_NumGroups();
    std::vector<std::vector<std::vector<valid_t> > > gfs(numgroups);
    for (int gi = 0; gi < numgroups; ++gi) {
      const int numvars = CCTK_NumVarsInGroupI(gi);
      gfs.at(gi).resize(numvars);
      for (int vi = 0; vi < numvars; ++vi) {
        const int numtimelevels = 1;
        gfs.at(gi).at(vi).resize(numtimelevels);
      }
    }
    for (const auto &wr : writes) {
      if (wr.tl.index() >= int(gfs.at(wr.gi.index()).at(wr.vi.index()).size()))
        gfs.at(wr.gi.index()).at(wr.vi.index()).resize(wr.tl.index() + 1);
      gfs.at(wr.gi.index()).at(wr.vi.index()).at(wr.tl.index()) |= wr.valid;
    }

    checksums = calculate_checksums(window, gfs);
  }

  const mode_t mode = decode_mode(attribute);
  switch (mode) {
  case mode_t::local:
    loop_parallel(window, [&](const GHExt::PatchData::LevelData &leveldata,
                              int index, int component,
                              const cGH *local_cctkGH) {
      update_cctkGH(const_cast<cGH *>(local_cctkGH), cctkGH);
      CCTK_CallFunction(function, attribute, const_cast<cGH *>(local_cctkGH));
    });
    synchronize();
    break;

  case mode_t::meta:
  case mode_t::global:
  case mode_t::level:
    CCTK_CallFunction(function, attribute, cctkGH);
    break;

  default:
    assert(0);
  }

  if (poison_undefined_values)
    check_checksums(window, checksums,
                    [event, iteration = cctkGH->cctk_iteration]() {
                      return event + " checking output (iteration " +
                             std::to_string(iteration) + ")";
                    });

  write_scopes.clear();
  array_write_scopes.clear();

  {
    const std::vector<clause_t> &invalids =
        decode_clauses(attribute, rdwr_t::invalid);
    for (const auto &inv : invalids) {
      const invalidate_spec_t spec{.gi = inv.gi,
                                   .tl = inv.tl,
                                   .vi = inv.vi,
                                   .where = inv.valid,
                                   .event = event};
      if (CCTK_GroupTypeI(inv.gi.index()) == CCTK_GF)
        invalidate(window, spec);
      else
        invalidate_array(spec);
    }
  }

  constexpr int didsync = 0;
  return didsync;
}

} // namespace CarpetX
