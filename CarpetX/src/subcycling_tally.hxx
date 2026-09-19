#ifndef CARPETX_CARPETX_SUBCYCLING_TALLY_HXX
#define CARPETX_CARPETX_SUBCYCLING_TALLY_HXX

// Counter report for the subcycling path (`CarpetX::out_subcycling_counts`).
//
// The tally counts *logical* events -- device waits, temporary buffers, kernel
// launches -- at the chokepoints every such event on the path passes through
// (`synchronize()`, `synchronize_device()`, `make_temp_mfab()`), so a CPU
// build reports the numbers a GPU build would. Whom an event is charged to is
// decided by an RAII scope stack, innermost scope wins:
//
//   ODESolvers_Solve_Subcycling          TallyScope(solver, level)
//     CallScheduleGroup(RHS | PostStep)  TallyScope(excluded)
//       SyncGroupsByDirISubcycling       TallyScope(ghost_sync_*, level)
//     poison_invalid_gf / check_valid_gf TallyScope(excluded)
//
// Events outside any scope, or under an `excluded` scope, are dropped. With
// the report off, a charge costs one branch on a global flag.
//
// The state is keyed by level index (not held in LevelData) so that it
// survives a regrid inside a coarse step. All entry points must be called from
// serial host code (not from inside an OpenMP parallel region).

namespace CarpetX {

enum class scope_kind_t {
  solver,                // one ODESolvers_Solve_Subcycling call on one level
  ghost_sync_exchange,   // a SYNC that only exchanges same-level ghosts
  ghost_sync_prolongate, // a SYNC that prolongates at least one group
  excluded               // thorn routines, validity tracking: not charged
};

namespace subcycling_tally_detail {
extern bool tally_enabled;
void charge_stream_wait_slow();
void charge_device_wait_slow();
void charge_temp_buffer_slow();
} // namespace subcycling_tally_detail

// RAII charge target. `level` is ignored for `excluded`.
struct TallyScope {
  explicit TallyScope(scope_kind_t kind, int level = -1);
  ~TallyScope();
  TallyScope(const TallyScope &) = delete;
  TallyScope &operator=(const TallyScope &) = delete;

private:
  bool active;
};

// An all-stream wait (`synchronize()`)
inline void charge_stream_wait() {
  if (subcycling_tally_detail::tally_enabled)
    subcycling_tally_detail::charge_stream_wait_slow();
}
// A full-device wait (`synchronize_device()`)
inline void charge_device_wait() {
  if (subcycling_tally_detail::tally_enabled)
    subcycling_tally_detail::charge_device_wait_slow();
}
// A temporary MultiFab (`make_temp_mfab()`)
inline void charge_temp_buffer() {
  if (subcycling_tally_detail::tally_enabled)
    subcycling_tally_detail::charge_temp_buffer_slow();
}
// The kernel launches of one RK linear combination; the report keeps the
// maximum per level
void charge_launches(int n);

// Read `out_subcycling_counts`; if set, enable the tally and (on rank 0) open
// `<out_dir>/subcycling-counts.tsv`. Called once at the start of `Evolve`.
void init_tally();
// A regrid changed the grid structure: flag the current coarse step.
void flag_regrid();
// Level 0 has stepped and all levels are aligned again: append one row per
// level and reset the per-level state. Collective over all processes.
void end_coarse_step();
// Print the worst steady-state coarse step per level. Called at shutdown.
void print_summary();

} // namespace CarpetX

#endif // #ifndef CARPETX_CARPETX_SUBCYCLING_TALLY_HXX
