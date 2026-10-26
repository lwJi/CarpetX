# CarpetX's public API

A thorn reaches CarpetX through two headers that `CarpetX/interface.ccl` exports from `CarpetX/src/include/`: `carpetx.hxx` for everything a general thorn needs, and `carpetx_rk.hxx` for the time integrator's contract. Every other header under `CarpetX/src/` is private: it includes `internal.hxx`, which stops with `#error` unless `CARPETX_INTERNAL` is defined, and only CarpetX's own `src/make.code.defn` defines it. The public headers include nothing under `CarpetX/src/`, so a client cannot reach the driver's internals transitively either.

`python3 scripts/boundary check` lists every `#include` outside `CarpetX/src/` that reaches into `CarpetX/src`, every `CARPETX_INTERNAL` outside CarpetX's `make.code.defn` and the private headers' guard, and every `directions` parameter in the sync layer outside the one flesh-facing overload; the `Comments` workflow runs it with `--since` on every PR. The data model behind the headers (`GHExt`, `LevelData`, `GroupData`, `active_levels`, the `FillPatch_*` pipelines, the validity bookkeeping) has no public form: the public counterpart of a private name is the operation CarpetX performs with it.

`TestCarpetXAPI` is a client of both headers, and `TestCarpetXAPI/test/errors/` holds one expected-failure parfile per refusal named below.

## Using the headers

In `interface.ccl`, `USES INCLUDE HEADER: carpetx.hxx` (and `carpetx_rk.hxx` for an integrator); in `configuration.ccl`, `REQUIRES CarpetX`, which also provides the AMReX, Loop and Arith headers that `carpetx.hxx` includes; in C++, `#include <carpetx.hxx>` and qualify every name with `CarpetX::`. The headers add no using-directive at any scope. `Loop::GridDescBase` is a Loop type and comes from `<loop.hxx>`, not from CarpetX.

The header is the list of calls; this file states only what a signature cannot. Every call is available between the driver's SetupGH and its Shutdown, and a bad argument is a `CCTK_VERROR`, never garbage.

## Indices and handles

An index is a type: `gi_t` names a Cactus group and `tl_t` a time level. Each is built explicitly from the flesh's `int`, `gi_t{CCTK_GroupIndex("…")}` or `tl_t{0}`, and read back only through `index()`, where a value leaves for the flesh (`CCTK_FullGroupName(gi.index())`). An index does not convert to an `int` or from one implicitly, nor to another kind of index, and it has no arithmetic; two of a kind compare equal and order. The one named step is `tl_t::next()`, the following time level, which is refused where it is used if the group has no such level. `TestCarpetXAPI/src/compile_checks.cxx` holds the `static_assert`s that pin this.

A handle is made of indices and resolved on every call, so a stale one names a level that no longer exists and the call reports it instead of touching freed memory. `level_t{patch, level}` names a refinement level of a patch, `group_t{level, gi}` a grid-function group on it (`gi` is a `gi_t`), and `window_t{min_level, max_level, min_patch, max_patch}` a half-open set of levels, both for traversal (`loop_serially`, `loop_coarse_to_fine`, and `loop_parallel`, whose kernel receives the component's cGH) and as the "which levels" argument of every operation. `active_window()` is the window of the current traversal and an error outside one; `all_levels()` is the whole hierarchy. `parent(level)` is an error on level 0, and `num_levels(patch)` is per patch.

## Data, validity and the write scope

`data(group, tl)` is the `const amrex::MultiFab &` of a time level; AMReX's const overloads give `Array4<const T>`. Writing goes through a scope: `auto w = write(group, tl, leaves_valid, "what"); w.mfab()`. When it opens, every variable of the group at `tl` gets exactly `leaves_valid` as its validity, with reason `"what (iteration N)"`, and the regions it does not promise are poisoned, so the scope must not read them. When it is destroyed, the promised regions are scanned for poison and NaN, as a scheduled function's WRITES are. While it is open, a second `write` on the same group and time level, any communication operation or `invalidate` on the group, and scheduling a function that reads, writes or syncs the group are all errors.

`valid(group, tl, vi)` reads the bits. `require_valid(group, tl, where, "why")` raises an error with CarpetX's explanation if any variable of the group lacks a bit of `where`, and runs the poison and NaN scan. `invalidate(window, groups, tl, "why")` clears every bit of every variable at `tl`. No thorn outside CarpetX sets a validity bit: the write scope and the communication operations are the only way a bit is set.

`scratch(group)` is a move-only buffer with the boxes, distribution, components and ghosts of `data(group, tl_t{0})`, drawn from a per-group pool that CarpetX keeps between calls. There is no init or free call; the destructor returns the buffer. The pool dies with the level on a regrid, and using a `scratch_t` after that is an error, not a dangling pointer. It has no validity bits.

## Communication

Each operation takes a window and a list of grid-function groups as a `std::span<const gi_t>`, so a `std::vector<gi_t>` or a `std::array<gi_t, N>` converts in place (a braced list does not; write `std::array{gi}` for one group), and is an error while a write scope is open on any of them.

- `sync(window, groups)` is a scheduled SYNC: every time level but the oldest, the subcycling variant when `use_subcycling`, and it sets the ghost and outer bits it fills. Under subcycling the SYNC of an integrated group on a refined level at iteration > 0 copies only the same-level ghosts and requires the ghost bit set already, by `state_updated` or by the driver's refill after a recovery; finding it unset is an error. On level 0 it fills and marks every ghost itself.
- `sync_ghosts_only(window, groups)`: the same-level copy only, on every time level but the oldest; `sync_ghosts_only(window, groups, tl)` does that one time level.
- `prolongate_ghosts(window, groups)`: coarse-to-fine interpolation only, on every time level but the oldest; marks the ghost and outer bits of the levels it filled. `prolongate_ghosts(window, groups, tl)` does that one time level.
- `prolongate_aligned(window, groups)` and `prolongate_aligned(window, groups, tl)`: as `prolongate_ghosts`, but under subcycling only into levels time-aligned with their parent.
- `restrict_no_poison(window, groups)`: fine-to-coarse over every level pair in the window.
- `synchronize()` waits for the device.

`reduce(gi, vi, tl)` and `reduction<T, D>`, `Timer` and `Interval`, `rat64` (the type of `iteration(level)` and `delta_iteration(level)`), `geometry(level)`, `symmetries(level)`, `numvars`, `indextype` and `nghostzones` are what their names say. The BoxArray and DistributionMapping of a level are those of any group's MultiFab on it, and the refinement ratio is 2 between every pair of levels. Driver parameters are read through the flesh (`SHARES: Driver` in `param.ccl`, or `CCTK_ParameterGet`), not through the headers.

## The integrator's contract: `carpetx_rk.hxx`

Exactly one thorn, the time integrator, publishes what it integrates and reports the events of its RK step; CarpetX derives everything else.

Registration is once, `AT WRAGH`: `register_integrator(rk_config_t{dense_output, b, integrated})`. `dense_output` names the method whose dense output fills the refinement boundary under subcycling (`ssprk3`, `rk4` or `none`); it is named, not inferred from the stage count, so a 3-stage method other than SSPRK3 registers `none` and cannot run under subcycling. `b` holds the final-update weight of every stage, so `b.size()` is the stage count; an implicit method registers an empty `b` and never feeds a flux register, and all-zero weights are allowed for a method that never updates the state. `integrated` holds `rk_pair_t{state, rhs}` pairs of `gi_t`. CarpetX checks the record against its own parameters and refuses a bad one; the refusals are the parfiles in `TestCarpetXAPI/test/errors/`. A driver without a registration still runs; only `open_rk_step` needs one.

`run_schedule_group_at(cctkGH, "G", t, delta_time)` sets `cctk_time` and `cctk_delta_time` on the given cGH and on every driver cGH, traverses the group, and restores both. `delta_time` is the raw `cctk_delta_time` the group sees, so pass `cctkGH->cctk_delta_time` unless the step itself changes, as an IMEX half step does.

The step is `auto step = open_rk_step(window)`, opened once the state holds u(t_n) and outside any OpenMP parallel region. It stores the old state of every registered pair with storage on each level of the window and requires their interior valid. Then `rhs_computed()` and `state_updated()` alternate, `b.size()` times each, and the destructor checks that every stage ran. `rhs_computed` requires the interior of every registered right-hand side valid on every level of the window, stores the stage band, and feeds the flux registers with the stage's weight times that level's `CCTK_DELTA_TIME`. `state_updated` says the state now holds the next stage value: under subcycling CarpetX copies the same-level ghosts of every refined level in the window, fills its refinement boundary from its own clocks, scans the ghosts, and marks the integrated groups' ghosts valid, and their outer boundary where every face has a symmetry or boundary; without subcycling the last `state_updated` applies the flux registers. A call out of order, a call after the last stage, and a step destroyed early are errors naming the window and the stage; no stage index is passed, so stages cannot run out of order. After a recovery the driver refills the integrated groups' refinement boundary itself before it traverses `CCTK_POST_RECOVER_VARIABLES`; the integrator schedules nothing for it.
