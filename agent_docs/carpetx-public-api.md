# CarpetX's public API

`python3 scripts/lint doc --write` writes this file from `scripts/lint/boundary/guide.py`, and the edit hook refuses an edit here. A row says only what a header cannot: a signature is read in the header.

A thorn reaches CarpetX through the two headers `CarpetX/interface.ccl` exports from `CarpetX/src/include/`: `carpetx.hxx` for a general thorn, and `carpetx_rk.hxx` for the time integrator. Every other header under `CarpetX/src/` refuses a compilation that is not CarpetX's own, and `agent_docs/carpetx-internals.md` maps the inside.

`TestCarpetXAPI` is a client of both headers: its `interface.ccl` and `configuration.ccl` are what a client declares, `TestCarpetXAPI/src/` is a worked use of the calls below, and each parfile of `TestCarpetXAPI/test/errors/` is a call CarpetX refuses, with the text it aborts with in the `.expect` file beside it. Every call is available between the driver's SetupGH and its Shutdown, and a bad argument is a `CCTK_VERROR`, never garbage.

## Indices and handles

`vi_t` is a variable's place in its group and `var_t` the flesh's variable index. A handle is made of indices and resolved on every call, so a stale one is reported, never dereferenced. `window_t` is a half-open range of levels on a half-open range of patches: `levels` spans every patch, `narrow` keeps the patches of its window, and `coarsest` and `finest` are on its first patch. A window checks its bounds when it is built, so the error names the line that built it. `component_t` is a box of a level, AMReX's global box index and the flesh's `cctk_component`; `tile_t` is one tile of the tiled traversal over this rank's boxes and carries the `component_t` it lies in, so `loop_parallel` hands its kernel the `level_t`, the `tile_t` and the local cGH, and a tile is looked up by its slot and checked against its box.

## Data, validity and the write scope

A write scope sets the validity of the group at `tl` to exactly `leaves_valid` when it opens and poisons the regions it does not promise, so the scope must not read them; when it closes, the promised regions are scanned for poison and NaN. While it is open, a communication operation, `invalidate` or a scheduled function touching the group is refused. `require_valid` also runs that scan over `where`. A validity bit is set only by a write scope or a communication operation.

The halo has two ghost regions with a bit each: the same-level ghosts, which another box of the level or a periodic image covers, and the coarse-fine ghosts, which no box of the level covers and only an interpolation from the parent fills. A ghost fill marks only the region it copies; the boundary condition reads the interior and both ghost regions, so it runs after the last ghost fill, requires all three valid, and alone marks the outer boundary. `make_valid_ghosts()` and a schedule clause's everywhere name both.

`scratch(group)` is shaped like `data(group, tl_t{0})` and comes from a per-group pool its destructor returns it to; the pool dies with its level on a regrid, after which a `scratch_t` is an error, not a dangling pointer. It has no validity bits.

## Communication

An operation without a `tl` acts on every time level but the oldest. `sync` is the scheduled SYNC, subcycling under `use_subcycling`, and sets the same-level, coarse-fine and outer bits it fills; `sync_ghosts_only` is its same-level copy alone and sets the same-level bit, and `prolongate_ghosts` its coarse-to-fine interpolation alone and sets the coarse-fine bit, neither applying the boundary condition; `prolongate_aligned` is `prolongate_ghosts` that under subcycling fills only levels time-aligned with their parent. `apply_outer_boundary` is the boundary condition alone: it requires the interior and both ghost regions, writes the outer region and sets the outer bit, so a thorn that has written a group everywhere and wants the driver's boundary condition on it calls this and no ghost fill. `restrict_no_poison` leaves the coarse ghost and outer regions as they were, where the scheduled restriction invalidates and poisons them. `synchronize` waits for the device.

## The level hierarchy and the parameters

The BoxArray and DistributionMapping of a level are those of any group's MultiFab on it, and the refinement ratio is 2. Driver parameters are read through the flesh, `SHARES: Driver` or `CCTK_ParameterGet`, and CarpetX checks its own.

## The clock a routine sees

`cctk_iteration`, `cctk_time`, `cctk_delta_time` and `cctk_timefac` are the run clock that CarpetX holds, with the time and step of a `run_schedule_group_at` in place of the run's for the length of its traversal, and every cGH shows it: the flesh's, and each level, patch and tile copy. A thorn reads them and never assigns them; when a scheduled function returns, CarpetX compares the cGH it was handed, the flesh's or for a local-mode function each tile copy, with the run clock in every build, and a cGH that disagrees stops the run, naming the function, the field, the cGH's value and the run clock's.

## The integrator's contract: `carpetx_rk.hxx`

Exactly one thorn, the time integrator, publishes what it integrates and reports the events of its RK step; CarpetX derives everything else, and runs without a registration until `open_rk_step`. Registration is `AT WRAGH`, once; the record freezes when that traversal ends. `dense_output` names the method whose dense output fills the refinement boundary under subcycling; it is named, not inferred from the stage count, and `none` cannot run under subcycling. `b` holds the final-update weight of every stage, all zero for a method that never updates the state.

`run_schedule_group_at` sets `cctk_time` and `cctk_delta_time` on every cGH for the traversal and then restores both, whichever cGH is passed; pass `cctkGH->cctk_delta_time` unless the step itself changes, as an IMEX half step does.

`open_rk_step` is called once the state holds u(t_n), outside any OpenMP parallel region, and stores the old state of every registered pair with storage on each level of the window. `rhs_computed` stores the stage band and feeds the flux registers with the stage's weight times that level's `CCTK_DELTA_TIME`. `state_updated` says the state holds the next stage value: under subcycling CarpetX fills the refinement boundary of every refined level in the window from its own clocks, scans the coarse-fine ghosts it wrote and marks them valid; the same-level ghosts and the outer boundary of the integrated groups stay invalid until the thorn's SYNC in ODESolvers_PostStep, which exchanges the same-level ghosts and then applies the boundary condition. Without subcycling the last `state_updated` applies the flux registers.

Under subcycling the SYNC of an integrated group copies the same-level ghosts and marks them on every level; on a refined level at iteration > 0 it keeps the coarse-fine ghosts that `state_updated` or the driver's refill after a recovery wrote, and elsewhere it fills and marks every ghost itself; on every level it then applies the boundary condition, which requires the interior and both ghost regions, and marks the outer boundary. After a recovery the driver refills the integrated groups' refinement boundary before `CCTK_POST_RECOVER_VARIABLES`, and the thorn's SYNC there copies and marks the rest; the integrator schedules nothing for it.
