# CarpetX's public API

`python3 scripts/lint doc --write` writes this file from `scripts/lint/boundary/guide.py`, and the edit hook refuses an edit here. A row says only what a header cannot: a signature is read in the header, and a refusal in its parfile. Each paragraph ends with the functions of `CarpetX/src/` that hold it.

A thorn reaches CarpetX through the two headers `CarpetX/interface.ccl` exports: `carpetx.hxx`, described here, and for the time integrator `carpetx_rk.hxx`, described in `agent_docs/carpetx-rk-api.md`. Every other header under `CarpetX/src/` refuses a compilation that is not CarpetX's own, and `agent_docs/carpetx-internals.md` maps the inside. Every call is available between the driver's SetupGH and its Shutdown, and a bad argument is a `CCTK_VERROR`, never garbage.

A refusal named under a section is a parfile of `TestCarpetXAPI/test/errors/`, and its `.expect` file is the text CarpetX aborts with.

`TestCarpetXAPI/src/` exercises every name below but `restrict_no_poison`, `scratch`.

## Indices and handles

`vi_t` is a variable's place in its group and `var_t` the flesh's variable index. A handle is made of indices and resolved on every call, so a stale one is reported, never dereferenced. `window_t` is a half-open range of levels on a half-open range of patches: `levels` spans every patch, `narrow` keeps the patches of its window, and `coarsest` and `finest` are on its first patch. A window checks its bounds when it is built, so the error names the line that built it. `component_t` is a box of a level, AMReX's global box index and the flesh's `cctk_component`; `tile_t` is one tile of the tiled traversal over this rank's boxes and carries the `component_t` it lies in, so `loop_parallel` hands its kernel the `level_t`, the `tile_t` and the local cGH, and a tile is looked up by its slot and checked against its box. Held by `variable`, `levels`, `narrow`, `coarsest`, `finest`, `require_window_bounds`, `loop_parallel` in `hierarchy/handles.cxx`; `level`, `get_local_cctkGH` in `hierarchy/model.hxx`; `tile_of` in `config/indices.hxx`.

Refusals: `level_missing`, `timelevel_missing`, `parent_of_level_0`, `window_empty`, `window_out_of_bounds`, `window_outside_traversal`, `hierarchy_absent`.

## Data, validity and the write scope

A write scope sets the validity of the group at `tl` to exactly `leaves_valid` when it opens and poisons the regions it does not promise, so the scope must not read them; when it closes, the promised regions are scanned for poison and NaN. While it is open, a communication operation, `invalidate` or a scheduled function touching the group is refused. `require_valid` also runs that scan over `where`. A validity bit is set only by a write scope or a communication operation. Held by `write`, `require_valid`, `invalidate`, `require_no_open_write` in `api/api.cxx`; `write_scope_t` in `hierarchy/scopes.cxx`; `CallFunction` in `schedule/callfunction.cxx`.

The halo has two ghost regions with a bit each: the same-level ghosts, which another box of the level or a periodic image covers, and the coarse-fine ghosts, which no box of the level covers and only an interpolation from the parent fills. A ghost fill marks only the region it copies; the boundary condition reads the interior and both ghost regions, so it runs after the last ghost fill, requires all three valid, and alone marks the outer boundary. `make_valid_ghosts()` and a schedule clause's everywhere name both. Held by `make_valid_ghosts` in `include/carpetx.hxx`; `decode_clauses` in `schedule/clauses.cxx`; `LevelData` in `hierarchy/model.cxx`; `loop_ghost_halves`, `check_valid_gf` in `hierarchy/scopes.cxx`; `apply_outer_boundary` in `comm/sync_restrict.cxx`.

`scratch(group)` is shaped like `data(group, tl_t{0})` and comes from a per-group pool its destructor returns it to; the pool dies with its level on a regrid, after which a `scratch_t` is an error, not a dangling pointer. It has no validity bits. Held by `scratch`, `resolve_scratch`, `scratch_t` in `api/api.cxx`.

Refusals: `write_twice`, `write_unfilled`, `require_invalid`, `invalidate_during_write`, `schedule_during_write`.

## Communication

An operation without a `tl` acts on every time level but the oldest. `sync` is the scheduled SYNC, subcycling under `use_subcycling`, and sets the same-level, coarse-fine and outer bits it fills; `sync_ghosts_only` is its same-level copy alone and sets the same-level bit, and `prolongate_ghosts` its coarse-to-fine interpolation alone and sets the coarse-fine bit, neither applying the boundary condition; `prolongate_aligned` is `prolongate_ghosts` that under subcycling fills only levels time-aligned with their parent. `apply_outer_boundary` is the boundary condition alone: it requires the interior and both ghost regions, writes the outer region and sets the outer bit, so a thorn that has written a group everywhere and wants the driver's boundary condition on it calls this and no ghost fill. `restrict_no_poison` leaves the coarse ghost and outer regions as they were, where the scheduled restriction invalidates and poisons them. `synchronize` waits for the device. Held by `sync`, `sync_ghosts_only`, `prolongate_ghosts`, `prolongate_aligned`, `apply_outer_boundary`, `restrict_no_poison` in `api/api.cxx`; `SyncGhostsOnly`, `ProlongateGhosts_impl`, `ApplyOuterBoundary`, `Restrict`, `RestrictNoPoison` in `comm/sync_restrict.cxx`; `synchronize` in `hierarchy/loops.cxx`.

Refusals: `sync_during_write`, `prolongate_unwritten_parent`, `sync_invalid_coarse`, `boundary_before_sync`.

## The level hierarchy and the parameters

The BoxArray and DistributionMapping of a level are those of any group's MultiFab on it, and the refinement ratio is 2. Driver parameters are read through the flesh, `SHARES: Driver` or `CCTK_ParameterGet`, and CarpetX checks its own. Held by `PatchData`, `GroupData` in `hierarchy/model.cxx`; `per_level_max_grid_sizes` in `config/grid_sizes.cxx`; `select_groups` in `io/selection.cxx`.

Refusals: `too_many_levels`, `output_unknown_variable`.

## The clock a routine sees

`cctk_iteration`, `cctk_time`, `cctk_delta_time` and `cctk_timefac` are the run clock that CarpetX holds, with the time and step of a `run_schedule_group_at` in place of the run's for the length of its traversal, and every cGH shows it: the flesh's, and each level, patch and tile copy. A thorn reads them and never assigns them; when a scheduled function returns, CarpetX compares the cGH it was handed, the flesh's or for a local-mode function each tile copy, with the run clock in every build, and a cGH that disagrees stops the run, naming the function, the field, the cGH's value and the run clock's. Held by `publish_clock`, `require_current_clock` in `hierarchy/clock.cxx`; `CallFunction` in `schedule/callfunction.cxx`.

Refusals: `clock_stale`.
