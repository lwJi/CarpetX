# CarpetX's public API

`python3 scripts/lint doc --write` writes this file from `scripts/lint/boundary/guide.py`. A row says only what a header cannot: a signature is read in the header.

A thorn reaches CarpetX through `carpetx.hxx`, and the time integrator also through `carpetx_rk.hxx`, whose contract is in `agent_docs/carpetx-rk-api.md`. `TestCarpetXAPI` is a client of both: its `interface.ccl` and `configuration.ccl` are what a client declares, `TestCarpetXAPI/src/` is a worked use of the calls below, and each parfile of `TestCarpetXAPI/test/errors/` is a call CarpetX refuses, with the text it aborts with in the `.expect` file beside it. Every call is available between the driver's SetupGH and its Shutdown, and a bad argument or a stale handle is a `CCTK_VERROR`.

## Indices and handles

`vi_t` is a variable's place in its group and `var_t` the flesh's variable index. `window_t` is a half-open range of levels on a half-open range of patches: `levels` spans every patch, `narrow` keeps the patches of its window, and `coarsest` and `finest` are on its first patch. `component_t` is a box of a level, AMReX's global box index and the flesh's `cctk_component`; `tile_t` is one tile of the tiled traversal over this rank's boxes and carries the `component_t` it lies in. `loop_parallel` hands its kernel the `level_t`, the `tile_t` and the local cGH.

## Data, validity and the write scope

A write scope sets the validity of the group at `tl` to exactly `leaves_valid` when it opens and poisons the regions it does not promise, so the scope must not read them; when it closes, the promised regions are scanned for poison and NaN. While it is open, a communication operation, `invalidate` or a scheduled function touching the group is refused. `require_valid` also runs that scan over `where`. A validity bit is set only by a write scope or a communication operation.

The halo has two ghost regions with a bit each: the same-level ghosts, which another box of the level or a periodic image covers, and the coarse-fine ghosts, which no box of the level covers and only an interpolation from the parent fills. A ghost fill marks only the region it copies; the boundary condition reads the interior and both ghost regions, so it runs after the last ghost fill, requires all three valid, and alone marks the outer boundary. `make_valid_ghosts()` and a schedule clause's everywhere name both.

`scratch(group)` is the interior of `data(group, tl_t{0})`, with its boxes, distribution, centring and variables and no ghost zones, and has no validity bits; a regrid frees it, after which a `scratch_t` is an error.

## Communication

An operation without a `tl` acts on every time level but the oldest. `sync` is the scheduled SYNC, subcycling under `use_subcycling`, and sets the same-level, coarse-fine and outer bits it fills; `sync_ghosts_only` is its same-level copy alone and sets the same-level bit, and `prolongate_ghosts` its coarse-to-fine interpolation alone and sets the coarse-fine bit, neither applying the boundary condition; `prolongate_aligned` is `prolongate_ghosts` that under subcycling fills only levels time-aligned with their parent. `apply_outer_boundary` is the boundary condition alone: it requires the interior and both ghost regions, writes the outer region and sets the outer bit, so a thorn that has written a group everywhere and wants the driver's boundary condition on it calls this and no ghost fill. `restrict_no_poison` leaves the coarse ghost and outer regions as they were, where the scheduled restriction invalidates and poisons them. `synchronize` waits for the device.

## The level hierarchy and the parameters

The BoxArray and DistributionMapping of a level are those of any group's MultiFab on it, and the refinement ratio is 2. Driver parameters are read through the flesh, `SHARES: Driver` or `CCTK_ParameterGet`.

## The clock a routine sees

`cctk_iteration`, `cctk_time`, `cctk_delta_time` and `cctk_timefac` are the run clock that CarpetX holds, and every cGH shows it. A thorn reads them and never assigns them: when a scheduled function returns, a cGH that disagrees with the run clock stops the run.
