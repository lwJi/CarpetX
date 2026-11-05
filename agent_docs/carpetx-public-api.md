# CarpetX's public API

`python3 scripts/lint doc --write` writes this file from the rows of `scripts/lint/boundary/guide.py` and holds them to the public headers, to `TestCarpetXAPI/src/` and to the parfiles of `TestCarpetXAPI/test/errors/` and their modes in `TestCarpetXAPI/param.ccl`: a name a row sets in code must be in a header, a parfile must be named under one section, and a refusal's text is its expect file; `scripts/lint/boundary/test_rules.py` fails when the file differs from what it writes, and the edit hook refuses an edit to it. A row says only what a header cannot: a signature is read in the header, and a refusal in its parfile.

A thorn reaches CarpetX through the two headers `CarpetX/interface.ccl` exports from `CarpetX/src/include/`: `carpetx.hxx` for a general thorn, and `carpetx_rk.hxx` for the time integrator. Every other header under `CarpetX/src/` includes `internal.hxx`, which refuses a compilation that is not CarpetX's own, and the public headers include nothing under `CarpetX/src/`.

`python3 scripts/lint check --only boundary` holds this; `python3 scripts/lint check --help` prints what each of its rules holds, `python3 scripts/lint --help` says how the hooks and the Lint workflow run it, and `agent_docs/carpetx-internals.md` maps the layers inside `CarpetX/src/`. The data model behind the headers has no public form: the public counterpart of a private name is the operation CarpetX performs with it.

`TestCarpetXAPI` is a client of both headers: its `interface.ccl` and `configuration.ccl` are what a client declares, `TestCarpetXAPI/src/compile_checks.cxx` holds the static assertions that pin the types, `TestCarpetXAPI/src/api_test.cxx` exercises the calls, and `TestCarpetXAPI/test/errors/` holds a parfile for every refusal named below, each listed under its section with what the run does and the text it aborts with. Every call is available between the driver's SetupGH and its Shutdown, and a bad argument is a `CCTK_VERROR`, never garbage.

What this file says of a name TestCarpetXAPI never uses is held by nothing but the headers. Those names are: `restrict_no_poison`, `scratch`.

## Indices and handles

`vi_t` is a variable's place in its group and `var_t` the flesh's variable index. A handle is made of indices and resolved on every call, so a stale one names a level that no longer exists and the call reports it instead of touching freed memory. `window_t` is a half-open range of levels on a half-open range of patches: `levels` spans every patch, `narrow` keeps the patches of its window, and `coarsest` and `finest` are on its first patch. A window checks its bounds against the hierarchy when it is built, so a bad window is an error naming the line that built it, not the operation that later uses it.

`TestCarpetXAPI/src/compile_checks.cxx` and `TestCarpetXAPI_Fill` exercise this section. Its refusals, each a parfile of `TestCarpetXAPI/test/errors/`:

- `level_missing`: open a write scope on state through a handle naming refinement level 99 of patch 0, which does not exist; CarpetX aborts the run, with `CarpetX::write: CarpetX level 99 does not exist on patch 0: it has 1 levels`
- `timelevel_missing`: open a write scope on state at time level 7, which it does not have; CarpetX aborts the run, with `CarpetX::write: time level 7 of group TESTCARPETXAPI::STATE does not exist on patch 0, refinement level 0`
- `parent_of_level_0`: ask refinement level 0 of patch 0 for its parent level, which it does not have; CarpetX aborts the run, with `CarpetX level 0 of patch 0 has no parent level`
- `window_empty`: ask a window that holds no level, CarpetX::window_t::levels(0, 0), for its coarsest level; CarpetX aborts the run, with `CarpetX window of patches [0, 1), refinement levels [0, 0) holds no level`
- `window_out_of_bounds`: build a window of refinement levels [0, 2) on a hierarchy of 1 level; CarpetX aborts the run, with `CarpetX window levels [0, 2) must lie within [0, 1)`
- `window_outside_traversal`: call CarpetX::active_window at STARTUP, before the driver has opened any traversal; CarpetX aborts the run, with `CarpetX::active_window() called outside a traversal: no levels are active`
- `hierarchy_absent`: call CarpetX::num_levels at SHUTDOWN after Driver_Shutdown, once the driver has destroyed the grid hierarchy; CarpetX aborts the run, with `The CarpetX grid hierarchy does not exist`

## Data, validity and the write scope

A write scope sets the validity of every variable of the group at `tl` to exactly `leaves_valid` when it opens and poisons the regions it does not promise, so the scope must not read them; when it closes, the promised regions are scanned for poison and NaN. While it is open, another write scope on the same group and time level, a communication operation or `invalidate` on the group, and scheduling a function that reads, writes or syncs the group are refused. `require_valid` also runs the poison and NaN scan over `where`. A validity bit is set only by a write scope or a communication operation.

`scratch(group)` is a buffer with the boxes, distribution, components and ghosts of `data(group, tl_t{0})`, drawn from a per-group pool that its destructor returns it to. The pool dies with its level on a regrid, and a `scratch_t` used after that is an error, not a dangling pointer. It has no validity bits.

`TestCarpetXAPI_Fill` exercises this section. Its refusals, each a parfile of `TestCarpetXAPI/test/errors/`:

- `write_twice`: open a second write scope on state at the same time level while the first is open; CarpetX aborts the run, with `CarpetX::write: time level 0 of group TESTCARPETXAPI::STATE has 1 open write scope`
- `write_unfilled`: open a write scope on state that promises every region and fill only the interior; CarpetX aborts the run when the scope closes, with `Grid function "TESTCARPETXAPI::state" contains nans, infinities, or poison; expected valid`
- `require_invalid`: require valid ghosts of state before anything has filled them; CarpetX aborts the run, with `Grid function "TESTCARPETXAPI::state" is invalid on patch 0, refinement level 0, time level 0`
- `invalidate_during_write`: call CarpetX::invalidate on state while a write scope on it is open; CarpetX aborts the run, with `CarpetX::invalidate: group TESTCARPETXAPI::STATE has 1 open write scope`
- `schedule_during_write`: traverse TestCarpetXAPI_CheckGroup through CarpetX::run_schedule_group_at while a write scope on state is open; TestCarpetXAPI_Check in that group reads state, and CarpetX aborts the run, with `TestCarpetXAPI::TestCarpetXAPI_Check reads, writes or syncs group TESTCARPETXAPI::STATE, which has 1 open write scope`

## Communication

An operation without a `tl` acts on every time level but the oldest; with one, on that time level alone. `sync` is the scheduled SYNC, the subcycling variant under `use_subcycling`, and sets the ghost and outer bits it fills; `sync_ghosts_only` is its same-level copy alone, and `prolongate_ghosts` its coarse-to-fine interpolation alone, marking the ghost and outer bits of the levels it filled. `prolongate_aligned` is `prolongate_ghosts` that under subcycling fills only the levels time-aligned with their parent. `restrict_no_poison` restricts fine to coarse and leaves the coarse level's ghost and outer regions as they were, where the scheduled restriction invalidates and poisons them. `synchronize` waits for the device.

`TestCarpetXAPI_Fill` exercises this section. Its refusals, each a parfile of `TestCarpetXAPI/test/errors/`:

- `sync_during_write`: call CarpetX::sync on state while a write scope on it is open; CarpetX aborts the run, with `CarpetX::sync: group TESTCARPETXAPI::STATE has 1 open write scope`
- `prolongate_unwritten_parent`: fill state as write-scope does, and once the hierarchy has two levels call CarpetX::prolongate_ghosts on state_rhs, which nothing has written on the parent level; CarpetX aborts the run, with `Grid function "TESTCARPETXAPI::state_rhs" is invalid on patch 0, refinement level 0, time level 0`
- `sync_invalid_coarse`: once the hierarchy has two levels fill the interior of state on both, invalidate it on level 0, then call CarpetX::sync over both levels under CarpetX::use_subcycling without poisoning; the sync reads level 0 for its own ghosts and for the refinement boundary of level 1, and CarpetX aborts the run, with `SyncGroupsByDirISubcycling before syncing (iteration 0): Grid function "TESTCARPETXAPI::state" is invalid on patch 0, refinement level 0, time level 0`

## The hierarchy and the parameters

The BoxArray and DistributionMapping of a level are those of any group's MultiFab on it, and the refinement ratio is 2 between every pair of levels. Driver parameters are read through the flesh, `SHARES: Driver` or `CCTK_ParameterGet`, not through the headers, and CarpetX checks its own.

`TestCarpetXAPI_Fill` exercises this section. Its refusals, each a parfile of `TestCarpetXAPI/test/errors/`:

- `too_many_levels`: CarpetX aborts the run with `CarpetX::max_num_levels is 21, but CarpetX::max_grid_sizes_x, max_grid_sizes_y and max_grid_sizes_z have 20 entries`
- `output_unknown_variable`: CarpetX aborts the run with `CarpetX::OutputTSV: CarpetX::out_tsv_vars is "TestCarpetXAPI::nosuchvar", which names a group or variable that does not exist`

## The integrator's contract: `carpetx_rk.hxx`

Exactly one thorn, the time integrator, publishes what it integrates and reports the events of its RK step; CarpetX derives everything else. A driver without a registration still runs; only `open_rk_step` needs one.

Registration is `AT WRAGH`, once; the driver freezes the record when its WRAGH traversal ends. `dense_output` names the method whose dense output fills the refinement boundary under subcycling; it is named, not inferred from the stage count, and a method without one registers `none` and cannot run under subcycling. `b` holds the final-update weight of every stage; all-zero weights are allowed for a method that never updates the state.

`run_schedule_group_at` sets `cctk_time` and `cctk_delta_time` on the given cGH and on every driver cGH, traverses the group, and restores both; pass `cctkGH->cctk_delta_time` unless the step itself changes, as an IMEX half step does.

`open_rk_step` is called once the state holds u(t_n) and outside any OpenMP parallel region; it stores the old state of every registered pair with storage on each level of the window and requires their interior valid. `rhs_computed` and `state_updated` then alternate, `b.size()` times each. `rhs_computed` requires the interior of every registered right-hand side valid on every level of the window, stores the stage band, and feeds the flux registers with the stage's weight times that level's `CCTK_DELTA_TIME`. `state_updated` says the state now holds the next stage value: under subcycling CarpetX copies the same-level ghosts of every refined level in the window, fills its refinement boundary from its own clocks, scans the ghosts, and marks the integrated groups' ghosts valid, and their outer boundary where every face has a symmetry or boundary; without subcycling the last `state_updated` applies the flux registers.

Two promises CarpetX makes the integrator in return, which no parfile below holds. Under subcycling the SYNC of an integrated group on a refined level at iteration > 0 copies only the same-level ghosts and requires the ghost bit set already, by `state_updated` or by the driver's refill after a recovery; finding it unset is an error, and on level 0 the SYNC fills and marks every ghost itself. After a recovery the driver refills the integrated groups' refinement boundary itself before it traverses `CCTK_POST_RECOVER_VARIABLES`; the integrator schedules nothing for it.

`TestCarpetXAPI_Register` and `TestCarpetXAPI_Fill` exercise this section. Its refusals, each a parfile of `TestCarpetXAPI/test/errors/`:

- `register_at_startup`: register a valid integrator at STARTUP, before the driver has set up its grid hierarchy; CarpetX aborts the run, with `The CarpetX grid hierarchy does not exist`
- `register_at_paramcheck`: register a valid integrator at PARAMCHECK, after the driver has frozen its integrator record at the end of WRAGH; CarpetX aborts the run, with `CarpetX::register_integrator: the driver froze its integrator record after the WRAGH traversal`
- `register_after_use`: register a valid integrator at CCTK_INITIAL, after the driver has frozen its integrator record; CarpetX aborts the run, with `CarpetX::register_integrator: the driver froze its integrator record after the WRAGH traversal`
- `register_twice`: register a valid integrator at WRAGH, then register it again; CarpetX aborts the second call, with `CarpetX::register_integrator: a time integrator is already registered`
- `dense_output_stage_mismatch`: register a 2-stage integrator whose dense output is RK4; CarpetX aborts the run, with `the RK4 dense output belongs to a method with 4 RK stages, but the integrator registered 2`
- `weights_sum_mismatch`: register a 3-stage integrator whose weights b sum to 1.5; CarpetX aborts the run, with `CarpetX::register_integrator: the 3 RK stage weights b sum to 1.5, not 1`
- `reflux_without_weights`: register an integrator without weights b (an implicit method) that integrates conserved, which carries a fluxes= tag, under CarpetX::do_reflux with CarpetX::max_num_levels > 1; CarpetX aborts the run, with `but it registered no weights and integrates group TESTCARPETXAPI::CONSERVED, which carries a fluxes= tag`
- `subcycling_without_dense_output`: register an integrator without a dense output under CarpetX::use_subcycling; CarpetX aborts the run, with `CarpetX::use_subcycling = yes fills the refinement boundary by dense output, but the integrator registered dense_output = none`
- `integrated_not_checkpointed`: register an integrator that integrates unsaved, which is tagged checkpoint=no; CarpetX aborts the run, with `integrated group TESTCARPETXAPI::UNSAVED is tagged checkpoint="no"`
- `unregistered_rk_call`: call CarpetX::open_rk_step at CCTK_INITIAL without having registered an integrator; CarpetX aborts the run, with `CarpetX::open_rk_step: no time integrator is registered`
- `rk_before_freeze`: register a valid integrator at WRAGH and open an RK step in the same call, before the driver has frozen its integrator record; CarpetX aborts the run, with `CarpetX::open_rk_step: the integrator record is read before the driver froze it`
- `rk_open_invalid_state`: register a valid integrator and open an RK step before anything has written the interior of state; CarpetX aborts the run, with `CarpetX::open_rk_step: the RK step starts from the state's current time level (iteration 0): Grid function "TESTCARPETXAPI::state" is invalid on patch 0, refinement level 0, time level 0`
- `rhs_invalid`: register a valid integrator, fill state, open an RK step and report the right-hand side computed while nothing has written state_rhs; CarpetX aborts the run, with `RK stage 1 of 4 reads the interior of the right-hand side (iteration 0): Grid function "TESTCARPETXAPI::state_rhs" is invalid on patch 0, refinement level 0, time level 0`
- `rhs_repeated`: open an RK step on level 0 and report the right-hand side of RK stage 1 computed twice, without state_updated in between; CarpetX aborts the run, with `CarpetX::rk_step_t::rhs_computed on patches [0, 1), refinement levels [0, 1), RK stage 1 of 4: the stage already has its right-hand side; expected state_updated`
- `state_before_rhs`: open an RK step on level 0 and report the state updated before any right-hand side; CarpetX aborts the run, with `CarpetX::rk_step_t::state_updated on patches [0, 1), refinement levels [0, 1), RK stage 1 of 4: expected rhs_computed first`
- `rhs_after_last_stage`: open an RK step on level 0, run all four stages, and report a fifth right-hand side; CarpetX aborts the run, with `CarpetX::rk_step_t::rhs_computed on patches [0, 1), refinement levels [0, 1): all 4 stages of this step have run; destroy the rk_step_t and open the next step`
- `step_closed_early`: open an RK step on level 0 and close it after one of its four stages; CarpetX aborts the run, with `CarpetX::rk_step_t on patches [0, 1), refinement levels [0, 1) destroyed after 1 of its 4 stages`
