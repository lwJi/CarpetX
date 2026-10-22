# CarpetX's public API

A thorn reaches CarpetX through two headers exported by `CarpetX/interface.ccl` from `CarpetX/src/include/`: `carpetx.hxx` for everything a general thorn needs, and `carpetx_rk.hxx` for the time integrator's opt-in RK contract. Every other header under `CarpetX/src/` is private: it opens with `#ifndef CARPETX_INTERNAL` / `#error`, and only CarpetX's own `src/make.code.defn` defines `CARPETX_INTERNAL`, so a translation unit of any other thorn that includes a private header by any path (the old exported name, `../../CarpetX/src/...`, `../../../CarpetX/CarpetX/src/...`) stops at that `#error`. The public headers include nothing under `CarpetX/src/`; if one ever did, every client would fail to build, so they cannot leak the driver's internals transitively.

`python3 scripts/boundary check` lists every `#include` outside `CarpetX/src/` whose path reaches into `CarpetX/src`, and every `CARPETX_INTERNAL` that is not CarpetX's own `make.code.defn` line or a private header's `#ifndef` guard. `check --since REV` reports only the lines added since the merge base with `REV`, which is what the `Comments` workflow runs on every PR. The whole-tree check is clean: no thorn in this repository includes a private CarpetX header, and `driver.hxx`, `schedule.hxx`, `reduction.hxx` and `timer.hxx` are no longer exported (`timer.hxx` no longer exists; `Timer` and `Interval` live in `carpetx.hxx`).

## Using the header

In `interface.ccl`, declare `USES INCLUDE HEADER: carpetx.hxx` (and `carpetx_rk.hxx` for an integrator); in `configuration.ccl`, `REQUIRES CarpetX` (the headers include AMReX, Loop and Arith headers, which that capability provides); in C++, `#include <carpetx.hxx>` or `<carpetx_rk.hxx>` and qualify every name with `CarpetX::`. The headers add no using-directive at any scope, so a client sees CarpetX names only when it names them.

A handle is a name made of indices, resolved on every call: `level_t{patch, level}` names a refinement level of a patch, `group_t{level, gi}` a grid-function group on it. A stale handle names a level that no longer exists and the call reports it with `CCTK_VERROR`; it never dereferences freed memory. `window_t{min_level, max_level, min_patch, max_patch}` names a set of levels, both for traversal and as the "which levels" argument of every operation; `active_window()` is the window the current traversal covers (a `CCTK_VERROR` outside a traversal) and `all_levels()` the whole hierarchy.

## Symbol map

"Before" names what a thorn reached through the old exports or by relative include. The table covers every symbol the research catalog (`.tasks/carpetx-public-api/02-research-carpetx-public-api.md`, sections 2, 6 and 7) found a client using; the section after it lists the private names a client could reach through the old headers that have no public counterpart, and why.

| Before | After | Notes |
|---|---|---|
| `#include <driver.hxx>`, `<schedule.hxx>`, `"../../CarpetX/src/driver.hxx"`, ... | `#include <carpetx.hxx>` | with `USES INCLUDE HEADER: carpetx.hxx` |
| `ghext->num_patches()` | `CarpetX::num_patches()` | |
| `ghext->num_levels()`, `ghext->num_levels(patch)` | `CarpetX::num_levels(patch)` | per patch; the old no-argument form was the maximum over patches |
| `ghext->patchdata[p].leveldata[l]` | `CarpetX::level_t{p, l}` | a value, not a reference |
| `ghext->patchdata[p].leveldata[l - 1]` | `CarpetX::parent(level_t{p, l})` | `CCTK_VERROR` on level 0 |
| `leveldata.iteration`, `leveldata.delta_iteration` | `CarpetX::iteration(level)`, `CarpetX::delta_iteration(level)` | `CarpetX::rat64` |
| `patchdata.amrcore->Geom(level)` | `CarpetX::geometry(level)` | `const amrex::Geometry &` |
| `ghext->patchdata.at(patch).symmetries` | `CarpetX::symmetries(level_t{patch, level})` | `symmetry_t` now lives in `carpetx.hxx` |
| `active_levels->loop_serially(f)`, `->loop_coarse_to_fine(f)` | `CarpetX::active_window().loop_serially(f)`, `.loop_coarse_to_fine(f)` | `f` receives a `level_t`, not a `LevelData &` |
| `active_levels->loop_parallel(f)` | `CarpetX::active_window().loop_parallel(f)` | `f(level_t, index, component, cctkGH)` |
| `active_levels_t(min, max, pmin, pmax)` built by hand | `CarpetX::window_t{min, max, pmin, pmax}`, `CarpetX::all_levels()` | |
| `active_levels->min_level`, `->max_level` | `CarpetX::active_window().min_level`, `.max_level` | |
| `CarpetX::RestrictNoPoison(cctkGH, min, max, groups)` | `CarpetX::restrict_no_poison(window, groups)` | groups must be grid-function groups; `CCTK_VERROR` while a write scope on one of them is open |
| `CarpetX::synchronize()` | `CarpetX::synchronize()` | unchanged |
| `groupdata.numvars`, `.indextype`, `.nghostzones` | `CarpetX::numvars(group)`, `indextype(group)`, `nghostzones(group)` | `group_t{level, gi}` |
| `*groupdata.mfab.at(tl)` (read) | `CarpetX::data(group, tl)` | `const amrex::MultiFab &`; AMReX's const overloads give `Array4<const T>` and `const T *` |
| `*groupdata.mfab.at(tl)` (write), then `valid.at(tl).at(vi).set_all(...)` by hand | `auto w = CarpetX::write(group, tl, leaves_valid, "what"); w.mfab()` | move-only scope; when it is destroyed every variable of the group at `tl` gets exactly `leaves_valid`, reason `"what (iteration N)"`; a communication operation, `invalidate` or a scheduled function that reads, writes or syncs the group while a scope is open is a `CCTK_VERROR` |
| `valid_t`, `make_valid_int/outer/ghosts/all` | the same names in `carpetx.hxx` | `why_valid_t` stays private |
| `groupdata.valid.at(tl).at(vi).get()` | `CarpetX::valid(group, tl, vi)` | |
| `error_if_invalid(groupdata, vi, tl, where, msg)` + `check_valid_gf(...)` | `CarpetX::require_valid(group, tl, where, "why")` | every variable of the group; raises the error with CarpetX's explanation and runs the poison and NaN scan |
| `why_valid = why_valid_t(...)` / `set_all(valid_t(), ...)` to revoke | `CarpetX::invalidate(window, groups, tl, "why")` | all bits of every variable at `tl` |
| `SyncGroupsByDirI[Subcycling](cctkGH, n, groups, nullptr)` | `CarpetX::sync(window, groups)` | picks the subcycling variant when `use_subcycling`; every timelevel but the oldest, as a scheduled SYNC; sets the ghost (and outer) bits it fills |
| `SyncGroupsByDirIGhostOnly(cctkGH, n, groups, nullptr, tl)`, a `FillPatch_Sync` pipeline | `CarpetX::sync_ghosts_only(window, groups, tl = -1)` | same-level copy only; `tl = -1` is every timelevel but the oldest |
| `SyncGroupsByDirIProlongateOnly(...)`, a `FillPatch_Prolongate*` pipeline | `CarpetX::prolongate_ghosts(window, groups, tl = -1)` | coarse-to-fine interpolation only; marks the ghost (and outer) bits of the levels it filled |
| `SyncGroupsByDirIProlongateOnlyAligned(...)` | `CarpetX::prolongate_aligned(window, groups, tl = -1)` | under subcycling only into levels time-aligned with their parent; marks the ghost (and outer) bits of the levels it filled |
| `average_down*` by index-type rank over every level pair | `CarpetX::restrict_no_poison(window, groups)` | the rank dispatch is CarpetX's |
| `task_manager`, `FillPatch_Sync`, `FillPatch_ProlongateGhosts`, `FillPatch_ProlongateOnly` | the five operations above | the primitives stay private |
| `CarpetX::reduce(gi, vi, tl)`, `reduction<T, D>` | the same names in `carpetx.hxx` | moved verbatim; `reduction.hxx` is no longer exported and keeps only the MPI and OpenMP glue |
| `groupdata.init_tmp_mfabs()`, `alloc_tmp_mfab()`, `free_tmp_mfabs()` | `auto tmp = CarpetX::scratch(group); tmp.mfab()` | a move-only buffer with the boxes, distribution, components and ghosts of `data(group, 0)`, drawn from a per-group pool that CarpetX keeps from call to call and that dies with the level on a regrid; the destructor returns the buffer to the pool, so there is no init or free call, and a `scratch_t` that outlives its level is a `CCTK_VERROR` instead of a dangling pointer; it has no validity bits |
| `CarpetX::Timer`, `CarpetX::Interval` from `timer.hxx` | the same names in `carpetx.hxx` | moved verbatim; `timer.hxx` is gone |
| `CarpetX::rat64` | the same name in `carpetx.hxx` | `Arith::rational<std::int64_t>`, the type of `iteration` and `delta_iteration` |
| `ghext->use_subcycling`, `ghext->do_reflux`, `CCTK_ParameterGet("max_num_levels", "CarpetX")` | `SHARES: Driver` / `USES BOOLEAN use_subcycling` in `param.ccl`; `CCTK_ParameterGet(name, "CarpetX", &type)` | the flesh's parameter channel, not a CarpetX internal; the one read an integrator needed (`max_num_levels` for the reflux check) is done by `register_integrator` |
| `patchdata.amrcore->GetParGDB()`, `->finestLevel()`, `->refRatio(level)` | `CarpetX::num_levels(patch)`, `CarpetX::geometry(level)`, `CarpetX::data(group, tl).boxArray()` / `.DistributionMap()` | the BoxArray and DistributionMapping of a level are those of any group's MultiFab on it; `refRatio` is 2 between every pair of levels (`CarpetX::geometry` of parent and child give the cell sizes) |
| `*groupdata.mfab.at(tl)` then `mfab.FillBoundary(...)` by hand | `CarpetX::sync_ghosts_only(window, groups, tl)` | the same-level ghost exchange, with the ghost bits set by CarpetX |
| `Arith::synchronize()`, `Arith::update_cctkGH(...)` (RITHandOffX, resolved through the `namespace std` route) | `CarpetX::synchronize()`; the `cctkGH` argument of `window_t::loop_parallel` | the headers add nothing to `namespace std` or `namespace Arith`, so these spellings no longer resolve |
| `CarpetX::GridDescBase`, `Loop::GridDescBase` through `schedule.hxx` | `Loop::GridDescBase` from `<loop.hxx>` (`USES INCLUDE HEADER: loop.hxx`), or `DECLARE_CCTK_ARGUMENTSX_*` | a Loop type; no CarpetX header is needed |
| `CallScheduleGroup(cctkGH, "G")` (aliased function) | unchanged | the eleven aliased functions in `CarpetX/interface.ccl` are not part of this change; `run_schedule_group_at` is for a group that must run at another time or step |

## Private names with no public counterpart

Everything else the four old headers declared is driver-internal, and a thorn that named it was reaching past the driver's contract. The replacement is the operation CarpetX performs with it.

| Private name | Why there is no public form |
|---|---|
| `GHExt`, `PatchData`, `LevelData`, `GroupData`, `CommonGroupData`, `GlobalData`, `ArrayGroupData`, `ghext`, `active_levels`, `active_levels_t` | the data model; `level_t`, `group_t` and `window_t` name its parts, every public call resolves them |
| `CactusAmrCore`, `boundary_t`, `GroupData::interpolator`, `bcrecs`, `boundaries`, `parities`, `dirichlet_values`, `robin_values`, `fluxes`, `freg`, `freg_scratch`, `ks_source_band`, `old_source_band`, `rk_crse_patch`, `rk_fine_patch`, `face_area`, `cell_volume` | AMReX plumbing, group tags CarpetX parses, and the RK and flux buffers the RK step and `reflux` manage |
| `is_subcycling_level`, `last_step_iteration`, `last_finalized_iteration`, `recovered_iteration`, `recovered_levels`, `in_recovery_traversal`, `tail_bin`, `last_called_function`, `storage_frozen`, `active_timelevels`, `all_levels_synchronized`, `recovered_finer_level_is_behind` | traversal and recovery bookkeeping; `iteration`, `delta_iteration` and `window_is_restricted` answer what a client asked of them |
| `warn_inert_flux_tags`, `band_kind`, `subcycling_band_tag`, `rk_source_band` | the driver's own startup check and the checkpoint readers' view of the RK bands |
| `carpetx_epoch`, `CarpetX_GetEpoch` | the aliased function `GetEpoch` in `CarpetX/interface.ccl` |
| `Initialise`, `Evolve`, `Shutdown`, `CallFunction`, `GroupStorageIncrease`, `GroupStorageDecrease`, `EnableGroupStorage`, `DisableGroupStorage`, `QueryGroupStorageB`, `enter_*_mode`, `leave_*_mode`, `update_cctkGH`, `update_cctkGHs` | the flesh's driver callbacks; storage goes through `CCTK_EnableGroupStorage` and the schedule's `STORAGE:` lines, a group runs through `CallScheduleGroup` or `run_schedule_group_at`, and `loop_parallel` hands out the component's cGH |
| `MFPointer`, `GridDesc`, `GridPtrDesc`, `GridPtrDesc1` | how CarpetX builds the per-component `cGH` and `DECLARE_CCTK_ARGUMENTSX_*`; a thorn uses the result |
| `why_valid_t`, `error_if_invalid`, `warn_if_invalid`, `poison_invalid_gf`, `poison_invalid_ga`, `check_valid_gf`, `check_valid_ga`, `checksum_t`, `tiletag_t`, `checksums_t`, `calculate_checksums`, `check_checksums`, `ipoison_t`, `poison_value_t` | validity bookkeeping; `valid`, `require_valid`, `invalidate` and the write scope are the four things a client does with it, and CarpetX poisons and checksums around every scheduled call itself |
| `FillPatch_Sync`, `FillPatch_Prolongate`, `FillPatch_ProlongateGhosts`, `FillPatch_ProlongateOnly`, `Prolongate_Start`, `Prolongate_Finish`, `FillPatch_NewLevel`, `FillPatch_RemakeLevel`, `task_t`, `task_manager`, `average_down*` over level pairs | the pipelines behind the five communication operations and the regrid |
| `EnsureRKBuffers`, `StoreRKOldState`, `StoreRKStage`, `FillRKBoundary`, `AccumulateFluxes`, `EnsureFluxGeometry`, `Reflux`, `group_is_integrated`, `group_has_flux_register`, `num_rk_stages`, `rk_integrated_group` | the RK step object, `rk_recover_boundary`, `reflux` and `register_integrator` |
| `combine`, `reduction_CCTK_REAL`, the `omp declare reduction` and MPI type and op of `reduction.hxx` | glue of `reduce`; `reduction::operator+` combines two results in C++ |
| `symmetries` on `PatchData` | `CarpetX::symmetries(level)` |

## The integrator's contract: `carpetx_rk.hxx`

Exactly one thorn, the time integrator, publishes what it integrates and drives the flux registers. It registers once, `AT WRAGH`, with `register_integrator(rk_config_t{nstages, subcycling_ok, reflux_ok, integrated})`, where `integrated` holds `rk_pair_t{state_gi, rhs_gi}` pairs. CarpetX checks the record there, against its own parameters, and refuses: a second registration; a registration after the driver first read the record (`warn_inert_flux_tags`, right after PARAMCHECK, is the first reader, so anything later than WRAGH is too late); `nstages` outside {3, 4} or `subcycling_ok = false` under `use_subcycling`; `reflux_ok = false` under `do_reflux` with `max_num_levels > 1` when an integrated group carries a `fluxes=` tag; an integrated group that is not a grid function, is tagged `checkpoint="no"` or `evolve="no"`, is paired with a right-hand side of another size, or appears twice. A driver without a registration still runs (subcycling without an integrator, as before); only the RK operations below need one, and they name it in their error. `TestCarpetXAPI/test/errors/` holds one expected-failure parfile per refusal.

| Before | After | Notes |
|---|---|---|
| `ghext->num_rk_stages = n; ghext->rk_integrated_group = ...` at WRAGH | `CarpetX::register_integrator(config)` | once, `AT WRAGH`; `GHExt` holds one `optional<rk_config_t>` |
| `ODESolvers_CheckTags`, `ODESolvers_CheckMethod`'s subcycling and reflux checks, `CCTK_ParameterGet("max_num_levels", "CarpetX")` | the checks of `register_integrator` | CarpetX reads its own parameters |
| `CarpetX::max_num_rk_stages` from `driver.hxx` | the same name in `carpetx_rk.hxx` | |
| `*const_cast<CCTK_REAL *>(&cctkGH->cctk_time) = t; CallScheduleGroup(cctkGH, "G"); ... = t0` | `CarpetX::run_schedule_group_at(cctkGH, "G", t, delta_time)` | sets `cctk_time` and `cctk_delta_time` on the given cGH and on every driver cGH, traverses, restores both; `delta_time` is the raw `cctk_delta_time` the group sees (`CCTK_DELTA_TIME` divides it by `cctk_timefac`), so pass `cctkGH->cctk_delta_time` unless the step itself changes, as the IMEX half step does |
| `CarpetX::StoreRKOldState(patch, level, var_groups, tl)` | `auto step = CarpetX::open_rk_step(level_t{patch, level})` | one move-only `rk_step_t` per (level, step), opened once the state holds u(t_n); it stores the old state of the registered pairs that have storage on the level, requires their interior valid, and must run outside an OpenMP parallel region; `CCTK_VERROR` without a registration |
| `CarpetX::StoreRKStage(patch, level, var_groups, rhs_groups, n)` + `CarpetX::AccumulateFluxes(patch, level, n, weight)` | `auto stage = step.stage(n); stage.rhs_computed(weight)` | `stage(n)` yields the stages in order, each once, and refuses a call while another stage is open; `rhs_computed` stores the stage band and feeds the flux registers with `weight = b_n * dt`, and refuses a repeat |
| `CarpetX::FillRKBoundary(patch, level, var_groups, 0, n + 1, xsi, dtc)` after stage `n`, then `set_valid(make_valid_all())` by hand | `stage.fill_next_boundary(xsi, dtc)` | `xsi` is the substep's start within the parent's step (0 or 1/2), `dtc` the parent's step; stage `nstages` fills the virtual end of the substep (`xsi + 1/2`, stage 1); requires `rhs_computed` first; marks the integrated groups' ghosts, and outer boundary where all faces have symmetries or boundaries, valid; with `use_subcycling` a stage on a refined level must fill before it is destroyed, and level 0 refuses the call |
| the six calls in an order comments described | `~rk_stage_t`, `~rk_step_t` | a stage destroyed before `rhs_computed` (or, under subcycling on a refined level, before `fill_next_boundary`), and a step destroyed before every stage ran, are `CCTK_VERROR`s; `TestCarpetXAPI/test/errors/` holds one expected-failure parfile per refusal |
| recovery's `FillRKBoundary(..., 1, xsi + 1/2, dtc)` at the virtual end, then `set_valid(make_valid_all())` | `CarpetX::rk_recover_boundary(level, xsi, dtc)` | outside a step; marks what it filled like `fill_next_boundary`; `CarpetX::prolongate_aligned` covers the levels the end-of-step restriction re-prolongates and marks them likewise |
| `CarpetX::Reflux(cctkGH, min_level, max_level)` | `CarpetX::reflux(window)` | the integrator without subcycling (`all_levels()`); the driver under subcycling; `CCTK_VERROR` without a registration or while a write scope on an integrated group is open |
| `CarpetX::window_is_restricted(min_level, max_level)` | `CarpetX::window_is_restricted(window)` | an empty window is not restricted |
| `CarpetX::group_is_integrated(gi)`, `group_has_flux_register(gi)` | none | the integrator knows its own groups; CarpetX answers these from the registration |
| `CarpetX::get_group_checkpoint_flag(gi)`, `get_group_evolve_flag(gi)` | none | `register_integrator` checks both tags of every integrated group |

Under subcycling the SYNC of an integrated group on a refined level at iteration > 0 copies only the same-level ghosts and requires the ghost bit to be set already: the integrator's `fill_next_boundary`, or `rk_recover_boundary` and `prolongate_aligned` after a recovery, set it. A SYNC that finds it unset is a `CCTK_VERROR` naming the rule, where it used to skip those ghosts silently and rely on the integrator to mark them. On level 0 the SYNC fills every ghost itself and marks them, as without subcycling. No thorn outside CarpetX sets a validity bit.

## External thorns

AsterX, MCNuX, BNSTools (VolumeIntegrals_GRMHDX, VolumeIntegrals_vacuumX, TriggerX), CapyrX_MultiPatch, SpacetimeX/NewRadX, RITHandOffX/ReadInterpTransformX and CurvBase include CarpetX's private headers, by the old exported names or by relative path. Their builds stop at the `#error` at their next CarpetX update; the symbol map above is the migration path, and this table names the replacement for each thorn's use.

| Thorn | What it reaches today | Public replacement |
|---|---|---|
| AsterX (`src/sync.cxx`) | `FillPatch_Sync` in a `task_manager` pipeline, `RestrictNoPoison`, `SyncGroupsByDirIGhostOnly`, `SyncGroupsByDirIProlongateOnly`, `synchronize`, `active_levels->loop_serially` and `min_level`/`max_level`, `ghext->patchdata ... mfab.at(tl)` with `FillBoundary`, `amrcore->Geom`, `num_patches` | `sync_ghosts_only`, `restrict_no_poison`, `prolongate_ghosts`, `synchronize`, `active_window().loop_serially` and `.min_level`/`.max_level`, `geometry`, `num_patches`; a write that fills data by other means is a `write` scope stating what it leaves valid |
| MCNuX (`mcnux_cadence.cxx`, `mcnux_gather.hxx`, `mcnux_srcterms.cxx`) | the `ghext` tree walked by hand down to `mfab.at(0)->array(box)` in particle kernels, deposits into `Array4`s, `amrcore->Geom`/`GetParGDB`/`finestLevel`, `reduce` | `all_levels().loop_serially` or `level_t{p, l}` with `num_levels(patch)`, `data(group, tl)` for gathers (`const_array`), a `write` scope per deposited group stating interior validity, `geometry`, `data(...).boxArray()`/`.DistributionMap()` for the particle container, `reduce` |
| BNSTools VolumeIntegrals_GRMHDX, VolumeIntegrals_vacuumX | relative `driver.hxx`, `reduction.hxx` for `reduce` and `reduction<>` | `#include <carpetx.hxx>`; the names are unchanged |
| BNSTools TriggerX | exported `reduction.hxx` for `reduce` and every `reduction` field | `USES INCLUDE HEADER: carpetx.hxx`, `#include <carpetx.hxx>`; the names are unchanged, and the `namespace CarpetX { using Loop::dim; }` prelude is no longer needed |
| CapyrX_MultiPatch | relative `schedule.hxx`, `timer.hxx` for a hand-built `active_levels_t().loop_parallel`, `Timer`, `Interval`, and `GridDescBase` | `all_levels().loop_parallel` (or `active_window()`), `Timer` and `Interval` from `carpetx.hxx`, `Loop::GridDescBase` from `<loop.hxx>` |
| SpacetimeX/NewRadX | exported `driver.hxx` for `symmetry_t` and `ghext->patchdata.at(cctk_patch).symmetries` | `USES INCLUDE HEADER: carpetx.hxx`; `CarpetX::symmetries(level_t{cctk_patch, cctk_level})` and `symmetry_t` from `carpetx.hxx` |
| RITHandOffX/ReadInterpTransformX | exported `schedule.hxx` for `active_levels->loop_parallel`, `GridDescBase`, `Arith::synchronize()`, `Arith::update_cctkGH()` | `USES INCLUDE HEADER: carpetx.hxx`; `active_window().loop_parallel`, `Loop::GridDescBase`, `CarpetX::synchronize()`, and the component cGH the kernel receives |
| CurvBase | `USES INCLUDE HEADER: driver.hxx`, `schedule.hxx` in `interface.ccl`, used by no source | delete the two lines |

Patches to the repositories under this fork's control (AsterX, SpacetimeX, MCNuX, CurvBase) are follow-up work; BNSTools, CapyrX and RITHandOffX receive this guide.
