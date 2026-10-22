# CarpetX's public API

A thorn reaches CarpetX through two headers exported by `CarpetX/interface.ccl` from `CarpetX/src/include/`: `carpetx.hxx` for everything a general thorn needs, and `carpetx_rk.hxx` for the time integrator's opt-in RK contract. Every other header under `CarpetX/src/` is private: it opens with `#ifndef CARPETX_INTERNAL` / `#error`, and only CarpetX's own `src/make.code.defn` defines `CARPETX_INTERNAL`, so a translation unit of any other thorn that includes a private header by any path (the old exported name, `../../CarpetX/src/...`, `../../../CarpetX/CarpetX/src/...`) stops at that `#error`. The public headers include nothing under `CarpetX/src/`; if one ever did, every client would fail to build, so they cannot leak the driver's internals transitively.

`python3 scripts/boundary check` lists every `#include` outside `CarpetX/src/` whose path reaches into `CarpetX/src`, and every `CARPETX_INTERNAL` that is not CarpetX's own `make.code.defn` line or a private header's `#ifndef` guard. `check --since REV` reports only the lines added since the merge base with `REV`, which is what the `Comments` workflow runs on every PR. The whole-tree check still lists the ODESolvers and PDESolvers includes until those thorns move over; the `--since` check is clean from the first slice on.

## Using the header

In `interface.ccl`, declare `USES INCLUDE HEADER: carpetx.hxx` (and `carpetx_rk.hxx` for an integrator); in `configuration.ccl`, `REQUIRES CarpetX` (the headers include AMReX, Loop and Arith headers, which that capability provides); in C++, `#include <carpetx.hxx>` or `<carpetx_rk.hxx>` and qualify every name with `CarpetX::`. The headers add no using-directive at any scope, so a client sees CarpetX names only when it names them.

A handle is a name made of indices, resolved on every call: `level_t{patch, level}` names a refinement level of a patch, `group_t{level, gi}` a grid-function group on it. A stale handle names a level that no longer exists and the call reports it with `CCTK_VERROR`; it never dereferences freed memory. `window_t{min_level, max_level, min_patch, max_patch}` names a set of levels, both for traversal and as the "which levels" argument of every operation; `active_window()` is the window the current traversal covers (a `CCTK_VERROR` outside a traversal) and `all_levels()` the whole hierarchy.

## Symbol map

Each later slice adds its rows. "Before" names what a thorn reached through the old exports or by relative include.

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
| `SyncGroupsByDirIProlongateOnly(...)`, a `FillPatch_Prolongate*` pipeline | `CarpetX::prolongate_ghosts(window, groups, tl = -1)` | coarse-to-fine interpolation only |
| `SyncGroupsByDirIProlongateOnlyAligned(...)` | `CarpetX::prolongate_aligned(window, groups, tl = -1)` | under subcycling only into levels time-aligned with their parent |
| `average_down*` by index-type rank over every level pair | `CarpetX::restrict_no_poison(window, groups)` | the rank dispatch is CarpetX's |
| `task_manager`, `FillPatch_Sync`, `FillPatch_ProlongateGhosts`, `FillPatch_ProlongateOnly` | the five operations above | the primitives stay private |
| `CarpetX::reduce(gi, vi, tl)`, `reduction<T, D>` | the same names in `carpetx.hxx` | moved verbatim; `reduction.hxx` is no longer exported and keeps only the MPI and OpenMP glue |

## The integrator's contract: `carpetx_rk.hxx`

Exactly one thorn, the time integrator, publishes what it integrates and drives the flux registers. It registers once, `AT WRAGH`, with `register_integrator(rk_config_t{nstages, subcycling_ok, reflux_ok, integrated})`, where `integrated` holds `rk_pair_t{state_gi, rhs_gi}` pairs. CarpetX checks the record there, against its own parameters, and refuses: a second registration; a registration after the driver first read the record (`warn_inert_flux_tags`, right after PARAMCHECK, is the first reader, so anything later than WRAGH is too late); `nstages` outside {3, 4} or `subcycling_ok = false` under `use_subcycling`; `reflux_ok = false` under `do_reflux` with `max_num_levels > 1` when an integrated group carries a `fluxes=` tag; an integrated group that is not a grid function, is tagged `checkpoint="no"` or `evolve="no"`, is paired with a right-hand side of another size, or appears twice. A driver without a registration still runs (subcycling without an integrator, as before); only the RK operations below need one, and they name it in their error. `TestCarpetXAPI/test/errors/` holds one expected-failure parfile per refusal.

| Before | After | Notes |
|---|---|---|
| `ghext->num_rk_stages = n; ghext->rk_integrated_group = ...` at WRAGH | `CarpetX::register_integrator(config)` | once, `AT WRAGH`; `GHExt` holds one `optional<rk_config_t>` |
| `ODESolvers_CheckTags`, `ODESolvers_CheckMethod`'s subcycling and reflux checks, `CCTK_ParameterGet("max_num_levels", "CarpetX")` | the checks of `register_integrator` | CarpetX reads its own parameters |
| `CarpetX::max_num_rk_stages` from `driver.hxx` | the same name in `carpetx_rk.hxx` | |
| `*const_cast<CCTK_REAL *>(&cctkGH->cctk_time) = t; CallScheduleGroup(cctkGH, "G"); ... = t0` | `CarpetX::run_schedule_group_at(cctkGH, "G", t, delta_time)` | sets `cctk_time` and `cctk_delta_time` on the given cGH and on every driver cGH, traverses, restores both; `delta_time` is the raw `cctk_delta_time` the group sees (`CCTK_DELTA_TIME` divides it by `cctk_timefac`), so pass `cctkGH->cctk_delta_time` unless the step itself changes, as the IMEX half step does |
| `CarpetX::AccumulateFluxes(patch, level, stage, weight)` | `CarpetX::accumulate_fluxes(level_t{patch, level}, stage, weight)` | `CCTK_VERROR` without a registration or for a stage outside `1..nstages`; Phase 4 folds it into `rk_stage_t::rhs_computed` |
| `CarpetX::Reflux(cctkGH, min_level, max_level)` | `CarpetX::reflux(window)` | the integrator without subcycling (`all_levels()`); the driver under subcycling; `CCTK_VERROR` without a registration or while a write scope on an integrated group is open |
| `CarpetX::window_is_restricted(min_level, max_level)` | `CarpetX::window_is_restricted(window)` | an empty window is not restricted |
| `CarpetX::group_is_integrated(gi)`, `group_has_flux_register(gi)` | none | the integrator knows its own groups; CarpetX answers these from the registration |
| `CarpetX::get_group_checkpoint_flag(gi)`, `get_group_evolve_flag(gi)` | none | `register_integrator` checks both tags of every integrated group |

## External thorns

AsterX, MCNuX, BNSTools (VolumeIntegrals_GRMHDX, VolumeIntegrals_vacuumX, TriggerX), CapyrX_MultiPatch, SpacetimeX/NewRadX, RITHandOffX/ReadInterpTransformX and CurvBase include CarpetX's private headers today. Their builds stop at the `#error` once the headers they include are guarded; this guide's symbol map is the migration path, and it grows with each slice (scratch buffers, timers, the RK step object in `carpetx_rk.hxx`).
