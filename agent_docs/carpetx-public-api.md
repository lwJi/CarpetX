# CarpetX's public API

A thorn reaches CarpetX through one header, `carpetx.hxx`, exported by `CarpetX/interface.ccl` from `CarpetX/src/include/`. Every other header under `CarpetX/src/` is private: it opens with `#ifndef CARPETX_INTERNAL` / `#error`, and only CarpetX's own `src/make.code.defn` defines `CARPETX_INTERNAL`, so a translation unit of any other thorn that includes a private header by any path (the old exported name, `../../CarpetX/src/...`, `../../../CarpetX/CarpetX/src/...`) stops at that `#error`. The public header includes nothing under `CarpetX/src/`; if it ever did, every client would fail to build, so it cannot leak the driver's internals transitively.

`python3 scripts/boundary check` lists every `#include` outside `CarpetX/src/` whose path reaches into `CarpetX/src`, and every `CARPETX_INTERNAL` that is not CarpetX's own `make.code.defn` line or a private header's `#ifndef` guard. `check --since REV` reports only the lines added since the merge base with `REV`, which is what the `Comments` workflow runs on every PR. The whole-tree check still lists the ODESolvers and PDESolvers includes until those thorns move over; the `--since` check is clean from the first slice on.

## Using the header

In `interface.ccl`, declare `USES INCLUDE HEADER: carpetx.hxx`; in `configuration.ccl`, `REQUIRES CarpetX` (the header includes AMReX, Loop and Arith headers, which that capability provides); in C++, `#include <carpetx.hxx>` and qualify every name with `CarpetX::`. The header adds no using-directive at any scope, so a client sees CarpetX names only when it names them.

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
| `CarpetX::RestrictNoPoison(cctkGH, min, max, groups)` | `CarpetX::restrict_no_poison(window, groups)` | groups must be grid-function groups |
| `CarpetX::synchronize()` | `CarpetX::synchronize()` | unchanged |

## External thorns

AsterX, MCNuX, BNSTools (VolumeIntegrals_GRMHDX, VolumeIntegrals_vacuumX, TriggerX), CapyrX_MultiPatch, SpacetimeX/NewRadX, RITHandOffX/ReadInterpTransformX and CurvBase include CarpetX's private headers today. Their builds stop at the `#error` once the headers they include are guarded; this guide's symbol map is the migration path, and it grows with each slice (field data and communication operations, reductions and timers, the RK contract in `carpetx_rk.hxx`).
