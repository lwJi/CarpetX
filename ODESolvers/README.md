# ODESolvers

| Author(s)      | Erik Schnetter and Liwei Ji |
|:---------------|:----------------------------|
| Maintainer(s)  | Erik Schnetter and Liwei Ji |
| Licence        | LGPL |


## Purpose

Solve systems of coupled ordinary differential equations


## Tag requirements

A group with an `rhs=` tag is integrated by ODESolvers. ODESolvers checks its tags at startup (`CCTK_PARAMCHECK`) and aborts with an error naming the group if they are not allowed:

- It must be checkpointed, i.e. it must not set `checkpoint="no"`, whatever its `evolve` tag. A checkpoint would omit its state and, under subcycling, its Runge-Kutta source bands, so recovery could not restore it. Enforced by `ODESolvers_CheckTags`, which tests this rule first.
- It must be evolved, i.e. it must not set `evolve="no"`. Regrid would not reliably refill its state. Enforced by `ODESolvers_CheckTags`.
- Under subcycling (`CarpetX::use_subcycling = yes`) it must also have a single timelevel. Enforced by `ODESolvers_CheckTimelevels`.

The default tags (no `checkpoint` or `evolve` tag) satisfy the first two rules.


## SYNC rule under subcycling

Under subcycling (`CarpetX::use_subcycling = yes`), `CarpetX_PreRestrict` and `CCTK_POSTRESTRICT` run over windows whose lowest level may be behind its parent, so a `SYNC` there of a group that ODESolvers does not integrate prolongates the group's refinement-boundary ghosts from the parent level's current data. On recovery, CarpetX replays these bins for the finest clock group only, so the parent's value of a non-checkpointed group may not have been recomputed. CarpetX therefore rejects a routine reached in `CarpetX_PreRestrict` or `CCTK_POSTRESTRICT` whose `SYNC:` clause names a grid function group that is neither integrated (`rhs=` tag) nor checkpointed.

- The bin is the one CarpetX is traversing, so the rule also applies to routines in schedule groups nested in it. A routine scheduled `IN ODESolvers_PostStep` is checked when it is reached through `ODESolvers_PostStep AT postrestrict`, but not when it is called from inside a Runge-Kutta stage.
- Only grid function groups are checked; grid scalars and arrays have no refinement-boundary ghosts.
- Without subcycling, nothing is checked.

CarpetX checks a routine's `SYNC:` clause the first time it calls the routine in one of these bins, before the routine runs, and aborts if the clause names a rejected group. The message names the routine, the bin (for a nested schedule group, as `<group>, reached from <bin>`), every rejected group by its full name, and the fix, for example:

```
Under subcycling, FluxWaveToyX::FluxWaveToyX_PreRestrictEnergy (scheduled in CarpetX_PreRestrict) SYNCs the non-checkpointed group FLUXWAVETOYX::PRERESTRICT_ENERGY. Routines in CarpetX_PreRestrict and CCTK_POSTRESTRICT may SYNC only integrated or checkpointed groups: after a restart, a non-checkpointed group's value on the parent level may not have been recomputed. Checkpoint the group, or write it without a SYNC.
```

To write such a group without a `SYNC`, the routine writes it on the interior and drops the `SYNC:` clause. Its ghost zones stay invalid, and CarpetX's 1D TSV output omits every point in a region where the group is not valid.


## Subcycling

Add the following parameters to your parameter file

```
CarpetX::use_subcycling = yes
CarpetX::restrict_during_sync = no
```

Memory: under subcycling each refined level holds, per evolved group, two extra zero-ghost work buffers over its coarse-fine boundary footprint, one at the coarse and one at the fine resolution (the latter has about 8 times the cells of the former). They come on top of the `num_rk_stages + 1` coarse-resolution source bands that each refined level keeps per evolved group, which hold its parent's start-of-step state and stage derivatives on the same footprint. The bands and the two work buffers are allocated together, when the parent level first stores its old state after the refined level was made, and stay allocated until the next regrid remakes the level, so that the per-stage boundary fill allocates nothing. The bands of a group cover the footprint of that group's own prolongation stencil, so groups with different `prolongation_type` / `prolongation_order` tags may be mixed freely. Runs without subcycling never allocate any of these.

Recovery: under subcycling the refinement levels advance on separate clocks, so a checkpoint can hold levels at up to as many different times as there are levels (with three levels, at every iteration of the form 4m+1). On recovery, CarpetX restores each level's clock right after reading the checkpoint and then visits the clock groups (maximal runs of adjacent levels sharing one clock) from the coarsest to the finest. For each group it runs `CCTK_RECOVER_VARIABLES`, `CCTK_POST_RECOVER_VARIABLES`, (for the finest group only) the restriction with `CarpetX_PreRestrict` and `CCTK_POSTRESTRICT`, `CCTK_POSTSTEP`, and `CCTK_ANALYSIS`, with the iteration, time and time factor the uninterrupted run gave the group's last end-of-step bins. That iteration is the one in which the group's finest level stepped last, which for a coarser group can precede the checkpoint iteration (with three levels at iteration 6, level 0 last stepped at iteration 5). `CCTK_CPINITIAL` then runs once, over the finest group. `ODESolvers_Solve_Subcycling_Recovery` runs in `CCTK_RECOVER_VARIABLES` and rebuilds the refinement-boundary ghosts of the integrated groups on the group's levels, from the parent's restored Runge-Kutta source bands for a level that is behind its parent and by spatial prolongation otherwise, so they are valid before `ODESolvers_PostStep` runs in `CCTK_POST_RECOVER_VARIABLES`. Routines in these bins therefore run once per clock group on recovery. Because `CCTK_ANALYSIS` runs again for the replayed iterations, analysis output that is appended to a file (reductions, horizon finders, wave extraction) gets duplicate rows for them on every subcycling restart. Without subcycling, recovery runs these bins once over all levels and skips `CCTK_ANALYSIS`, as before.


## To Do

Implement IMEX methods as e.g. described in

Ascher, Ruuth, Spiteri: "Implicit-Explicit Runge-Kutta Methods for
Time-Dependent Partial Differential Equations", Appl. Numer. Math 25
(1997), pages 151-167,
<http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.48.1525&rep=rep1&type=pdf>.
