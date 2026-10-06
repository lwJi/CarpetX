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


## Subcycling

Add the following parameters to your parameter file

```
CarpetX::use_subcycling = yes
CarpetX::restrict_during_sync = no
```

Memory: under subcycling each refined level holds, per evolved group, two extra zero-ghost work buffers over its coarse-fine boundary footprint, one at the coarse and one at the fine resolution (the latter has about 8 times the cells of the former). They come on top of the `num_rk_stages + 1` coarse-resolution source bands that each refined level keeps per evolved group, which hold its parent's start-of-step state and stage derivatives on the same footprint. The bands and the two work buffers are allocated together, when the parent level first stores its old state after the refined level was made, and stay allocated until the next regrid remakes the level, so that the per-stage boundary fill allocates nothing. The bands of a group cover the footprint of that group's own prolongation stencil, so groups with different `prolongation_type` / `prolongation_order` tags may be mixed freely. Runs without subcycling never allocate any of these.

Recovery: under subcycling the refinement levels advance on separate clocks, so a checkpoint can hold levels at up to as many different times as there are levels (with three levels, at every iteration of the form 4m+1). On recovery, CarpetX restores each level's clock right after reading the checkpoint and then visits the clock groups (maximal runs of adjacent levels sharing one clock) from the coarsest to the finest. For each group it runs `CCTK_RECOVER_VARIABLES`, `CCTK_POST_RECOVER_VARIABLES`, (for the finest group only) the restriction with `CarpetX_PreRestrict` and `CCTK_POSTRESTRICT`, `CCTK_POSTSTEP`, and `CCTK_ANALYSIS`, with that group's time and time factor, the same ones the uninterrupted run gave the group's last end-of-step bins. `CCTK_CPINITIAL` then runs once, over the finest group. `ODESolvers_Solve_Subcycling_Recovery` runs in `CCTK_RECOVER_VARIABLES` and rebuilds the refinement-boundary ghosts of the integrated groups on the group's levels, from the parent's restored Runge-Kutta source bands for a level that is behind its parent and by spatial prolongation otherwise, so they are valid before `ODESolvers_PostStep` runs in `CCTK_POST_RECOVER_VARIABLES`. Routines in these bins therefore run once per clock group on recovery. Because `CCTK_ANALYSIS` runs again at the checkpoint iteration, analysis output that is appended to a file (reductions, horizon finders, wave extraction) gets a duplicate row for that iteration on every subcycling restart. Without subcycling, recovery runs these bins once over all levels and skips `CCTK_ANALYSIS`, as before.


## To Do

Implement IMEX methods as e.g. described in

Ascher, Ruuth, Spiteri: "Implicit-Explicit Runge-Kutta Methods for
Time-Dependent Partial Differential Equations", Appl. Numer. Math 25
(1997), pages 151-167,
<http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.48.1525&rep=rep1&type=pdf>.
