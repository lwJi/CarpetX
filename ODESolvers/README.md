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

Refinement-boundary ghosts: under subcycling ODESolvers alone fills the coarse-fine ghosts of the evolved (`rhs=`) groups on refined levels; the subcycling SYNC leaves them alone. Before each Runge-Kutta stage of a fine substep, it evaluates the parent's dense-output polynomial at the stage time and prolongates that coarse state in space. At the end of each fine substep (the "virtual end-of-step") it fills them the way the next substep's first stage needs them:

- A level that lags its parent (its first substep) gets dense output at the end of its substep, with the parent's step taken from the level clocks.
- A level that is aligned with its parent (its second substep, which lands on the parent's time) gets the parent's final state (`tl = 0`) prolongated in space, as restriction windows do too. Dense output at the end of the parent's step is the same state mathematically, but it rounds differently.

After recovery from a checkpoint, `ODESolvers_Solve_Subcycling_Recovery` (at `CCTK_CPINITIAL`) refills these ghosts on the levels of its window with the same function as the virtual end-of-step, `fill_end_of_step_ghosts`, so that both cases are filled exactly as evolution filled them.

Memory: under subcycling each refined level holds, per evolved group, two extra zero-ghost work buffers over its coarse-fine boundary footprint, one at the coarse and one at the fine resolution (the latter has about 8 times the cells of the former). They come on top of the `num_rk_stages + 1` coarse-resolution source bands that each refined level keeps per evolved group, which hold its parent's start-of-step state and stage derivatives on the same footprint. The bands and the two work buffers are allocated together, when the parent level first stores its old state after the refined level was made, and stay allocated until the next regrid remakes the level, so that the per-stage boundary fill allocates nothing. The bands of a group cover the footprint of that group's own prolongation stencil, so groups with different `prolongation_type` / `prolongation_order` tags may be mixed freely. Runs without subcycling never allocate any of these.


## To Do

Implement IMEX methods as e.g. described in

Ascher, Ruuth, Spiteri: "Implicit-Explicit Runge-Kutta Methods for
Time-Dependent Partial Differential Equations", Appl. Numer. Math 25
(1997), pages 151-167,
<http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.48.1525&rep=rep1&type=pdf>.
