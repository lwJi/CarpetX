# ODESolvers

| Author(s)      | Erik Schnetter and Liwei Ji |
|:---------------|:----------------------------|
| Maintainer(s)  | Erik Schnetter and Liwei Ji |
| Licence        | LGPL |

## Purpose

Solve systems of coupled ordinary differential equations

## Subcycling

* Parameter `use_odesolvers_poststep_during_rksubsteps`

    * Set to `yes`:
        - After each RK substep, `ODESolvers_PostStep` will be called (user must scheudule a `SYNC` operation of state vector within this bin).
    * Set to `no`:
        - ~~After each RK substep, ODESolver will first call `ODESolvers_PostSubStepBeforeSync`~~
        - Automatically synchronize the state vector (**interprocess only**)
        - Call `ODESolvers_PostSubStep`.
    * Tips (when set to `no`):
        - we should remove `SYNC` from `ODESolvers_PostStep`, **no sync** (both interprocess and prolongation) should happend in this time bin. `interprocess` is harmless but redundant, while `prolongation` might fill the ghost points with wrong data (wrong time step).

### To Do

* [ ] Make Restriction sync interprocess ghosts only.
* [ ] Remove the first call of `calcys_rmbnd`.

### Rules

* [ ] Sync of state vector should only happen at RK substep and no where else.
* [ ] Restrict should not contain prolongation.
* [ ] We always fill the refinement boundary ghost zones right after we update the interior state vector.

### Tips

* We always fill the refinement boundary ghost zones right after we update the interior state vector, so that we can remove the extra time bin `ODESolvers_PostSubStepBeforeSync`.
Notice that changing of the order of, say `Z4cowGPU_Enforce` and `Sync` results in **different numerical values** of the state vector, because they don't commute.
    - *without subcycling*:
        * calling interior `Z4cowGPU_Enforce` and `Sync` within `ODESolvers_PostStep`
        will make the state vector at the refinement boundary **violate** the algebraic
        constraints due to interpolation.
        * calling `Sync` and apply `Z4cowGPU_Enforce` everywhere within `ODESolvers_PostStep`
        will ensure that the state vector **satisfies** the algebraic constraints everywhere.
        However, it loops more ghost zones compared to the previous case.
    - *with subcycling*:
        * *use `ODESolvers_PostStep` at RK substep*
            - calling interior `Z4cowGPU_Enforce` and `Sync` within `ODESolvers_PostStep`,
            then the refinement boundary ghost will be overwrite by `calcys_rmbnd`.
            (algebraic constraints are **violated** at refinement boundary ghosts).
            - calling `Sync` and apply `Z4cowGPU_Enforce` everywhere within `ODESolvers_PostStep`
            then the refinement boundary ghost will be overwrite by `calcys_rmbnd`.
            (algebraic constraints are **violated** at refinement boundary ghosts).
            - the reason these two cases are still different is that `ODESolvers_PostStep`
            will be called elsewhere (postrestrict for example).
            In the first case, the refinement boundary ghosts will violate the
            algebraic constraints, while in the second case, they will satisfy
            the algebraic constrains.
        * *use `ODESolvers_PostSubStep` at RK substep*
            - Compared to the case using `ODESolvers_PostStep` (`Sync` + `Z4cowGPU_Enforce`),
            since `Z4cowGPU_Enforce` is scheduled after `calcys_rmbnd`
            the ghost zones will **satisfy** the algebraic constraints.

## To Do

Implement IMEX methods as e.g. described in

Ascher, Ruuth, Spiteri: "Implicit-Explicit Runge-Kutta Methods for
Time-Dependent Partial Differential Equations", Appl. Numer. Math 25
(1997), pages 151-167,
<http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.48.1525&rep=rep1&type=pdf>.
