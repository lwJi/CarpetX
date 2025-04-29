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

### Rules

* [ ] Sync of state vector should only happen at RK substep and no where else.
* [ ] Restrict should not contain prolongation.

## To Do

Implement IMEX methods as e.g. described in

Ascher, Ruuth, Spiteri: "Implicit-Explicit Runge-Kutta Methods for
Time-Dependent Partial Differential Equations", Appl. Numer. Math 25
(1997), pages 151-167,
<http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.48.1525&rep=rep1&type=pdf>.
