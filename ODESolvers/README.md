# ODESolvers

| Author(s)      | Erik Schnetter and Liwei Ji |
|:---------------|:----------------------------|
| Maintainer(s)  | Erik Schnetter and Liwei Ji |
| Licence        | LGPL |

## Purpose

Solve systems of coupled ordinary differential equations

## Subcycling

* Parameter `interprocess_ghost_sync_during_substep`

    * Set to `no`: After each RK substep, `ODESolvers_PostStep` will be called. The user must scheudule a `SYNC` operation of state vector within this bin.
    * Set to `yes`: After each RK substep, ODESolver will automatically synchronize the state vector (**interprocess only**) and then call `ODESolvers_PostSubStep`. The user should schedule the same operations as in `ODESolvers_PostStep` except for the `SYNC` of the state vector.

## To Do

Implement IMEX methods as e.g. described in

Ascher, Ruuth, Spiteri: "Implicit-Explicit Runge-Kutta Methods for
Time-Dependent Partial Differential Equations", Appl. Numer. Math 25
(1997), pages 151-167,
<http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.48.1525&rep=rep1&type=pdf>.
