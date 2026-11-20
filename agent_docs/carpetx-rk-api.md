# CarpetX's integrator API

`python3 scripts/lint doc --write` writes this file from `scripts/lint/boundary/guide.py`. A row says only what a header cannot: a signature is read in the header.

`carpetx_rk.hxx` is for the one thorn that is the time integrator. `agent_docs/carpetx-public-api.md` covers `carpetx.hxx` and the validity regions named here, and `TestCarpetXAPI` is a client of both.

## The integrator's contract

Exactly one thorn, the time integrator, publishes what it integrates and reports the events of its RK step. Registration is `AT WRAGH`, once. `dense_output` names the method whose dense output fills the refinement boundary under subcycling; it is named, not inferred from the stage count, and `none` cannot run under subcycling. `b` holds the final-update weight of every stage, all zero for a method that never updates the state.

`rk_dense_output_method` states the method each dense output was written for as a whole Butcher tableau: its stage count, its order, `a`, its weights `b`, and its stage times `c`, in the order the stages arrive through `rhs_computed`. The dense output's weights reduce to that `b` at the end of the parent step, which CarpetX asserts at build. An integrator that registers a dense output is that method; registration checks only its stage count, so the integrator holds its own `b` and `c` to this one.

`run_schedule_group_at` sets `cctk_time` and `cctk_delta_time` on every cGH for the traversal and then restores both; pass `cctkGH->cctk_delta_time` unless the group is to see a step of another size.

`open_rk_step` is called once the state holds u(t_n), outside any OpenMP parallel region. `rhs_computed` feeds the flux registers with the stage's weight times that level's `CCTK_DELTA_TIME`. `state_updated` says the state holds the next stage value: under subcycling it fills the coarse-fine ghosts of every refined level in the window and marks them valid; the same-level ghosts and the outer boundary of the integrated groups stay invalid until the thorn's SYNC in ODESolvers_PostStep. Without subcycling the last `state_updated` applies the flux registers.

Under subcycling, on a refined level at iteration > 0 the SYNC of an integrated group keeps the coarse-fine ghosts that `state_updated` or the driver's refill after a recovery wrote; elsewhere it fills every ghost itself. After a recovery the driver refills the integrated groups' refinement boundary before `CCTK_POST_RECOVER_VARIABLES`; the integrator schedules nothing for it.
