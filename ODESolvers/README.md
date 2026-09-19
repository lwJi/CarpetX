# ODESolvers

| Author(s)      | Erik Schnetter and Liwei Ji |
|:---------------|:----------------------------|
| Maintainer(s)  | Erik Schnetter and Liwei Ji |
| Licence        | LGPL |


## Purpose

Solve systems of coupled ordinary differential equations


## Subcycling

Add the following parameters to your parameter file

```
CarpetX::use_subcycling = yes
CarpetX::restrict_during_sync = no
```

Subcycling supports the methods `RK4` and `SSPRK3`, and evolved groups must have a single time level.

### Device waits on GPUs

Under subcycling the solver takes a complete RK step on one level per call. ODESolvers itself never waits for the device; the three driver calls it makes do, once each: where the start-of-step state is stored (`StoreRKOldState`), once per stage where the stage's right-hand side is stored (`StoreRKStage`), and once per stage where the refinement-boundary ghosts are filled from the parent level's dense output (`FillRKBoundary`; level 0 has no such fill). This gives the following number of waits per solver call, independent of the number of evolved groups and of how a level is split into boxes:

| Method   | Refined level | Level 0 |
|:---------|:--------------|:--------|
| `RK4`    | 9             | 5       |
| `SSPRK3` | 7             | 4       |

All of these wait for all GPU streams; there is no full-device wait. Each RK linear combination is one kernel launch per evolved group. A `SYNC` that only exchanges ghosts between boxes of the same level waits once; a `SYNC` that prolongates from the next coarser level waits three times. Whatever the scheduled routines in `ODESolvers_RHS` and `ODESolvers_PostStep` do comes on top of this.

### Counter report

These numbers can be checked for any run, on CPU builds as well as on GPU builds, with

```
CarpetX::out_subcycling_counts = yes
```

(default `no`). The driver then counts the device waits, temporary buffers and kernel launches of the solver, the refinement-boundary fill and the ghost `SYNC`s. It counts logical wait points: on a CPU build the wait itself is empty, but it is still counted, so a CPU run reports the numbers a GPU run would. The report covers the evolution only, not initialisation or recovery.

The detail goes to the file `subcycling-counts.tsv` in `IO::out_dir`, written by process 0, with one row per refinement level per coarse step (a coarse step is complete when level 0 has stepped and all levels are aligned in time again). The columns are separated by tabs:

| Column | Name                      | Meaning |
|:-------|:--------------------------|:--------|
| 1      | `coarse_step`             | Iteration of level 0 at the end of the coarse step |
| 2      | `level`                   | Refinement level |
| 3      | `solver_calls`            | Solver calls on this level in this coarse step (normally `2^level`) |
| 4      | `stream_waits_per_call`   | Waits for all GPU streams per solver call |
| 5      | `device_waits_per_call`   | Waits for the whole device per solver call |
| 6      | `temp_buffers_per_call`   | Temporary `MultiFab`s allocated by the refinement-boundary fill per solver call |
| 7      | `launches_per_lincomb`    | Kernel launches per RK linear combination |
| 8      | `waits_per_ghost_sync`    | Waits per `SYNC` that only exchanges same-level ghosts; 0 if there was none on this level |
| 9      | `regrid`                  | 1 if a regrid changed a level during this coarse step, else 0 |
| 10     | `buffer_bytes`            | Memory held by subcycling's persistent buffers on this level, in bytes, summed over all processes |

Columns 4 to 8 report the worst case of the coarse step (the worst solver call, the linear combination with the most launches, the worst `SYNC`), not an average, so that a single outlier cannot hide. For example, `RK4` with three levels and one evolved group gives

```
# subcycling-counts.tsv
# 1:coarse_step	2:level	3:solver_calls	4:stream_waits_per_call	5:device_waits_per_call	6:temp_buffers_per_call	7:launches_per_lincomb	8:waits_per_ghost_sync	9:regrid	10:buffer_bytes
1	0	1	5	0	0	1	1	0	408800
1	1	2	9	0	0	1	1	0	606624
1	2	4	9	0	0	1	1	0	197824
```

Things to know when reading the file:

- Waits inside the scheduled routines of `ODESolvers_RHS` and `ODESolvers_PostStep` are not charged to the solver call. The waits of a `SYNC` are charged to the `SYNC`, and show up in column 8 only.
- The waits of `CarpetX::poison_undefined_values` (poisoning, validity checks, checksums) are not counted anywhere, and neither are waits inside AMReX (in its parallel copies, or the first time a `MultiFab` hands out its `arrays()` for a fused kernel).
- A `SYNC` that covers several levels is charged to the finest of them. Column 8 only reports `SYNC`s that exchange same-level ghosts and nothing else: every `SYNC` on level 0, and on a refined level a `SYNC` of evolved groups (group tag `evolve`, which defaults to `checkpoint`) once that level has taken its first step. `SYNC`s that prolongate keep their three waits and are not reported.
- A coarse step is flagged in column 9 only if a regrid actually changed a level, not every time `Driver::regrid_every` comes around. `buffer_bytes` can only change in flagged steps.
- All numbers are independent of the number of processes: the counts are logical, and `buffer_bytes` is a global sum.

At shutdown the driver prints a summary with the worst coarse step per level, taken over the unflagged (steady-state) coarse steps; if every coarse step was flagged it says so and takes all of them:

```
INFO (CarpetX): Subcycling path counts, worst steady-state coarse step (3 levels, RK4, G=1):
  level  solver calls  stream waits/call  device waits/call  temp buffers/call  launches/lincomb  waits/ghost sync
  0      1             5                  0                  0                  1                 1
  1      2             9                  0                  0                  1                 1
  2      4             9                  0                  0                  1                 1
  subcycling buffers held (bands + fill buffers): level 0: 0.39 MiB, level 1: 0.58 MiB, level 2: 0.19 MiB
  note: poison_undefined_values = yes - validity/poison/checksum waits are NOT included above;
        set it to "no" for production GPU runs.
```

`G` is the number of evolved groups, and the buffer line shows the last such coarse step. A run that reports more than the numbers in the table above, a nonzero `device_waits_per_call` or `temp_buffers_per_call`, or a `launches_per_lincomb` different from `G`, has picked up a wait, an allocation or a per-box launch on the subcycling path that should not be there. The tests `TestSubcyclingMC2/test/counts_rk4` and `counts_ssprk3` pin these numbers.

### Memory held by subcycling

Subcycling keeps two kinds of buffers alive between steps, per evolved group. A level that has a finer level holds the *source bands*: its start-of-step state and the right-hand side of each RK stage (5 buffers for `RK4`, 4 for `SSPRK3`) on the strip of coarse points that the finer level's refinement-boundary ghosts are interpolated from. A refined level holds the two *fill buffers* of `FillRKBoundary`: the parent's dense output on that same strip, and its interpolation onto the level's refinement-boundary ghosts. So level 0 holds bands only and the finest level fill buffers only. None of these buffers covers a whole level.

The fill buffers are allocated by the first fill after a level was created or changed by a regrid, and are kept until the next regrid changes that level; a fill in between allocates nothing. They used to be allocated and freed at every RK stage, so a run now holds somewhat more device memory than before. How much is what `buffer_bytes` and the last line of the summary report (bands plus fill buffers, per level). Take it into account when sizing a run that fills the device. The fill buffers are pure scratch and are not written to checkpoints; the source bands are, when a checkpoint is taken in the middle of a coarse step.

### Production runs on GPUs

Set

```
CarpetX::poison_undefined_values = no
```

for production runs on GPUs. It is `yes` by default, and then every scheduled routine is wrapped in poisoning, validity checks and checksums, each of which ends in a device wait of its own, and the checksums read grid functions on the host. These waits come on top of the ones listed above and are not part of the counter report; its summary ends in a note saying so whenever the parameter is set. Keep the default while developing a thorn; it is what catches reads of undefined grid points.


## To Do

Implement IMEX methods as e.g. described in

Ascher, Ruuth, Spiteri: "Implicit-Explicit Runge-Kutta Methods for
Time-Dependent Partial Differential Equations", Appl. Numer. Math 25
(1997), pages 151-167,
<http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.48.1525&rep=rep1&type=pdf>.
