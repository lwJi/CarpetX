# CarpetX's internals

`python3 scripts/lint doc --write` writes this file from `scripts/lint/boundary/rules.py`, `scripts/lint/boundary/doc.py`, the `#include` lines of `CarpetX/src/` and the registrations in `CarpetX/src/flesh/startup.cxx`; `scripts/lint/boundary/test_rules.py` fails when the file differs from what it writes, and the edit hook refuses an edit to it. A client thorn sees none of this: it reaches CarpetX only through the headers `agent_docs/carpetx-public-api.md` describes. `python3 scripts/lint check --help` prints what every rule named below holds.

## The stack

```mermaid
flowchart TB
  subgraph door["driver() allowed"]
    flesh["flesh/ no header entry"]
    api["api/"]
    schedule["schedule/"]
    io["io/ records, codec"]
    ops["ops/"]
    regrid["regrid/ no header"]
    comm["comm/ directions"]
    hierarchy["hierarchy/ validity, binding, varindex, model"]
  end
  subgraph nodoor["no driver()"]
    traversal["traversal/ traversal"]
    util["util/"]
  end
  flesh --> api --> schedule --> io --> ops --> regrid --> comm --> hierarchy --> traversal --> util
  hierarchy -. "CactusAmrCore" .-> regrid
  hierarchy -. "apply_boundary_conditions, get_interpolator" .-> comm
%% a solid edge is may include: a layer includes the layer below it and, through it, every layer further down, which the layer rule holds
%% the two boxes are the door rule: driver() is called in the upper box, never in the lower
%% a dashed edge is a declaration whose body is in the higher layer, the one way a call reaches up, at link time
%% a node names its layer, then the rules whose home is in it, and no header where nothing can include it
```

## The includes

```mermaid
flowchart TB
  flesh["flesh/ no header"]
  api["api/"]
  schedule["schedule/"]
  io["io/"]
  ops["ops/"]
  regrid["regrid/ no header"]
  comm["comm/"]
  flesh -- "schedule.hxx" --> schedule
  flesh -- "checkpoint.hxx, io.hxx" --> io
  flesh -- "interp.hxx, linsolve.hxx" --> ops
  flesh -- "comm.hxx, reflux.hxx, subcycling.hxx" --> comm
  api -- "schedule.hxx" --> schedule
  api -- "comm.hxx, reflux.hxx, subcycling.hxx" --> comm
  schedule -- "io.hxx" --> io
  schedule -- "comm.hxx, reflux.hxx, subcycling.hxx" --> comm
  io -- "reduction.hxx" --> ops
  io -- "subcycling.hxx" --> comm
  regrid -- "buffers.hxx, fillpatch.hxx" --> comm
%% what each layer includes today, read from its #include lines, with the headers on the edges
%% util/, traversal/ and hierarchy/ are not drawn: every layer reads them
%% an edge missing here is a coupling CarpetX does not have
```

## Who decides

```mermaid
flowchart LR
  cactus["Cactus flesh"]
  amrex["AMReX"]
  flesh["flesh/"]
  api["api/"]
  schedule["schedule/"]
  io["io/"]
  ops["ops/"]
  regrid["regrid/"]
  comm["comm/"]
  hierarchy["hierarchy/"]
  cactus -- "extern C in aliased.cxx, scheduled.cxx, startup.cxx" --> flesh
  cactus -- "SetupGH, InitGH, ScheduleTraverseGH, MyProc, nProcs, Exit, Abort, Barrier, CarpetX_InterpGridArrays, ArrayGroupSizeB, GroupDynamicData" --> flesh
  cactus -- "Initialise, Evolve, Shutdown, EnableGroupStorage, DisableGroupStorage, GroupStorageIncrease, GroupStorageDecrease, QueryGroupStorageB" --> schedule
  cactus -- "OutputGH" --> io
  cactus -- "InterpLocalUniform" --> ops
  cactus -- "SyncGroupsOnActiveLevels" --> comm
  flesh -- "CCTK_ScheduleTraverse with CallFunction" --> cactus
  api -- "CCTK_ScheduleTraverse with CallFunction" --> cactus
  hierarchy -- "regrid" --> amrex
  amrex -- "ErrorEst, MakeNewLevelFromScratch, MakeNewLevelFromCoarse, RemakeLevel, ClearLevel" --> regrid
%% an edge from an outsider is a function the layer defines and the outsider calls: the registrations of CarpetX/src/flesh/startup.cxx, the extern C entries, and the AMReX callbacks of CactusAmrCore
%% an edge to an outsider is a call that comes back through an edge from it, so the outsider decides when the lower layer works
%% a call from a layer to a higher one goes through an outsider or, at link time, through a dashed edge of the stack
```

## What each layer owns

New code goes in the layer that owns what it works on, and a lower layer never decides for a higher one. Each row says what its layer owns and what it never decides.

- `flesh/`: every `extern "C"`. It owns no data: an entry converts the flesh's arguments and calls the layer that does the work.
- `api/`: the bodies of the public headers. An operation names itself with `operation_scope_t`, checks its arguments, and calls the layer that owns the work.
- `schedule/`: the driver's loop and clocks, and `CallFunction`, which every routine runs through. How a level is stored, synced or written is decided below it, and its header declares only what `flesh/` registers and `api/` hands to the flesh.
- `io/`: output, checkpoint, recovery and the file reader's input. When to do each is decided by the flesh's bins and by `schedule/`.
- `ops/`: operations on grid data that other thorns ask for. Their flesh entry points are in `flesh/`.
- `regrid/`: the bodies of the AMReX callbacks, never data-model code: what a level holds and how it is built belongs to `hierarchy/`.
- `comm/`: moving data between ranks, between levels, and into ghost and boundary zones. It never decides when: `schedule/` and `api/` ask. A body names no `cGH` but the global one, and takes the iteration from `current_iteration()`.
- `hierarchy/`: the data model and the hierarchy's views of itself. What a checkpoint carries is a field of `LevelData` or `GroupData`; what `comm/` builds on a level is in the slot `buffers`; a change to the hierarchy that another layer needs is a member of `hierarchy_writer`.
- `traversal/`: the run's phase, one slot per fact with one writer. A fact about the phase that `comm/`, `ops/` or `io/` must read without depending on `schedule/` is a slot here; a fact about the grid is not.
- `util/`: leaves that know nothing of the grid. Nothing here names the hierarchy, a level, a window or the scheduler.
