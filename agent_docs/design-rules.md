# Design rules

CarpetX is arranged so that AI coding agents and humans can work in it safely and efficiently. Agents copy existing patterns and take the shortest path to a working change, so the code makes that path the correct one.

- Architecture: the layout makes it obvious where new code belongs.
- Boundaries: a component has a defined interface, with few cross-cutting dependencies.
- Types: an invariant is encoded in the type system, so that misuse fails at compile time rather than at runtime.
- The easy path is the correct one: a change that copies an existing pattern is right.
- A class of bugs is made impossible by code, not by convention or review, so that it never has to be thought about again.

The files beside this one say how the repository realizes them. `carpetx-public-api.md` is the interface a thorn sees of CarpetX, and its index and handle types are where the invariants live. `carpetx-internals.md` is the map of `CarpetX/src/` and says where new code belongs and which lines hold between its layers. `code-comments.md` is the one rule that applies to every thorn in the repository. `python3 scripts/lint` holds the lines each of them draws, and `python3 scripts/lint --help` says how the hooks and CI run it.
