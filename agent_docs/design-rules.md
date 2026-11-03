# Design rules

CarpetX is arranged so that AI coding agents and humans can work in it safely and efficiently. Agents copy existing patterns and take the shortest path to a working change, so the code makes that path the correct one.

- Architecture: the layout makes it obvious where new code belongs.
- Boundaries: a component has a defined interface, with few cross-cutting dependencies.
- Data first: a change starts from the data structures and their relationships, what each holds, who writes it and who reads it, and the code follows; the map in `agent_docs/carpetx-internals.md` is of the data and its owners, not of the functions.
- Types: an invariant is encoded in the type system, so that misuse fails at compile time rather than at runtime.
- The easy path is the correct one: a change that copies an existing pattern is right.
- A class of bugs is made impossible by code, not by convention or review, so that it never has to be thought about again.
- The obvious version first: a change is written for the reader, not the profiler. A faster version is its own commit, with the measurement that justifies it and how to repeat it in the message; a change that trades clarity for speed without one is refused.
