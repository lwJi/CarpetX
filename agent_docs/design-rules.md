# Design rules

CarpetX is arranged so that AI coding agents and humans can work in it safely and efficiently. Agents copy existing patterns and take the shortest path to a working change, so the code makes that path the correct one.

- Data: a design is its data structures and their relationships, what a thing holds, who writes it and who reads it; the code follows from them.
- Abstraction: a new semantic level at which one can be exact, not a way to be vague. A thing a layer cannot name exactly is not named there.
- Architecture: the layout makes it obvious where new code belongs.
- Boundaries: a component has a defined interface, with few cross-cutting dependencies.
- Types: an invariant is encoded in the type system, so that misuse fails at compile time rather than at runtime.
- The easy path is the correct one: a change that copies an existing pattern is right.
- A class of bugs is made impossible by code, not by convention or review, so that it never has to be thought about again.
