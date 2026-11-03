# Design rules

CarpetX is arranged so that whoever copies an existing pattern and takes the shortest path to a working change makes the right one. A rule is held by lint, by a test or by the compiler where one of them can; a line held by review says so. `python3 scripts/lint check --help` and `python3 scripts/lint symbols --help` print each lint rule as the code holds it, and `carpetx-internals.md` is the map, tested against the code.

## What the tree is

- Layers: a file includes its own layer or a lower one, and new code goes in the layer that owns what it works on. Lint, over the includes and over the objects a build made.
- A public interface: another thorn reaches CarpetX only through its public headers. Lint.
- Boundaries: a module inside a layer takes a narrow interface, the record and not the hierarchy. Lint, one rule per interface.
- Ownership: a piece of state has one file that writes it, and every other site calls that file. Lint, one rule per owner.
- Vocabulary: a layer or a handle type names one thing, so that code can say it exactly. Lint for the handles, since C++ accepts a bare `int` where a handle is meant; the doc test for the layer list; review for what a layer means.
- Invariants are types where the compiler can hold them: the hierarchy is reached const and its writer is private. Compiler.
- A tolerated violation is listed in the rule that would refuse it, and the list only shrinks; copying a listed reach is wrong. Review.
- Explanations are in code or in the commit message, never in a comment. Lint; `code-comments.md` is the rule.

## What a change does

- Data first: a change is designed from its data structures and their owners, and the code follows. Review.
- Plain first: the plainest form, and a change made for speed says in its commit message on which machine the timers found the hot spot and what they showed before and after, because development is on a laptop and production on a cluster, so a laptop number cannot justify a cluster change. Review.
- A new layer, owner or interface goes where a rule can see it; the map says how under "Placing code".
