# Design rules

CarpetX is arranged so that whoever copies an existing pattern and takes the shortest path to a working change makes the right one. A rule is held by a check or by the compiler, never by convention; a line held by review says so. `python3 scripts/lint check --help` prints each rule as the code holds it, and `carpetx-internals.md` is the map.

## What the tree is

- Layers: a file includes its own layer or a lower one, and new code goes in the layer that owns what it works on. Held by a check over the includes and one over the objects a build made.
- A public interface: another thorn reaches CarpetX only through its public headers. Held by a check.
- Ownership: a piece of state has one file that writes it, and every other site calls that file. Held by a check, one rule per owner.
- Vocabulary: a layer or a handle type names one thing, so that code can say it exactly. Held by a check for the handles and by the doc test for the layers.
- Invariants are types, so misuse fails at compile time. Held by the compiler: the hierarchy is reached const, and the mutable path is private.
- A tolerated violation is listed in the check that would refuse it, and the list only shrinks. Copying a listed reach is wrong. Held by review.
- Explanations are in code or in the commit message, never in a comment. Held by a check; `code-comments.md` is the rule.

## What a change does

- Data first: a change is designed from its data structures and their owners, and the code follows; a new owner of state is a new rule. Held by review.
- Plain first: the plainest form, and a change made for speed says in its commit message on which machine the timers found the hot spot and what they showed before and after. Held by review.
- A new abstraction is put where its check can see it: a layer in the layer list, an owner in the rules, its meaning in the map. Held by the doc test for a layer, by review for the rest.
