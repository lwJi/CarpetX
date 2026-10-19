# CarpetX

CarpetX is a Cactus driver based on AMReX for block-structured adaptive mesh refinement (AMR), targeting the Einstein Toolkit for numerical relativity simulations. It is written in C++ and supports both CPU and GPU accelerators.

## Project Structure & Module Organization

This repository is organized as Cactus thorns. Each top-level directory such as `CarpetX/`, `ODESolvers/`, `TestSubcyclingCC/`, or `TestSubcyclingMC2/` is a thorn with Cactus metadata files like `interface.ccl`, `param.ccl`, `schedule.ccl`, and implementation under `src/`. Tests usually live beside the thorn in `test/` and use parameter files plus checked-in reference output, for example `TestSubcyclingMC2/test/gaussian.par` and `TestSubcyclingMC2/test/gaussian/*.tsv`.

Do **not** read reference test data files (e.g. `*/test/*/*.tsv`, `*/test/*/*.h5`, or other large numerical output) unless the user explicitly asks for it. They are large, binary-like, and rarely informative for code-level reasoning. Inspect the `.par` parameter file and the thorn source instead.

## Build & Test

```bash
./agent_scripts/build.sh
./agent_scripts/test.sh
```

An incremental build takes ~30 s and the testsuite ~1 min, but a from-clean build takes ~5 min — give `build.sh` a 10-minute timeout.

To access test logs and build artifacts, read `./agent_docs/accessing-logs-and-artifacts.md`. To generate reference output for a new test, read `./agent_docs/generating-reference-output.md`.

### Inside an sbx sandbox

The same two scripts work unchanged inside a Docker Sandboxes (sbx) microVM created from the committed template (recipe and full instructions: `agent_scripts/sandbox/README.md`):

```bash
sbx run -t lwji/sandbox-templates:claude-carpetx --name claude-CarpetX claude . ../amrex:ro   # on the host, from the repo root
agent_scripts/sandbox/setup.sh                                     # in-sandbox, once per sandbox (~30 s)
```

The first in-sandbox `build.sh` configures and compiles from scratch (~13 min — give it a 30-minute timeout); the named sandbox persists the build across stop/start, so later builds are incremental (~10 s) and the testsuite takes ~25 s.

## Code comments

Thorn files contain no comments. A thorn is a top-level directory with an `interface.ccl`, and its files are the C, C++ and Fortran sources under `src/`, every `.ccl`, every `.par` and `.par-DISABLED`, and `make.code.defn`. Repo tooling (`scripts/`, `agent_scripts/`, `docker/`, workflows, option lists) keeps its comments, and so does `BenchX/`.

A comment is allowed only if all of it is one of these:

- A tool marker: a namespace closer such as `} // namespace CarpetX`, a bare `//`, or `// clang-format off` / `// clang-format on`. A bare `//` after code, or on a line of its own between two code lines, steers clang-format's line breaks; on a line of its own between comments, it separates a citation from the next comment.
- A citation that consists of a single URL, arXiv ID or DOI, optionally introduced by `See`, `From`, `Taken from` or `cf.` and followed by an arXiv category and equation numbers, for example `// See arXiv:1111.2177 [gr-qc], (74-75)` or `// doi:10.1016/0021-9991(88)90177-5`. Write an author-year citation as its DOI.

Everything else is banned, however short: prose, TODO and FIXME, `/*arg=*/` labels, line references into other code bases, commented-out code and `.par` settings, `#if 0` / `#if 1` blocks, and compile-time messages (`#warning`, `#pragma message`). There is no per-line or per-file exception. CCL description strings and runtime messages are not comments, but don't move comment prose into them.

Explain in code (an `assert`, a name, a `CCTK_VERROR`) or in your commit message, never in a comment.

When the code doesn't explain itself, read its history. Start with the thorn's `<Thorn>: remove comments` commit, whose message records what the removed comments said that the code doesn't. Then use `git log -L <start>,<end>:<file>` for the lines you are changing, and `git blame --ignore-revs-file .git-blame-ignore-revs <file>`, which skips the commits that only removed comments. History describes the code as it was at that commit, so check every claim against the current code before relying on it.

A comment-removal commit is the output of `python3 scripts/comments strip <Thorn>` and nothing else, and its SHA goes on `.git-blame-ignore-revs`. The `no-comments` check runs `python3 scripts/comments verify <SHA>^ <SHA>` on each SHA a PR adds there, to prove the commit changed only comments.

## Commit & Pull Request Guidelines

Recent history favors short subject lines with a thorn prefix, for example `ODESolvers: check more of the RK method table at compile time` or `CarpetX: add par poison_undefined_arrays`. Keep commits focused on one thorn or one behavior change. PRs should explain the scientific or runtime impact, list the tests run, and link the relevant issue or discussion. Include plots or output snippets when a change affects diagnostics, IO, or visible simulation results.

## Further Reading

- `.github/workflows/ci.yml` — CI matrix and environment variable combinations
