# CarpetX

CarpetX is a Cactus driver based on AMReX for block-structured adaptive mesh refinement (AMR), targeting the Einstein Toolkit for numerical relativity simulations. It is written in C++ and supports both CPU and GPU accelerators.

## Project Structure & Module Organization

This repository is organized as Cactus thorns. Each top-level directory such as `CarpetX/`, `ODESolvers/`, `TestSubcyclingCC/`, or `TestSubcyclingMC2/` is a thorn with Cactus metadata files like `interface.ccl`, `param.ccl`, `schedule.ccl`, and implementation under `src/`. Tests usually live beside the thorn in `test/` and use parameter files plus checked-in reference output, for example `TestSubcyclingMC2/test/gaussian.par` and `TestSubcyclingMC2/test/gaussian/*.tsv`.

Do **not** read reference test data files (e.g. `*/test/*/*.tsv`, `*/test/*/*.h5`, or other large numerical output) unless the user explicitly asks for it. They are large, binary-like, and rarely informative for code-level reasoning. Inspect the `.par` parameter file and the thorn source instead.

CarpetX's own sources are ten layers, one directory each under `CarpetX/src/`, and a file includes only its own layer or a lower one. Before adding code to CarpetX, read `agent_docs/design-rules.md`, the rules the thorn is arranged to, and `agent_docs/carpetx-internals.md`, which says what each layer owns and which `boundary` rules of `scripts/lint` hold the lines between them; the edit hook enforces them, and `python3 scripts/lint check --only boundary` shows what it would report. Other thorns reach CarpetX only through `<carpetx.hxx>` and `<carpetx_rk.hxx>`, described in `agent_docs/carpetx-public-api.md`.

## Build & Test

```bash
./agent_scripts/build.sh
./agent_scripts/test.sh
```

An incremental build takes ~30 s and the testsuite ~1 min, but a from-clean build takes ~5 min — give `build.sh` a 10-minute timeout.

Where the logs and build artifacts land, and how to generate reference output for a new test, are in `agent_scripts/README.md`.

### Inside an sbx sandbox

The same two scripts work unchanged inside a Docker Sandboxes (sbx) microVM created from the committed template (recipe and full instructions: `agent_scripts/sandbox/README.md`):

```bash
sbx run -t lwji/sandbox-templates:claude-carpetx --name claude-CarpetX claude . ../amrex:ro   # on the host, from the repo root
agent_scripts/sandbox/setup.sh                                     # in-sandbox, once per sandbox (~30 s)
```

The first in-sandbox `build.sh` configures and compiles from scratch (~13 min — give it a 30-minute timeout); the named sandbox persists the build across stop/start, so later builds are incremental (~10 s) and the testsuite takes ~25 s.

## Code comments

Comments in thorn files can only be deleted. Don't add a comment line or change one, and when your change edits the code under a comment (its paragraph, or the whole block that code opens), delete the whole comment. `python3 scripts/lint check --since origin/dev` shows what the hooks and the `Lint` CI workflow would report, and `python3 scripts/lint --help` says how they run it. The full rule, including what counts as a thorn file and the few allowed comment lines, is in `agent_docs/code-comments.md`.

A comment already in a file is a claim to check against the code, never the reason for a change. When your change makes a comment false that the check can't see, such as one on a declaration in a header, delete it too. Explain in code (an `assert`, a name, a `CCTK_VERROR`) or in your commit message, never in a comment; `git log -S '<words>'` finds what deleted comments said.

## Commit & Pull Request Guidelines

Recent history favors short subject lines with a thorn prefix, for example `ODESolvers: check more of the RK method table at compile time` or `CarpetX: add par poison_undefined_arrays`. Keep commits focused on one thorn or one behavior change. PRs should explain the scientific or runtime impact, list the tests run, and link the relevant issue or discussion. Include plots or output snippets when a change affects diagnostics, IO, or visible simulation results.

## Further Reading

- `.github/workflows/ci.yml` — CI matrix and environment variable combinations
