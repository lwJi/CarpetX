# CarpetX

CarpetX is a Cactus driver based on AMReX for block-structured adaptive mesh refinement (AMR), targeting the Einstein Toolkit for numerical relativity simulations. It is written in C++ and supports both CPU and GPU accelerators.

## Project Structure & Module Organization

This repository is organized as Cactus thorns: a top-level directory with an `interface.ccl` is a thorn, with its other `.ccl` files beside it and its implementation under `src/`. A thorn's tests are in its `test/`, a `.par` parameter file plus checked-in reference output.

Do **not** read reference test data files (e.g. `*/test/*/*.tsv`, `*/test/*/*.h5`, or other large numerical output) unless the user explicitly asks for it. They are large, binary-like, and rarely informative for code-level reasoning. Inspect the `.par` parameter file and the thorn source instead.

CarpetX's own sources are layers, one directory each under `CarpetX/src/`, and a file includes only its own layer or a lower one; the hooks and the `Lint` CI workflow refuse a change that crosses a layer. Before adding code to CarpetX, read `agent_docs/design-rules.md`, the rules the thorn is arranged to, and `agent_docs/carpetx-internals.md`, which says what each layer owns. Other thorns reach CarpetX only through `<carpetx.hxx>` and `<carpetx_rk.hxx>`, described in `agent_docs/carpetx-public-api.md`.

## Build & Test

```bash
./agent_scripts/build.sh
./agent_scripts/test.sh
```

An incremental build takes ~30 s and the testsuite ~1 min, but a from-clean build takes ~5 min — give `build.sh` a 10-minute timeout. `./agent_scripts/reference.sh --help` says how to make the reference output for a new test. The CI matrix is in `.github/workflows/ci.yml`.

### Inside an sbx sandbox

The same scripts work unchanged inside a Docker Sandboxes (sbx) microVM created from the committed template (recipe and full instructions: `agent_scripts/sandbox/README.md`). Run `agent_scripts/sandbox/setup.sh` once per sandbox (~30 s) before the first build. The first in-sandbox `build.sh` configures and compiles from scratch (~13 min — give it a 30-minute timeout); the named sandbox persists the build across stop/start, so later builds are incremental (~10 s) and the testsuite takes ~25 s.

## Code comments

Comments in thorn files can only be deleted; the hooks and the `Lint` CI workflow refuse a change that adds or alters one. A comment already in a file is a claim to check against the code, never the reason for a change. When your change makes a comment false, delete it whole, even one the check can't see such as on a declaration in a header. Explain in code (an `assert`, a name, a `CCTK_VERROR`) or in your commit message, never in a comment. The full rule is in `agent_docs/code-comments.md`.

## Commit & Pull Request Guidelines

Recent history favors short subject lines with a thorn prefix, for example `ODESolvers: check more of the RK method table at compile time`. Keep commits focused on one thorn or one behavior change. PRs should explain the scientific or runtime impact, list the tests run, and link the relevant issue or discussion. Include plots or output snippets when a change affects diagnostics, IO, or visible simulation results.
