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

## Commit & Pull Request Guidelines

Recent history favors short subject lines with a thorn prefix, for example `ODESolvers: update comments` or `CarpetX: add par poison_undefined_arrays`. Keep commits focused on one thorn or one behavior change. PRs should explain the scientific or runtime impact, list the tests run, and link the relevant issue or discussion. Include plots or output snippets when a change affects diagnostics, IO, or visible simulation results.

## Further Reading

- `.github/workflows/ci.yml` — CI matrix and environment variable combinations
