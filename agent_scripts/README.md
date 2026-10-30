# Building and testing on the host

`build.sh` and `test.sh` build and test directly on the host in the Cactus tree at `$CACTUSX` (pre-set in the host shell): this repo is symlinked into it at `$CACTUSX/arrangements/CarpetX`, and the executable they build and run is `$CACTUSX/exe/cactus_carpetx`. Inside an sbx sandbox (`sandbox/README.md`) everything below applies unchanged at the same `$CACTUSX`-relative locations, with `CACTUSX=/home/agent/cactus/Cactus` baked into the sandbox environment.

## Logs and artifacts

Per-test logs: `$CACTUSX/TEST/carpetx/<Thorn>/<test>.log`, with value-level differences in `<test>.diffs` (testsuite summary: `$CACTUSX/TEST/carpetx/summary.log`; when tests fail it ends with a `Tests failed:` section).
Full run logs from the scripts: `$CACTUSX/last-build.log` and `$CACTUSX/last-test.log` (the scripts print bounded excerpts on failure; Grep/Read these for anything beyond the excerpt).
Build artifacts: `$CACTUSX/configs/carpetx/`.
Executable: `$CACTUSX/exe/cactus_carpetx`.

## Generating reference output for a new test

Tests are auto-discovered from `<Thorn>/test/*.par` and compared against a checked-in reference directory of the same basename (`<Thorn>/test/<test>/`). To create that reference for a new `.par`, run the built executable on the par file once and copy the resulting output dir into the test tree.

```bash
# Run the par file; with IO::out_dir = $parfile the output lands in ./<test>/
cd <repo>/<Thorn>/test &&
  "$CACTUSX/exe/cactus_carpetx" <test>.par
```

Then curate the generated `<Thorn>/test/<test>/` directory to mirror an existing reference of the same family before checking it in:

- Match the **file set** of a sibling reference dir (e.g. `recover-openpmd/`). Recovery references typically omit the recovered iteration's output (`it000000`) and the `it00000000.bp5` / `performance.yaml` files — keep only the iterations the sibling keeps.
- For physics-invariant changes (e.g. a re-decomposition), `diff` the new `*.tsv` against the sibling reference and confirm they are bit-for-bit identical.
- Delete `profiling.json` from every `.bp5` directory before checking it in (checkpoint directories under `<Thorn>/test/checkpoints*/` as well as `.bp5` output inside a reference dir). ADIOS2's BP5 engine writes it unconditionally, it holds wall-clock timings that differ on every run, and nothing reads it on recovery, so it only adds a spurious diff to every regeneration.

Finally run `./test.sh` to confirm the new test is discovered and passes (`Number failed -> 0`).
