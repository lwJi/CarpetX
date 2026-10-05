# How to generate reference output for a new test

Tests are auto-discovered from `<Thorn>/test/*.par` and compared against a checked-in reference directory of the same basename (`<Thorn>/test/<test>/`). To create that reference for a new `.par`, run the built executable on the par file once and copy the resulting output dir into the test tree.

Everything runs on the host: `$CACTUSX` (pre-set in the host shell) is the Cactus tree, this repo is symlinked into it at `$CACTUSX/arrangements/CarpetX`, and the executable is `$CACTUSX/exe/cactus_carpetx`.

```bash
# Run the par file; with IO::out_dir = $parfile the output lands in ./<test>/
cd <repo>/<Thorn>/test &&
  "$CACTUSX/exe/cactus_carpetx" <test>.par
```

## Recover tests

A recover test is a pair: `checkpoint-<name>.par` writes a checkpoint, and `recover-<name>.par` restarts from it. Generate the checkpoint half first, as above. Then copy its checkpoint (`<Thorn>/test/checkpoint-<name>/checkpoints/checkpoint.chkpt.itNNNNNNNN.bp5/`) into the directory the recover par's `IO::recover_dir` names (e.g. `<Thorn>/test/checkpoints-<name>/`), deleting `profiling.json` from it.

Do **not** run the recover par from `<Thorn>/test`. Its `IO::recover_dir` is relative to where the testsuite runs it, `$CACTUSX/TEST/<config>/<Thorn>/` (e.g. `"../../../arrangements/CarpetX/<Thorn>/test/checkpoints-<name>"`). From `<Thorn>/test` that path resolves to nothing, and the run silently starts fresh instead of recovering. Its output looks plausible but is wrong as a reference. Run it from a scratch directory that mirrors the testsuite's depth instead:

```bash
S=<scratch-dir>
mkdir -p "$S/arrangements" "$S/TEST/carpetx/<Thorn>" &&
  ln -sfn <repo> "$S/arrangements/CarpetX" &&
  cd "$S/TEST/carpetx/<Thorn>" &&
  "$CACTUSX/exe/cactus_carpetx" <repo>/<Thorn>/test/recover-<name>.par
```

Check the log before using the output. It must contain `Recovering parameters from checkpoint file "..." iteration N` with the expected `N`. If it says `Not recovering parameters:` instead, the path is wrong. Then copy `$S/TEST/carpetx/<Thorn>/recover-<name>/` to `<Thorn>/test/recover-<name>/` and curate it as below.

## Curating the reference

Then curate the generated `<Thorn>/test/<test>/` directory to mirror an existing reference of the same family before checking it in:

- Match the **file set** of a sibling reference dir (e.g. `recover-openpmd/`). Recovery references typically omit the recovered iteration's output (`it000000`) and the `it00000000.bp5` / `performance.yaml` files — keep only the iterations the sibling keeps.
- For physics-invariant changes (e.g. a re-decomposition), `diff` the new `*.tsv` against the sibling reference and confirm they are bit-for-bit identical.
- Delete `profiling.json` from every `.bp5` directory before checking it in (checkpoint directories under `<Thorn>/test/checkpoints*/` as well as `.bp5` output inside a reference dir). ADIOS2's BP5 engine writes it unconditionally, it holds wall-clock timings that differ on every run, and nothing reads it on recovery, so it only adds a spurious diff to every regeneration.

Finally run `./agent_scripts/test.sh` to confirm the new test is discovered and passes (`Number failed -> 0`).
