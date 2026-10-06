# How to generate reference output for a new test

Tests are auto-discovered from `<Thorn>/test/*.par` and compared against a checked-in reference directory of the same basename (`<Thorn>/test/<test>/`). To create that reference for a new `.par`, run the built executable on the par file once and copy the resulting output dir into the test tree.

Everything runs on the host: `$CACTUSX` (pre-set in the host shell) is the Cactus tree, this repo is symlinked into it at `$CACTUSX/arrangements/CarpetX`, and the executable is `$CACTUSX/exe/cactus_carpetx`.

```bash
# Run the par file; with IO::out_dir = $parfile the output lands in ./<test>/
cd <repo>/<Thorn>/test &&
  "$CACTUSX/exe/cactus_carpetx" <test>.par
```

## Recover tests

A recover test is a pair: `checkpoint-<name>.par` writes a checkpoint, and `recover-<name>.par` restarts from it. The pair asserts that recovery is transparent, i.e. that the restarted run reproduces the uninterrupted one from the recovered iteration on:

- The checkpoint half is also the uninterrupted run: its `Cactus::cctk_itlast` is the recover half's last iteration, not the checkpoint iteration.
- The recover half sets `CarpetX::out_initial_data = yes`, so the recovered iteration itself is output.
- Both halves activate the same thorns and output the same variables.
- The recover half's reference files are copies of the checkpoint half's: for every iteration both halves share, starting with the recovered iteration, each file in `<Thorn>/test/recover-<name>/` is byte-for-byte the file of the same name in `<Thorn>/test/checkpoint-<name>/`.

Generate the checkpoint half first, as above. Then copy its checkpoint (`<Thorn>/test/checkpoint-<name>/checkpoints/checkpoint.chkpt.itNNNNNNNN.bp5/`) into the directory the recover par's `IO::recover_dir` names (e.g. `<Thorn>/test/checkpoints-<name>/`), deleting `profiling.json` from it.

Do **not** run the recover par from `<Thorn>/test`. Its `IO::recover_dir` is relative to where the testsuite runs it, `$CACTUSX/TEST/<config>/<Thorn>/` (e.g. `"../../../arrangements/CarpetX/<Thorn>/test/checkpoints-<name>"`). From `<Thorn>/test` that path resolves to nothing, and the run silently starts fresh instead of recovering. Its output looks plausible but is wrong as a reference. Run it from a scratch directory that mirrors the testsuite's depth instead:

```bash
S=<scratch-dir>
mkdir -p "$S/arrangements" "$S/TEST/carpetx/<Thorn>" &&
  ln -sfn <repo> "$S/arrangements/CarpetX" &&
  cd "$S/TEST/carpetx/<Thorn>" &&
  "$CACTUSX/exe/cactus_carpetx" <repo>/<Thorn>/test/recover-<name>.par
```

Check the log before using the output. It must contain `Recovering parameters from checkpoint file "..." iteration N` with the expected `N`. If it says `Not recovering parameters:` instead, the path is wrong. Do not check in the recover run's own output. Use it only to learn the file set (which files the recover half writes, from the recovered iteration on), then fill `<Thorn>/test/recover-<name>/` with copies of those files from the checkpoint half's reference directory and confirm the copy:

```bash
cd <repo>/<Thorn>/test &&
  for f in "$S/TEST/carpetx/<Thorn>/recover-<name>"/*.tsv; do
    cp "checkpoint-<name>/$(basename "$f")" "recover-<name>/"
  done &&
  for f in recover-<name>/*.tsv; do
    cmp "$f" "checkpoint-<name>/$(basename "$f")"
  done
```

The second loop must print nothing. A file the recover run writes that the checkpoint half's reference lacks means the two halves do not output the same variables or iterations; fix the par files instead of checking in the stray file. If the testsuite then reports differences for the recover half, recovery is not transparent: that is a bug to report, not a reason to regenerate the reference from the recover run.

When a change adds a checkpointed group to the pair (e.g. a newly activated thorn), the checked-in checkpoint under `<Thorn>/test/checkpoints-<name>/` no longer carries every group the recover half reads; regenerate it from the checkpoint half as described above.

## Curating the reference

Then curate the generated `<Thorn>/test/<test>/` directory to mirror an existing reference of the same family before checking it in:

- Match the **file set** of a sibling reference dir (e.g. `checkpoint-openpmd/`): omit `performance.yaml`, the `checkpoints/` directory and any `.bp5` output the sibling does not keep. A recover reference keeps the recovered iteration and every later one, as copies of the checkpoint half's files (see "Recover tests" above).
- For physics-invariant changes (e.g. a re-decomposition), `diff` the new `*.tsv` against the sibling reference and confirm they are bit-for-bit identical.
- Delete `profiling.json` from every `.bp5` directory before checking it in (checkpoint directories under `<Thorn>/test/checkpoints*/` as well as `.bp5` output inside a reference dir). ADIOS2's BP5 engine writes it unconditionally, it holds wall-clock timings that differ on every run, and nothing reads it on recovery, so it only adds a spurious diff to every regeneration.

Finally run `./agent_scripts/test.sh` to confirm the new test is discovered and passes (`Number failed -> 0`).
