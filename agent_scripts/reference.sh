#!/bin/bash

usage() {
  cat <<'EOF'
usage: agent_scripts/reference.sh <Thorn>/test/<name>.par [<sibling>]

Regenerates the checked-in reference output <Thorn>/test/<name>/ for one
test parfile:

  1. runs that one test through the testsuite (make carpetx-testsuite
     restricted with CCTK_TESTSUITE_RUN_TESTS), so the run is the one
     agent_scripts/test.sh and CI make: the same directory, which relative
     paths in the parfile (IO::recover_dir above all) resolve from, the
     same MPI launcher and rank count, per-test NPROCS from test.ccl
     included. <Thorn>/test/<name>/ must exist for the testsuite to
     discover the parfile, so it is created first when missing; the
     testsuite's comparison against it is ignored, and only a failed
     Cactus run stops the script, with the reference left as it was;
  2. replaces <Thorn>/test/<name>/ with the run's files whose extension is
     in the thorn's EXTENSIONS (test.ccl, default tsv), the files the
     testsuite compares, and lists the files it did not copy; a file the
     run no longer writes therefore shows up as a deletion in git status;
  3. when the run leaves <name>/checkpoints/ and <name> is
     checkpoint<suffix>, replaces <Thorn>/test/checkpoints<suffix>/ with it,
     minus every profiling.json: that directory is the recovery input of
     recover<suffix>.par;
  4. with <sibling>, a reference directory of the same family (a name under
     <Thorn>/test/ or a path), prints diff -rq of the new reference against
     it: "Only in" lines are the file-set difference, "differ" lines the
     files present in both whose contents differ.

What the script cannot decide:

  - A recovery reference keeps the iterations its sibling keeps; compare
    the file-set difference against the sibling and delete what the
    sibling omits.
  - After a physics-invariant change (a re-decomposition, a refactor) the
    *.tsv must be identical to the sibling's; after any other change they
    are expected to differ, and the difference is what you are checking in.
  - The four TestOutput references that carry .bp5 directories
    (output-openpmd, output-openpmd-2d, recover-openpmd,
    recover-openpmd-rechop) are not compared by the testsuite; regenerating
    them shows those directories as deletions, to restore with git checkout
    or to accept.
  - ./agent_scripts/test.sh must pass before the directory is checked in.

profiling.json and performance.yaml hold per-run timings and are ignored
by .gitignore; nothing under <Thorn>/test/ may carry them.
EOF
}

case "${1:-}" in
  -h|--help) usage; exit 0 ;;
  "") usage >&2; exit 1 ;;
esac

if [ -z "${CACTUSX:-}" ]; then
  echo "✗ missing required env var: CACTUSX (the Cactus tree with this repo symlinked at arrangements/CarpetX)" >&2
  exit 1
fi
if [ ! -x "$CACTUSX/exe/cactus_carpetx" ]; then
  echo "✗ missing executable: $CACTUSX/exe/cactus_carpetx (run ./agent_scripts/build.sh)" >&2
  exit 1
fi

parfile="$1"
sibling="${2:-}"
if [ ! -f "$parfile" ]; then
  echo "✗ no such parfile: $parfile" >&2
  exit 1
fi
parfile="$(cd "$(dirname "$parfile")" && pwd)/$(basename "$parfile")"
testdir="$(dirname "$parfile")"
thorn="$(basename "$(dirname "$testdir")")"
name="$(basename "$parfile" .par)"
if [ "$(basename "$testdir")" != test ] || [ "$name.par" != "$(basename "$parfile")" ]; then
  echo "✗ expected <Thorn>/test/<name>.par, got $1" >&2
  exit 1
fi

extensions="$(sed -n -E 's/^[[:space:]]*EXTENSIONS[[:space:]]+(.*)$/\1/Ip' "$testdir/test.ccl" 2>/dev/null | tr -s '[:space:]' ' ')"
extensions="${extensions:-tsv }"

refdir="$testdir/$name"
rundir="$CACTUSX/TEST/carpetx/$thorn"
runout="$rundir/$name"
log="$CACTUSX/last-reference.log"
mkdir -p "$refdir"
if ! (cd "$CACTUSX" && CCTK_TESTSUITE_RUN_TESTS="$thorn/$name" make carpetx-testsuite PROMPT=no) > "$log" 2>&1 ||
   grep -q 'Cactus exited with error code' "$log"; then
  echo "--- last 40 lines ---"
  tail -40 "$log"
  echo "✗ $name.par failed — testsuite log: $log, run log: $rundir/$name.log" >&2
  exit 1
fi
if ! grep -q "^  Test $thorn: $name " "$log"; then
  echo "✗ the testsuite did not run $thorn/$name; is $thorn in $CACTUSX/configs/carpetx/ThornList? — log: $log" >&2
  exit 1
fi
if [ ! -d "$runout" ]; then
  echo "✗ the run left no $runout; does $name.par set IO::out_dir = \$parfile? — log: $log" >&2
  exit 1
fi

kept=()
skipped=()
while IFS= read -r -d '' f; do
  rel="${f#"$runout"/}"
  base="${rel##*/}"
  ext="${base##*.}"
  if [ "$ext" != "$base" ] && [[ " $extensions" == *" $ext "* ]]; then
    kept+=("$rel")
  else
    skipped+=("$rel")
  fi
done < <(find "$runout" -type f -print0 | sort -z)
if [ "${#kept[@]}" -eq 0 ]; then
  echo "✗ the run wrote no file with an extension in EXTENSIONS ($extensions) under $runout; $refdir is left as it was" >&2
  exit 1
fi
rm -rf "$refdir"
mkdir -p "$refdir"
for rel in "${kept[@]}"; do
  mkdir -p "$refdir/$(dirname "$rel")"
  cp "$runout/$rel" "$refdir/$rel"
done
if [ "${#skipped[@]}" -gt 0 ]; then
  echo "not copied (extension not in EXTENSIONS: $extensions):"
  printf '  %s\n' "${skipped[@]}"
fi

if [ -d "$runout/checkpoints" ] && [[ "$name" == checkpoint* ]]; then
  ckptdir="$testdir/checkpoints${name#checkpoint}"
  rm -rf "$ckptdir"
  cp -R "$runout/checkpoints" "$ckptdir"
  find "$ckptdir" -name profiling.json -delete
  echo "✓ recovery input $ckptdir"
fi

if [ -n "$sibling" ]; then
  sibdir="$sibling"
  [ -d "$sibdir" ] || sibdir="$testdir/$sibling"
  if [ ! -d "$sibdir" ]; then
    echo "✗ no such sibling reference: $sibling" >&2
    exit 1
  fi
  echo "--- diff -rq $refdir $sibdir ---"
  if diff -rq "$refdir" "$sibdir"; then
    echo "(identical)"
  fi
fi

echo "✓ reference $refdir (run ./agent_scripts/test.sh before checking it in)"
