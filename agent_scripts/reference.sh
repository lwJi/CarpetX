#!/bin/bash

usage() {
  cat <<'EOF'
usage: agent_scripts/reference.sh <Thorn>/test/<name>.par [<sibling>]

Regenerates the checked-in reference output <Thorn>/test/<name>/ for one
test parfile, the way the testsuite would produce it:

  1. runs $CACTUSX/exe/cactus_carpetx on the parfile from
     $CACTUSX/TEST/carpetx/<Thorn>/, the directory `make carpetx-testsuite`
     runs it from, after removing that directory's <name>/, so relative
     paths in the parfile (IO::recover_dir above all) resolve as under
     agent_scripts/test.sh;
  2. replaces <Thorn>/test/<name>/ with the run's files whose extension is
     in the thorn's EXTENSIONS (test.ccl, default tsv), the files the
     testsuite compares, and lists the files it did not copy; a file the
     run no longer writes therefore shows up as a deletion in git status;
  3. when the run leaves <name>/checkpoints/ and <name> is
     checkpoint<suffix>, replaces <Thorn>/test/checkpoints<suffix>/ with it,
     minus every profiling.json: that directory is the recovery input of
     recover<suffix>.par;
  4. with <sibling>, a reference directory of the same family (a name under
     <Thorn>/test/ or a path), prints the file-set difference between the
     new reference and the sibling and, for every *.tsv present in both,
     whether they are identical.

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
exe="$CACTUSX/exe/cactus_carpetx"
if [ ! -x "$exe" ]; then
  echo "✗ missing executable: $exe (run ./agent_scripts/build.sh)" >&2
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

rundir="$CACTUSX/TEST/carpetx/$thorn"
runout="$rundir/$name"
log="$rundir/$name.reference.log"
mkdir -p "$rundir"
rm -rf "$runout"
if ! (cd "$rundir" && "$exe" "$parfile") > "$log" 2>&1; then
  echo "--- last 40 lines ---"
  tail -40 "$log"
  echo "✗ $name.par failed — full log: $log" >&2
  exit 1
fi
if [ ! -d "$runout" ]; then
  echo "✗ the run left no $runout; does $name.par set IO::out_dir = \$parfile? — full log: $log" >&2
  exit 1
fi

refdir="$testdir/$name"
rm -rf "$refdir"
mkdir -p "$refdir"
skipped=()
while IFS= read -r -d '' f; do
  rel="${f#"$runout"/}"
  ext="${rel##*.}"
  if [ "$ext" != "$rel" ] && [[ " $extensions" == *" $ext "* ]]; then
    mkdir -p "$refdir/$(dirname "$rel")"
    cp "$f" "$refdir/$rel"
  else
    skipped+=("$rel")
  fi
done < <(find "$runout" -type f -print0 | sort -z)
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
  echo "--- file set: $refdir vs $sibdir ---"
  if diff <(cd "$refdir" && ls -R) <(cd "$sibdir" && ls -R); then
    echo "(same file set)"
  fi
  same=0; differ=0
  while IFS= read -r -d '' f; do
    rel="${f#"$refdir"/}"
    [ -f "$sibdir/$rel" ] || continue
    if cmp -s "$f" "$sibdir/$rel"; then
      same=$((same + 1))
    else
      differ=$((differ + 1))
      echo "differs: $rel"
    fi
  done < <(find "$refdir" -type f -name '*.tsv' -print0 | sort -z)
  echo "*.tsv present in both: $same identical, $differ differ"
fi

echo "✓ reference $refdir (run ./agent_scripts/test.sh before checking it in)"
