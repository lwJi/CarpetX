#!/bin/bash

if [ -z "${CACTUSX:-}" ]; then
  echo "✗ missing required env var: CACTUSX" >&2
  exit 1
fi

log="$CACTUSX/last-test.log"
summary="$CACTUSX/TEST/carpetx/summary.log"

# PROMPT=no makes RunTest.pl take the default answer for every prompt,
# so the testsuite runs non-interactively regardless of TTY
if ! (
  cd "$CACTUSX" &&
  make carpetx-testsuite PROMPT=no
) > "$log" 2>&1; then
  echo "--- last 40 lines ---"
  tail -40 "$log"
  echo "✗ testsuite run failed — full log: $log" >&2
  exit 1
fi

if [ -f "$summary" ] && grep -Eq 'Number failed *-> *0' "$summary"; then
  # The parfiles under <Thorn>/test/errors must abort with the text of
  # their .expect sidecar; the testsuite cannot express that
  repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
  if ! python3 "$repo/scripts/expect-error.py" "$CACTUSX/exe/cactus_carpetx" "$repo"/*/test/errors; then
    echo "✗ test: an expected-failure parfile did not fail as expected" >&2
    exit 1
  fi
  echo "✓ test"
  exit 0
fi

# List failed tests as Thorn/test from the "Tests failed:" section
failed=$(awk '/Tests failed:/{f=1;next} f && /^=+/{exit} f && NF {gsub(/[()]/,""); print $3 "/" $1}' "$summary" 2>/dev/null)
if [ -z "$failed" ]; then
  echo "--- last 40 lines ---"
  tail -40 "$log"
  echo "✗ test failed (could not parse $summary) — full log: $log" >&2
  exit 1
fi

n=$(printf '%s\n' "$failed" | grep -c .)
echo "✗ test: $n failed"
printf '%s\n' "$failed" | head -20
[ "$n" -gt 20 ] && echo "... and $((n - 20)) more (see $summary)"

# Bounded excerpts for the first few failures; read the full per-test
# logs at $CACTUSX/TEST/carpetx/<Thorn>/<test>.log for more
printf '%s\n' "$failed" | head -3 | while IFS=/ read -r thorn test; do
  tlog="$CACTUSX/TEST/carpetx/$thorn/$test.log"
  diffs="$CACTUSX/TEST/carpetx/$thorn/$test.diffs"
  echo "--- $tlog (last 20 lines) ---"
  tail -20 "$tlog" 2>/dev/null || echo "(missing)"
  if [ -s "$diffs" ]; then
    echo "--- $diffs (first 15 lines) ---"
    head -15 "$diffs"
  fi
done

echo "✗ test failed — per-test logs: \$CACTUSX/TEST/carpetx/<Thorn>/<test>.log, run log: $log" >&2
exit 1
