#!/bin/bash

if [ -z "${CACTUSX:-}" ]; then
  echo "✗ missing required env var: CACTUSX (the Cactus tree with this repo symlinked at arrangements/CarpetX)" >&2
  exit 1
fi

# Accept either the host interface (ETKGUIDE) or the sandbox interface
# (ETKCFG + ETKTHORNLIST); explicit ETKCFG/ETKTHORNLIST take precedence.
cfg="${ETKCFG:-${ETKGUIDE:+$ETKGUIDE/macos.cfg}}"
thornlist="${ETKTHORNLIST:-${ETKGUIDE:+$ETKGUIDE/../ThornList/carpetx.th}}"
if [ -z "$cfg" ] || [ -z "$thornlist" ]; then
  echo "✗ missing required env vars: set ETKGUIDE or ETKCFG+ETKTHORNLIST" >&2
  exit 1
fi

log="$CACTUSX/last-build.log"
if (
  cd "$CACTUSX" &&
  Compile-ETK -e carpetx -j8 \
    -c "$cfg" \
    -t "$thornlist"
) > "$log" 2>&1; then
  echo "✓ build $CACTUSX/exe/cactus_carpetx"
  repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
  python3 "$repo/scripts/lint" symbols "$CACTUSX/configs/carpetx/build/CarpetX"
else
  echo "--- first error lines ---"
  grep -n -m 12 -E 'error:|Error|\*\*\*' "$log" || echo "(no obvious error lines)"
  echo "--- last 40 lines ---"
  tail -40 "$log"
  echo "✗ build failed — full log: $log, build dir: $CACTUSX/configs/carpetx" >&2
  exit 1
fi
