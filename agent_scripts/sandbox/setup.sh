#!/bin/bash

# One-time setup inside an sbx sandbox created from the CarpetX sandbox
# template (idempotent, safe to re-run):
#   1. point the baked Cactus tree's arrangements/CarpetX at this repo
#   2. build AMReX from the mounted (read-only) amrex workspace into
#      $HOME/cactus/amrex-lib; without an amrex mount, fall back to the
#      AMReX release baked into the image at /usr/local
#
# Usage: agent_scripts/sandbox/setup.sh [--force-amrex]

set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

if [ -z "${CACTUSX:-}" ]; then
  echo "✗ CACTUSX is unset (is this the sandbox image?)" >&2
  exit 1
fi
if [ ! -d "$CACTUSX" ]; then
  echo "✗ CACTUSX=$CACTUSX does not exist: the sandbox rootfs is missing the template's layers" >&2
  echo "  (sbx bug docker/sbx-releases#366; see 'Known issue' in agent_scripts/sandbox/README.md)" >&2
  exit 1
fi

# 1. Link this repo into the Cactus tree
arr="$CACTUSX/arrangements/CarpetX"
if [ "$(readlink "$arr" 2>/dev/null)" != "$repo" ]; then
  rm -rf "$arr"
  ln -s "$repo" "$arr"
fi
echo "✓ $arr -> $repo"

# 2. AMReX
amrex_src="${AMREX_SRC:-$(dirname "$repo")/amrex}"
amrex_lib="$HOME/cactus/amrex-lib"
amrex_build="$HOME/cactus/amrex-build"

if [ ! -d "$amrex_src" ]; then
  if [ ! -e "$amrex_lib" ]; then
    ln -s /usr/local "$amrex_lib"
  fi
  echo "✓ no amrex workspace mounted; using baked AMReX ($amrex_lib -> $(readlink "$amrex_lib" 2>/dev/null || echo "$amrex_lib"))"
  exit 0
fi

if [ -d "$amrex_lib/lib" ] && [ ! -L "$amrex_lib" ] && [ "${1:-}" != "--force-amrex" ]; then
  echo "✓ AMReX already installed in $amrex_lib (rerun with --force-amrex to rebuild)"
  exit 0
fi

echo "Building AMReX from $amrex_src ..."
[ -L "$amrex_lib" ] && rm "$amrex_lib"
rm -rf "$amrex_build" "$amrex_lib"
log="$HOME/cactus/amrex-build.log"

# Same options as the host's Install-AMReX; out-of-source build because
# the amrex workspace is mounted read-only
if (
  export CC=mpicc CXX=mpicxx FC=mpif90 F90=mpif90 &&
  cmake -B "$amrex_build" -S "$amrex_src" \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCMAKE_INSTALL_PREFIX="$amrex_lib" \
    -DAMReX_FORTRAN=OFF \
    -DAMReX_FORTRAN_INTERFACES=OFF \
    -DAMReX_OMP=ON \
    -DAMReX_PARTICLES=ON \
    -DAMReX_PRECISION=DOUBLE &&
  cmake --build "$amrex_build" -j"$(nproc)" --target install
) > "$log" 2>&1; then
  echo "✓ AMReX installed in $amrex_lib"
else
  echo "--- last 40 lines ---"
  tail -40 "$log"
  echo "✗ AMReX build failed — full log: $log" >&2
  exit 1
fi
