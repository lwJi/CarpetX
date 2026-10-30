# CarpetX sandbox template for Docker Sandboxes (sbx)

This directory contains the complete, committed recipe for a Claude sandbox template in which `agent_scripts/build.sh` and `agent_scripts/test.sh` work exactly like on the host. The image is reproducible from this repository alone.

## Contents

| File | Purpose |
| --- | --- |
| `template.dockerfile` | Sandbox template image: default claude sandbox base + ETK dependency stack + pristine Cactus tree |
| `ubuntu-arm64.cfg` | Cactus option list for the image (arm64 port of `scripts/actions-cpu-real64.cfg`) |
| `carpetx.th` | Component/thorn list (copy of `~/Tools/ETK-Compile-Guides/ThornList/carpetx.th` — keep in sync) |
| `Compile-ETK` | Vendored copy of `~/bin/Compile-ETK` (uses `make`, auto-creates a missing config, passes `THORNLIST=` at configure) |
| `setup.sh` | One-time in-sandbox setup: links this repo into the Cactus tree, builds AMReX from the mounted source |

## Building the image

One dockerfile, starting from the maintained claude sandbox base image (`docker/sandbox-templates:claude-code-docker`), so the sbx provisioning (agent user, Claude Code, docker-in-sandbox, persistent-env plumbing) stays current with sbx releases:

```bash
cd <CarpetX>/agent_scripts/sandbox
docker build -f template.dockerfile -t lwji/sandbox-templates:claude-carpetx .
```

The from-scratch image build compiles ~8 libraries from source (~10 min on an Apple-silicon host: dependency layer ~8 min, Cactus checkout ~1 min). Rebuilds after edits to this directory are fast (the dependency layers are cached).

sbx has its own image store: it pulls `-t` images from a registry, **not** from the local docker daemon. To use a locally built template without pushing to Docker Hub (`docker save` ~10 s for the 1.7 GB tar, `sbx template load` ~5 s):

```bash
docker save lwji/sandbox-templates:claude-carpetx -o /tmp/claude-carpetx.tar
sbx template load /tmp/claude-carpetx.tar
```

Caveat: if the tag also exists on Docker Hub, `sbx run`/`sbx create` pull the registry copy and silently overwrite a loaded template. If `lwji/sandbox-templates:claude-carpetx` is published, for local iteration either build under a registry-free tag (e.g. `lwji/sandbox-templates:claude-carpetx-wip`) or `docker push` the rebuilt image so the registry copy is the one you built.

## Running

```bash
cd ~/docker-workspace/repos/CarpetX
sbx run -t lwji/sandbox-templates:claude-carpetx --name claude-CarpetX claude . ../amrex:ro
```

Inside the sandbox, once per sandbox (idempotent):

```bash
agent_scripts/sandbox/setup.sh   # link repo into Cactus tree, build AMReX from ../amrex (~30 s)
```

then the usual

```bash
./agent_scripts/build.sh   # first build configures and compiles from scratch (~13 min); later builds are incremental (~10 s)
./agent_scripts/test.sh    # full testsuite (~25 s; 94 tests across 22 thorns)
```

Timings measured end to end in an sbx sandbox on an Apple-silicon host (18 vCPUs); the microVM gets all host CPUs by default, so it is roughly as fast as the host itself.

Named sandboxes persist across `sbx stop`/restart, so `configs/`, `exe/`, `TEST/`, and the AMReX install are cached across sessions and only the first build is slow. Remove with `sbx rm <name>` to start clean.

For non-interactive use (no attached agent), `sbx create` + `sbx exec` mirror the above:

```bash
sbx create -t lwji/sandbox-templates:claude-carpetx --name claude-CarpetX claude . ../amrex:ro
sbx exec -w "$PWD" claude-CarpetX agent_scripts/sandbox/setup.sh
sbx exec -w "$PWD" claude-CarpetX ./agent_scripts/build.sh
sbx exec -w "$PWD" claude-CarpetX ./agent_scripts/test.sh
```

## How it works

- sbx mounts workspaces at their **host paths** (`/Users/liwei/docker-workspace/repos/CarpetX` appears at the same path inside the microVM), and the agent runs as the non-root `agent` user (uid 1000, passwordless sudo) provided by the base image.
- The image adds the ETK dependency stack on top of the base, preserving the `/usr` (apt) vs `/usr/local` (source-built) split that `ubuntu-arm64.cfg` reads back — a port of `docker/carpetx-arm64v8-cpu.dockerfile` trimmed to what the sandbox build needs (see the header of `template.dockerfile` for what was dropped).
- The image bakes a pristine GetComponents `--shallow` Cactus checkout at `/home/agent/cactus/Cactus` (no `configs/`, so the baked tree cannot go stale relative to the option list) with `arrangements/CarpetX` symlinked to the mounted workspace, and sets `CACTUSX`, `ETKCFG`, `ETKTHORNLIST` in the environment; `build.sh` uses those instead of the host's `ETKGUIDE`.
- `setup.sh` builds AMReX out-of-source from the read-only `../amrex` mount into `/home/agent/cactus/amrex-lib` (which `ubuntu-arm64.cfg` points `AMREX_DIR` at). Without an amrex mount it falls back to the AMReX release baked into the image at `/usr/local` via a symlink.
