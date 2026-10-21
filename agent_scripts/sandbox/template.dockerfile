# Claude sandbox template for CarpetX development with Docker Sandboxes (sbx).
#
# Layers on top of the default claude sandbox image:
#   - the Einstein Toolkit dependency stack, preserving the /usr (apt) vs
#     /usr/local (source-built) split that the option list (ubuntu-arm64.cfg)
#     reads back
#   - a pristine Cactus checkout at /home/agent/cactus/Cactus whose
#     arrangements/CarpetX points at the mounted CarpetX workspace
#   - Compile-ETK, option list, and thornlist (ENV CACTUSX/ETKCFG/ETKTHORNLIST)
#     so agent_scripts/build.sh and test.sh work unchanged
#
# The dependency layer is a port of docker/carpetx-arm64v8-cpu.dockerfile
# (Ubuntu noble) to the base image's Ubuntu (26.04 "resolute", GCC 15,
# CMake 4), trimmed to what the sandbox build needs:
#   - dropped source builds: HPCToolkit, ASDF (asdf-cxx), RePrimAnd, Conduit,
#     SimulationIO (not referenced by the sandbox option list or thornlist)
#   - dropped apt packages: libpetsc-real-dev (PDESolvers/PETSc are commented
#     out of the thornlist), libpapi-dev/papi-tools (profiling, CI-only)
#   - added source build: yaml-cpp into /usr/local (the option list points
#     YAML_CPP_DIR at /usr/local); apt's libyaml-cpp-dev is omitted so there
#     is exactly one copy
#
# Build (context = this directory):
#     docker build -f template.dockerfile -t lwji/sandbox-templates:claude-carpetx .
# The sandbox runtime does not share the host docker image store, so hand the
# image to sbx directly (or push to Docker Hub):
#     docker save lwji/sandbox-templates:claude-carpetx -o /tmp/claude-carpetx.tar
#     sbx template load /tmp/claude-carpetx.tar
# Run (from the CarpetX repo root):
#     sbx run -t lwji/sandbox-templates:claude-carpetx --name claude-CarpetX claude . ../amrex:ro

# The maintained claude sandbox base: agent user (uid 1000), Claude Code,
# docker-in-sandbox, persistent-env plumbing, tini entrypoint.
FROM docker/sandbox-templates:claude-code-docker

USER root

# Build-time only; do not bake DEBIAN_FRONTEND into the final image
ARG DEBIAN_FRONTEND=noninteractive

ENV LANGUAGE=en_US.en \
    LANG=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8

WORKDIR /var/tmp

# Install system packages
# - Boost on Ubuntu requires OpenMPI
RUN apt-get update && \
    apt-get --yes --no-install-recommends install \
        bzip2 \
        ca-certificates \
        clang-format \
        cmake \
        curl \
        cvs \
        diffutils \
        elfutils \
        g++ \
        gcc \
        gdb \
        gfortran \
        git \
        hdf5-filter-plugin \
        hdf5-filter-plugin-blosc-serial \
        hdf5-filter-plugin-zfp-serial \
        hdf5-plugin-lzf \
        hdf5-tools \
        hwloc-nox \
        language-pack-en \
        less \
        libblosc-dev \
        libblosc2-dev \
        libboost-all-dev \
        libbz2-dev \
        libfftw3-dev \
        libgit2-dev \
        libgsl-dev \
        libhdf5-dev \
        libhwloc-dev \
        libiberty-dev \
        liblz4-dev \
        liblzma-dev \
        libopenblas-dev \
        libopenmpi-dev \
        libprotobuf-dev \
        libreadline-dev \
        libtool \
        libudev-dev \
        libzfp-dev \
        libzstd-dev \
        locales \
        m4 \
        make \
        meson \
        ninja-build \
        nlohmann-json3-dev \
        numactl \
        patch \
        perl \
        pkgconf \
        protobuf-compiler \
        python3 \
        python3-numpy \
        python3-pip \
        python3-requests \
        rsync \
        subversion \
        tar \
        vim \
        wget \
        xz-utils \
        zlib1g-dev \
        zstd \
        && \
    rm -rf /var/lib/apt/lists/*

# CMake 4 (in this base) refuses to configure projects whose
# cmake_minimum_required is < 3.5; CMAKE_POLICY_VERSION_MINIMUM=3.5 is the
# documented escape hatch and is a no-op for projects that already require
# >= 3.5.  It is passed to the older source builds below.

# Install MGARD
# MGARD is a lossy compression library (linked into ADIOS2)
RUN mkdir src && \
    (cd src && \
    wget https://github.com/CODARcode/MGARD/archive/refs/tags/1.5.2.tar.gz && \
    tar xzf 1.5.2.tar.gz && \
    cd MGARD-1.5.2 && \
    cmake -B build -G Ninja \
        -DBUILD_TESTING=OFF \
        -DCMAKE_BUILD_TYPE=RelWithDebInfo \
        -DCMAKE_INSTALL_PREFIX=/usr/local \
        -DCMAKE_PREFIX_PATH=/usr/local \
        -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
        -DMGARD_ENABLE_OPENMP=ON \
        -DMGARD_ENABLE_SERIAL=ON \
        && \
    cmake --build build && \
    cmake --install build && \
    true) && \
    rm -rf src

# Install yaml-cpp
# yaml-cpp reads and writes YAML files (a hard CarpetX dependency); built from
# source into /usr/local, where the option list expects it.  Version 0.9.0:
# 0.8.0 and the copy vendored by ADIOS2 do not compile with GCC 15 (missing
# <cstdint> includes).  Built before ADIOS2, which links it as an external
# dependency.
RUN mkdir src && \
    (cd src && \
    wget https://github.com/jbeder/yaml-cpp/archive/refs/tags/yaml-cpp-0.9.0.tar.gz && \
    tar xzf yaml-cpp-0.9.0.tar.gz && \
    cd yaml-cpp-yaml-cpp-0.9.0 && \
    cmake -B build -G Ninja \
        -DCMAKE_BUILD_TYPE=RelWithDebInfo \
        -DCMAKE_INSTALL_PREFIX=/usr/local \
        -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
        -DYAML_BUILD_SHARED_LIBS=ON \
        -DYAML_CPP_BUILD_TESTS=OFF \
        && \
    cmake --build build && \
    cmake --install build && \
    true) && \
    rm -rf src

# Install ADIOS2
# ADIOS2 is a parallel I/O library, comparable to HDF5
# - depends on blosc2
# - depends on MGARD
# - depends on yaml-cpp (external: the vendored copy fails with GCC 15)
# - the HDF5 VOL connector is disabled: it needs parallel HDF5, but this image
#   carries apt's serial HDF5 (on noble it was auto-disabled by HDF5 < 1.14)
RUN mkdir src && \
    (cd src && \
    wget https://github.com/ornladios/ADIOS2/archive/refs/tags/v2.10.2.tar.gz && \
    tar xzf v2.10.2.tar.gz && \
    cd ADIOS2-2.10.2 && \
    cmake -B build -G Ninja \
        -DCMAKE_BUILD_TYPE=RelWithDebInfo \
        -DCMAKE_INSTALL_PREFIX=/usr/local \
        -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
        -DBUILD_SHARED_LIBS=ON \
        -DBUILD_TESTING=OFF \
        -DADIOS2_BUILD_EXAMPLES=OFF \
        -DADIOS2_Blosc2_PREFER_SHARED=ON \
        -DADIOS2_USE_BZip2=ON \
        -DADIOS2_USE_Blosc2=ON \
        -DADIOS2_USE_EXTERNAL_YAMLCPP=ON \
        -DADIOS2_USE_Fortran=OFF \
        -DADIOS2_USE_HDF5=ON \
        -DADIOS2_USE_HDF5_VOL=OFF \
        -DADIOS2_USE_MGARD=ON \
        -DADIOS2_USE_ZFP=ON \
        && \
    cmake --build build && \
    cmake --install build && \
    true) && \
    rm -rf src

# Install NSIMD
# NSIMD allows writing explicitly SIMD-vectorized code
RUN mkdir src && \
    (cd src && \
    wget https://github.com/agenium-scale/nsimd/archive/refs/tags/v3.0.1.tar.gz && \
    tar xzf v3.0.1.tar.gz && \
    cd nsimd-3.0.1 && \
    cmake -B build -G Ninja \
        -DCMAKE_BUILD_TYPE=RelWithDebInfo \
        -DCMAKE_INSTALL_PREFIX=/usr/local \
        -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
        -Dsimd=aarch64 \
        && \
    cmake --build build && \
    cmake --install build && \
    true) && \
    rm -rf src

# Install openPMD-api
# openPMD-api defines a standard for laying out AMR data in a file
# - depends on ADIOS2
RUN mkdir src && \
    (cd src && \
    wget https://github.com/openPMD/openPMD-api/archive/refs/tags/0.16.1.tar.gz && \
    tar xzf 0.16.1.tar.gz && \
    cd openPMD-api-0.16.1 && \
    cmake -B build -G Ninja \
        -DCMAKE_BUILD_TYPE=RelWithDebInfo \
        -DCMAKE_INSTALL_PREFIX=/usr/local \
        -DBUILD_EXAMPLES=OFF \
        -DBUILD_TESTING=OFF \
        -DopenPMD_BUILD_SHARED_LIBS=ON \
        -DopenPMD_USE_MPI=ON \
        && \
    cmake --build build && \
    cmake --install build && \
    true) && \
    rm -rf src

# Install Silo
# Silo defines a standard for laying out AMR data in a file
# - GCC 15 defaults C to C23, which Silo's old C is not ready for; force gnu17
RUN mkdir src && \
    (cd src && \
    wget https://github.com/LLNL/Silo/releases/download/4.11.1/silo-4.11.1.tar.xz && \
    tar xJf silo-4.11.1.tar.xz && \
    cd silo-4.11.1 && \
    mkdir build && \
    cd build && \
    ../configure \
        CFLAGS='-g -O2 -std=gnu17' \
        --disable-fortran \
        --disable-static \
        --enable-optimization \
        --enable-shared \
        --with-hdf5=/usr/lib/aarch64-linux-gnu/hdf5/serial/include,/usr/lib/aarch64-linux-gnu/hdf5/serial/lib \
        --prefix=/usr/local \
        && \
    make -j$(nproc) && \
    make -j$(nproc) install && \
    true) && \
    rm -rf src

# Install ssht
# ssht provides spin-weighted spherical harmonics
RUN mkdir src && \
    (cd src && \
    wget https://github.com/astro-informatics/ssht/archive/v1.5.2.tar.gz && \
    tar xzf v1.5.2.tar.gz && \
    cd ssht-1.5.2 && \
    cmake -B build -G Ninja \
        -DCMAKE_BUILD_TYPE=RelWithDebInfo \
        -DCMAKE_INSTALL_PREFIX=/usr/local \
        -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
        -DBUILD_TESTING=OFF \
        && \
    cmake --build build && \
    cmake --install build && \
    true) && \
    rm -rf src

ARG real_precision=real64

# Install AMReX
# AMReX provides adaptive mesh refinement
# - this is the fallback release used when no ../amrex workspace is mounted;
#   setup.sh builds the mounted checkout into /home/agent/cactus/amrex-lib
# - install this last because it changes most often
RUN mkdir src && \
    (cd src && \
    wget https://github.com/AMReX-Codes/amrex/archive/26.07.tar.gz && \
    tar xzf 26.07.tar.gz && \
    cd amrex-26.07 && \
    case $real_precision in \
        real32) precision=SINGLE;; \
        real64) precision=DOUBLE;; \
        *) exit 1;; \
    esac && \
    cmake -B build -G Ninja \
        -DCMAKE_BUILD_TYPE=RelWithDebInfo \
        -DCMAKE_INSTALL_PREFIX=/usr/local \
        -DBUILD_SHARED_LIBS=ON \
        -DAMReX_FORTRAN=OFF \
        -DAMReX_FORTRAN_INTERFACES=OFF \
        -DAMReX_OMP=ON \
        -DAMReX_PARTICLES=ON \
        -DAMReX_PRECISION="$precision" \
        && \
    cmake --build build && \
    cmake --install build && \
    true) && \
    rm -rf src

# Find libraries in /usr/local/lib64
RUN echo /usr/local/lib64 >/etc/ld.so.conf.d/usr-local-lib64.conf && \
    ldconfig

# --- Cactus layer ---

# Build driver used by agent_scripts/build.sh
COPY Compile-ETK /usr/local/bin/Compile-ETK
RUN chmod 755 /usr/local/bin/Compile-ETK

# Thornlist referenced by ETKTHORNLIST below (the option list is copied
# after the checkout so that cfg tweaks do not re-trigger GetComponents)
COPY --chown=agent:agent carpetx.th /home/agent/etk/

# Return to the base image's user (the entrypoint and cmd are inherited
# unchanged) so sbx provisioning keeps working.
USER agent

# Formaline commits the source tree to a git repository during the Cactus
# build and needs a git identity
RUN git config --global user.name "Liwei Ji" && \
    git config --global user.email "jiliwei.phys@gmail.com"

# Pristine Cactus tree (no configs/, so the baked tree cannot go stale
# relative to the option list): flesh + support thorns via GetComponents;
# arrangements/CarpetX is replaced by a symlink to the CarpetX workspace,
# which resolves at run time because sbx mounts workspaces at their host
# paths (setup.sh re-links it if the workspace lives elsewhere).  The first
# in-sandbox build configures from scratch; the named sandbox persists
# configs/, exe/, and TEST/ across stop/start.
ARG CARPETX_WORKSPACE=/Users/liwei/docker-workspace/repos/CarpetX
RUN mkdir -p /home/agent/cactus && \
    cd /home/agent/cactus && \
    curl -fsSLO https://raw.githubusercontent.com/gridaphobe/CRL/master/GetComponents && \
    chmod +x GetComponents && \
    ./GetComponents --no-parallel --shallow /home/agent/etk/carpetx.th && \
    rm -rf Cactus/arrangements/CarpetX && \
    ln -s "$CARPETX_WORKSPACE" Cactus/arrangements/CarpetX

# Option list referenced by ETKCFG below
COPY --chown=agent:agent ubuntu-arm64.cfg /home/agent/etk/

# The sandbox interface for agent_scripts/build.sh and test.sh
ENV CACTUSX=/home/agent/cactus/Cactus \
    ETKCFG=/home/agent/etk/ubuntu-arm64.cfg \
    ETKTHORNLIST=/home/agent/etk/carpetx.th

WORKDIR /home/agent/workspace
