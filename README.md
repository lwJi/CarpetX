# [CarpetX](https://github.com/eschnett/CarpetX)

<img
src="https://github.com/eschnett/CarpetX/blob/main/figures/carpetx.png"
width="200" />

**CarpetX** is a [Cactus](https://cactuscode.org/) driver based on
[AMReX](https://amrex-codes.github.io), a software framework for
block-structured AMR (adaptive mesh refinement). CarpetX is intended
for the [Einstein Toolkit](https://einsteintoolkit.org/).

* [![GitHub
  CI](https://github.com/eschnett/CarpetX/workflows/CI/badge.svg)](https://github.com/eschnett/CarpetX/actions)

## Overview

CarpetX is ready for production. You are welcome to give it a try, to look at what changes your code might need to benefit from CarpetX's new features, and to give us feedback.

The recorded talk "[Using CarpetX: A Guide for Early Adopters](http://einsteintoolkit.org/seminars/2021_03_18/index.html)". This presentation provides an overview of the current capabilities of CarpetX and showcases how to write Cactus code using it.

## Getting started

Instructions for downloading the Einstein Toolkit including CarpetX, building, and running an example are available on the [Wiki](https://github.com/eschnett/CarpetX/wiki/Getting-Started).

## Subcycling

The initial subcycling implementation applied prolongation to time derivatives rather than the state vector itself, causing issues during hydrodynamic evolution. This bug was identified by Jay Kalinani (jaykalinani@gmail.com) and fixed in [PR #114](https://github.com/lwJi/CarpetX/pull/114). The current codebase features a reimplementation of this fix.
