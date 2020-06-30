# CUDAExperiments

Experiments with NVIDIA's CUDA API.

## Table of Contents

- [Documentation](#documentation)
- [Building](#building)
  - [Requirements](#requirements)
- [Build Status](#build-status)

See [cmake/macros/Public.cmake](cmake/macros/Public.cmake) for the full listing.

## Documentation

Documentation based on the latest state of master, [hosted by GitHub Pages](https://moddyz.github.io/CUDAExperiments/).

## Building

A convenience build script is also provided, for building all targets, and optionally installing to a location:
```
./build.sh <OPTIONAL_INSTALL_LOCATION>
```

### Requirements

- `>= CMake-3.17`
- `>= C++17`
- `>= CUDA 10`

## Build Status

|       | master | 
| ----- | ------ | 
| macOS-10.14 | [![Build Status](https://travis-ci.com/moddyz/CUDAExperiments.svg?branch=master)](https://travis-ci.com/moddyz/CUDAExperiments) |
