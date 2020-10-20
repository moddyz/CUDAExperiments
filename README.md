<a href="https://github.com/moddyz/CUDASandbox/actions?query=workflow%3A%22Build+and+test%22"><img src="https://github.com/moddyz/CUDASandbox/workflows/Build%20and%20test/badge.svg"/></a>

# CUDASandbox

Sandbox for experiments with NVIDIA's CUDA API.

## Table of Contents

- [Dependencies](#dependencies)
- [Building](#building)

### Dependencies

The following dependencies are mandatory:
- C++ compiler (C++17)
- [CMake](https://cmake.org/documentation/) (>=3.12)
- [CUDA](https://developer.nvidia.com/cuda-toolkit) (>=10)

### Building

Example snippet for building this project:
```
mkdir build && cd build
cmake \
  -DCMAKE_CUDA_COMPILER="/usr/local/cuda/bin/nvcc" \
  -DCMAKE_INSTALL_PREFIX="/apps/CUDASandbox/" \
  ..
cmake --build  . -- VERBOSE=1 -j8 install
```

CMake options for configuring this project:

| CMake Variable name     | Description                                                            | Default |
| ----------------------- | ---------------------------------------------------------------------- | ------- |
| `BUILD_TESTING`         | Enable automated testing.                                              | `OFF`   |
