cmake_minimum_required(VERSION 3.17)

# Project Name
project(
    CUDASandbox
    VERSION 0.0
    DESCRIPTION "Experimentation with CUDA."
    LANGUAGES CXX CUDA
)

# Set module search paths.
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
                      ${CMAKE_CURRENT_SOURCE_DIR}/cmake/
                      ${CMAKE_CURRENT_SOURCE_DIR}/cmake/tools)
include(Options)
include(Packages)
include(Tools)
include(Defaults)

# Import third party libraries.
add_subdirectory(thirdparty)

# Build source code.
add_subdirectory(src)
