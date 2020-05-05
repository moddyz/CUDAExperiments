# Get the current source directory.
set(CUDAUTILS_INCLUDE_DIR
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Add cudaUtils as a library target.
add_library(
    cudaUtils
    IMPORTED  # This is external.
    INTERFACE # This library target does not provide source files.  (Header only!)
    GLOBAL    # Make this library target available in directories above this one.
)

# Any target which links against "cudaUtils" will inherit
# CUDAUTILS_INCLUDE_DIR as a compilation include directory.
target_include_directories(
    cudaUtils
    INTERFACE ${CUDAUTILS_INCLUDE_DIR}
)
