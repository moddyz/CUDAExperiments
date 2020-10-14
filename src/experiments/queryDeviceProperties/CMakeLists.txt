cpp_executable(
    cudaQueryDeviceProperties
    LIBRARIES
        CUDA::cudart
        cubase
    CPPFILES
        main.cpp
)
