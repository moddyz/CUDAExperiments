cpp_program(
    cudaQueryDeviceProperties
    LIBRARIES
        CUDA::cudart
        cubase
    CPPFILES
        main.cpp
)
