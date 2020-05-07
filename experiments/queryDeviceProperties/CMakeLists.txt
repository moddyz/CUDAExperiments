cpp_program(
    cudaQueryDeviceProperties
    LIBRARIES
        CUDA::cudart
        cudaUtils
    CPPFILES
        main.cpp
)
