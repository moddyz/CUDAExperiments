cpp_program(
    cudaQueryDeviceProperties
    LIBRARIES
        CUDA::cudart
        cudaExperiments
    CPPFILES
        main.cpp
)
