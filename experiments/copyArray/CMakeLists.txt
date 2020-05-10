cpp_program(
    cudaCopyArray

    LIBRARIES
        CUDA::cudart
        cudaExperiments
        cxxopts

    CPPFILES
        copyArray.cu
        main.cpp
)
