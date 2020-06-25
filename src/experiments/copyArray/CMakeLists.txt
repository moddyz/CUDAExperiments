cpp_program(
    cudaCopyArray

    LIBRARIES
        CUDA::cudart
        cubase
        cxxopts

    CPPFILES
        copyArray.cu
        main.cpp
)
