cpp_executable(
    cudaCopyArray

    LIBRARIES
        CUDA::cudart
        cubase
        cxxopts

    CPPFILES
        copyArray.cu
        main.cpp
)
