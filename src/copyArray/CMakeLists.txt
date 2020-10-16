cpp_executable(${EXECUTABLE_PREFIX}_copyArray

    LIBRARIES
        CUDA::cudart
        cudaTools
        cxxopts

    CPPFILES
        copyArray.cu
        main.cpp
)
