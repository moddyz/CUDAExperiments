cpp_executable(${EXECUTABLE_PREFIX}_matrixArrayProduct

    LIBRARIES
        CUDA::cudart
        cudaTools
        gm
        cxxopts

    CPPFILES
        matrixArrayProduct.cu
        main.cpp

    DEFINES
        GM_CUDA_SUPPORT
)
