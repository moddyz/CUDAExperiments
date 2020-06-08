cpp_program(
    cudaMatrixArrayProduct

    LIBRARIES
        CUDA::cudart
        cudaExperiments
        gm
        cxxopts

    CPPFILES
        matrixArrayProduct.cu
        main.cpp

    DEFINES
        GM_CUDA_SUPPORT
)
