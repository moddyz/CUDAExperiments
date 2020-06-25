cpp_program(
    cudaMatrixArrayProduct

    LIBRARIES
        CUDA::cudart
        cubase
        gm
        cxxopts

    CPPFILES
        matrixArrayProduct.cu
        main.cpp

    DEFINES
        GM_CUDA_SUPPORT
)
