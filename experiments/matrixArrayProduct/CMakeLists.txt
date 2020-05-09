cpp_program(
    cudaMatrixArrayProduct

    LIBRARIES
        CUDA::cudart
        cudaExperiments
        cxxopts

    CPPFILES
        matrixArrayProduct.cu
        main.cpp
)
