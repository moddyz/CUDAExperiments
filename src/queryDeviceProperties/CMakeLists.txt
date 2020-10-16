cpp_executable(${EXECUTABLE_PREFIX}_queryDeviceProperties

    LIBRARIES
        CUDA::cudart
        cudaTools

    CPPFILES
        main.cpp
)
