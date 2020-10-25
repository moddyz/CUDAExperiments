get_filename_component(EXECUTABLE_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
cpp_executable(${EXECUTABLE_PREFIX}_${EXECUTABLE_NAME}
    LIBRARIES
        CUDA::cudart
        cudaTools

    CPPFILES
        main.cu
)
