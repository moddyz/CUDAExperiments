file(GLOB CPPFILES *.cpp)
cpp_test(test_${ROOT_PREFIX}_${LIBRARY_SUFFIX}
    CPPFILES
        ${CPPFILES}
    LIBRARIES
        CUDA::cudart
)
