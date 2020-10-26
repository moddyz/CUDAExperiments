#pragma once

/// \file error.h
///
/// A set of useful utilities for error handling in CUDA programming.

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

/// \macro CUDA_CHECK
///
/// Check the error status.  On non-success, log to stderr and exit the program.
#define CUDA_CHECK( val ) CudaCheckError( ( val ), #val, __FILE__, __LINE__ )

/// Not intended to be used directly - use \ref CUDA_CHECK.
void CudaCheckError( cudaError_t i_error, const char* i_function, const char* i_file, int i_line )
{
    if ( i_error != cudaSuccess )
    {
        fprintf( stderr,
                 "CUDA error at %s:%d code=%d(%s) \"%s\" \n",
                 i_file,
                 i_line,
                 static_cast<unsigned int>( i_error ),
                 cudaGetErrorName( i_error ),
                 i_function );
        exit( EXIT_FAILURE );
    }
}
