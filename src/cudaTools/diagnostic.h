#pragma once

/// \file diagnostic.h
///
/// A set of useful utilities for diagnostic.handling in CUDA programming.

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

/// \macro CUDA_CHECK
///
/// Check the error status.  On non-success, log to stderr and exit the program.
#define CUDA_CHECK( val ) CudaCheckError( ( val ), #val, __FILE__, __LINE__ )

/// Not intended to be used directly - use \ref CUDA_CHECK.
cudaError_t CudaCheckError( cudaError_t i_error, const char* i_function, const char* i_file, int i_line )
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
        return i_error;
    }
}

/// \def ASSERT( expr )
///
/// Asserts that the expression \p expr is \em true in debug builds. If \p expr evalutes \em false,
/// an error message will be printed with contextual information including the failure site.
///
/// In release builds, this is compiled out.
#define ASSERT( expr )                                                                                             \
    if ( !( expr ) )                                                                                                   \
    {                                                                                                                  \
        _Assert( #expr, __FILE__, __LINE__ );                                                                  \
    }

/// Not intended to be used directly, \ref ASSERT instead.
inline void _Assert( const char* i_expression, const char* i_file, size_t i_line )
{
    fprintf( stderr, "Assertion failed for expression: %s, at %s:%lu\n", i_expression, i_file, i_line );
}

